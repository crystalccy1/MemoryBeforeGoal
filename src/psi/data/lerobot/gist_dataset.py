"""Verified, memory-mapped common feature bank for Mean-Fixed/All/Learned-All."""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import torch
from safetensors import safe_open

from psi.data.lerobot.framesamp_features import sha256_file
from psi.data.lerobot.gist_features import GIST_BANK_SCHEMA, assemble_gist_inputs


@lru_cache(maxsize=16)
def _verify_outputs(manifest_path: str, expected_sha: str):
    path = Path(manifest_path)
    if sha256_file(path) != expected_sha:
        raise RuntimeError("Gist manifest changed before verification")
    result = {}
    for row in json.loads(path.read_text())["outputs"]:
        shard = path.parent / row["relative_path"]
        if shard.is_symlink() or not shard.is_file():
            raise RuntimeError(f"Missing/unsafe Gist shard: {shard}")
        if not shard.resolve().is_relative_to((path.parent / "episodes").resolve()):
            raise RuntimeError("Gist shard escapes its episode directory")
        stat = shard.stat()
        if stat.st_size != row["bytes"] or sha256_file(shard) != row["sha256"]:
            raise RuntimeError(f"Gist shard checksum mismatch: {shard}")
        result[int(row["episode_index"])] = (stat.st_size, stat.st_mtime_ns, stat.st_ino)
    return result


class GistFeatureBank:
    def __init__(self, root, *, expected_manifest_sha256, expected_task,
                 expected_dataset_revision):
        self.root = Path(root).resolve(strict=True)
        for name in ("manifest.json", "READY", "source_contract.json", "source_manifest.json",
                     "source_parity_receipt.json", "source_PARITY_READY"):
            path = self.root / name
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"Missing/unsafe Gist release file: {path}")
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        ready = json.loads((self.root / "READY").read_text())
        manifest_sha = sha256_file(self.root / "manifest.json")
        if manifest_sha != expected_manifest_sha256 or ready.get("manifest_sha256") != manifest_sha:
            raise RuntimeError("Gist bank does not match its pinned READY manifest")
        if (self.manifest.get("schema_version") != GIST_BANK_SCHEMA
                or ready.get("schema_version") != GIST_BANK_SCHEMA
                or self.manifest.get("status") != "complete"
                or self.manifest.get("scope") != "full"
                or self.manifest.get("task") != expected_task
                or self.manifest.get("dataset_revision") != expected_dataset_revision):
            raise RuntimeError("Gist bank schema/task/dataset/scope mismatch")
        if sha256_file(self.root / "source_contract.json") != self.manifest["source_feature_contract_sha256"]:
            raise RuntimeError("Gist source feature contract changed")
        if sha256_file(self.root / "source_manifest.json") != self.manifest["source_feature_manifest_sha256"]:
            raise RuntimeError("Gist source feature manifest changed")
        if sha256_file(self.root / "source_parity_receipt.json") != self.manifest["source_parity_receipt_sha256"]:
            raise RuntimeError("Gist source parity receipt changed")
        source_manifest = json.loads((self.root / "source_manifest.json").read_text())
        self.source_contract = json.loads((self.root / "source_contract.json").read_text())
        if (self.source_contract.get("task") != expected_task
                or self.source_contract["dataset"].get("revision") != expected_dataset_revision):
            raise RuntimeError("Gist source contract is bound to another task/dataset")
        # Retain the familiar source contract API for checkpoint/producer checks.
        self.contract = self.source_contract
        producer = Path(__file__).resolve().parents[2] / "models/framesamp_qwen.py"
        producer_record = self.source_contract["code_files"]["shared_offline_online_producer"]
        if sha256_file(producer) != producer_record["sha256"]:
            raise RuntimeError("Gist bank's offline and current online Qwen producers differ")
        rows = self.manifest["outputs"]
        original_rows = {int(row["episode_index"]): row for row in source_manifest["outputs"]}
        self._outputs = {int(row["episode_index"]): row for row in rows}
        if (len(rows) != len(self._outputs) or len(rows) != self.manifest["episodes"]
                or sorted(self._outputs) != self.source_contract["dataset"]["selected_episode_indices"]
                or sum(int(row["raw_frames"]) for row in rows) != self.manifest["raw_frames"]
                or self.manifest["raw_frames"] != self.source_contract["dataset"]["frames"]
                or ready.get("episodes") != self.manifest["episodes"]
                or ready.get("raw_frames") != self.manifest["raw_frames"]):
            raise RuntimeError("Gist bank is missing full-dataset episodes/frames")
        for row in rows:
            original = original_rows.get(int(row["episode_index"]), {})
            if row["source_sha256"] != original.get("sha256") or row["raw_frames"] != original.get("source_frames"):
                raise RuntimeError("Gist episode provenance differs from the pinned original feature manifest")
        self.feature_dim = int(self.source_contract["encoder"]["feature_dim"])
        self._verified = _verify_outputs(str(self.root / "manifest.json"), manifest_sha)

    @lru_cache(maxsize=8)
    def _load_episode(self, episode_index):
        row = self._outputs[episode_index]
        path = self.root / row["relative_path"]
        stat = path.stat()
        if path.is_symlink() or (stat.st_size, stat.st_mtime_ns, stat.st_ino) != self._verified[episode_index]:
            raise RuntimeError(f"Gist shard changed after verification: {path}")
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            features = handle.get_tensor("features")
            raw = handle.get_tensor("raw_frame_indices")
            detail = handle.get_tensor("detail_raw_frame_indices")
        n = int(row["raw_frames"])
        if (metadata.get("schema_version") != GIST_BANK_SCHEMA
                or metadata.get("task") != self.manifest["task"]
                or metadata.get("episode_index") != str(episode_index)
                or metadata.get("source_sha256") != row["source_sha256"]
                or features.shape != (n, 16, self.feature_dim)
                or features.dtype != torch.bfloat16
                or raw.dtype != torch.int64 or not torch.equal(raw, torch.arange(n))
                or detail.shape != (n, 16) or detail.dtype != torch.int64):
            raise RuntimeError(f"Gist shard tensor/metadata mismatch: {path}")
        return features, raw, detail

    def memory_for_frame(self, episode_index: int, current_frame_index: int):
        episode_index, current_frame_index = int(episode_index), int(current_frame_index)
        if episode_index not in self._outputs:
            raise KeyError(f"Unknown Gist episode: {episode_index}")
        n = int(self._outputs[episode_index]["raw_frames"])
        if not 0 <= current_frame_index < n:
            raise ValueError("Gist current frame lies outside its episode")
        path = self.root / self._outputs[episode_index]["relative_path"]
        stat = path.stat()
        if path.is_symlink() or (stat.st_size, stat.st_mtime_ns, stat.st_ino) != self._verified[episode_index]:
            raise RuntimeError(f"Gist shard changed after verification: {path}")
        features, raw, detail = self._load_episode(episode_index)
        stop = current_frame_index + 1
        return assemble_gist_inputs(features[:stop], raw[:stop], detail_indices=detail[current_frame_index])

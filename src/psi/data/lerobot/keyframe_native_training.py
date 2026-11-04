"""Training adapter for immutable seven-frame, native-grid memory banks."""
from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
import warnings

import numpy as np
import torch
from safetensors import safe_open

from psi.config.transform_framesamp import FRAMESAMP_MEMORY_FIELDS
from psi.data.lerobot.framesamp_features import sha256_file
from psi.data.lerobot.keyframe_native import (
    FEATURE_DIM, GRID_HW, MAX_FRAMES, MEMORY_BUDGET, TOKENS_PER_FRAME,
    NativeKeyframeBank, bank_file, read_tensors, validate_selection,
)
from psi.trainers.qwen3vl_mixin import PaddedCollatorForTogether


def _mapped_features(path: Path) -> dict[str, torch.Tensor]:
    """Read a native feature shard with its [T,77,2048] bf16 "features" tensor memory-mapped, not copied.

    Values are identical to read_tensors. read_tensors copied the whole tensor into each data-loader worker's
    episode cache: a Lit-lamp shard is ~2.6 GB (7.5-11.5k frames), so 64 workers x 8 cached episodes exceeded the
    job memory and the copies made training data-bound (8-9 s/step). A mapped tensor costs no private memory, and
    indexing it copies only the selected frames. The small tensors are read as before.
    """
    with path.open("rb") as stream:
        header_len = int.from_bytes(stream.read(8), "little")
        header = json.loads(stream.read(header_len))
    entry = header["features"]
    begin, end = entry["data_offsets"]
    shape = tuple(int(v) for v in entry["shape"])
    if entry["dtype"] != "BF16" or end - begin != 2 * int(np.prod(shape)):
        raise ValueError(f"Native feature shard layout differs from bf16 {shape}: {path}")
    array = np.memmap(path, dtype=np.int16, mode="r", offset=8 + header_len + begin, shape=shape)
    with safe_open(path, framework="pt", device="cpu") as handle:
        tensors = {name: handle.get_tensor(name) for name in handle.keys() if name != "features"}
    with warnings.catch_warnings():   # read-only mapping: never written through
        warnings.simplefilter("ignore", UserWarning)
        tensors["features"] = torch.from_numpy(array).view(torch.bfloat16)
    return tensors


def _fingerprint(path: Path) -> tuple[int, int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ino


class NativeKeyframeTrainingBank(NativeKeyframeBank):
    """Check lineage once, then map verified shards without per-sample hashing."""

    def __init__(self, root, *, expected_manifest_sha256, expected_task,
                 expected_dataset_revision, selector="vcs"):
        super().__init__(root, expected_manifest_sha256=expected_manifest_sha256)
        if selector not in ("vcs", "uniform"):
            raise ValueError("Native keyframe selector must be vcs or uniform")
        self.selector = selector
        self.max_memory_frames, self.tokens_per_frame = MAX_FRAMES, TOKENS_PER_FRAME
        self.memory_budget, self.feature_dim = MEMORY_BUDGET, FEATURE_DIM
        self.source_contract = json.loads((self.root / "source_contract.json").read_text())
        self.source_manifest = json.loads((self.root / "source_manifest.json").read_text())
        self.contract = self.source_contract
        dataset = self.contract["dataset"]
        if (self.manifest.get("scope") != "full"
                or any(value.get("task") != expected_task for value in
                       (self.manifest, self.plan, self.source_contract, self.source_manifest))
                or self.manifest.get("dataset_revision") != expected_dataset_revision
                or self.plan.get("dataset_revision") != expected_dataset_revision
                or dataset.get("revision") != expected_dataset_revision
                or self.source_manifest.get("status") != "complete"
                or self.source_manifest.get("scope") != "full"
                or self.source_contract.get("scope") != "full"
                or sha256_file(self.root / "source_contract.json") != self.plan.get("source_contract_sha256")
                or sha256_file(self.root / "source_manifest.json") != self.plan.get("source_manifest_sha256")
                or self.source_manifest.get("contract_sha256") != self.plan.get("source_contract_sha256")):
            raise ValueError("Native training bank source task/revision/lineage mismatch")
        source_rows = self.source_manifest["outputs"]
        expected = {int(row["episode_index"]): int(row["source_frames"]) for row in source_rows}
        actual = {episode: int(row["raw_frames"]) for episode, row in self.records.items()}
        if (len(expected) != len(source_rows) or actual != expected
                or sorted(actual) != dataset.get("selected_episode_indices")
                or self.manifest["raw_frames"] != dataset.get("selected_frames")
                or self.manifest["raw_frames"] != self.source_manifest.get("raw_frames")
                or Path(dataset["root"]).resolve() != Path(self.plan["dataset_root"]).resolve()
                or Path(dataset["root"]).resolve() != Path(self.manifest["dataset_root"]).resolve()):
            raise ValueError("Native training bank must cover the complete bound source dataset")
        dataset_root = Path(dataset["root"]).resolve(strict=True)
        for name in ("info.json", "episodes.jsonl", "tasks.jsonl"):
            record = self.plan["metadata"][name]
            path = dataset_root / "meta" / name
            if (path.resolve() != Path(record["path"]).resolve()
                    or path.stat().st_size != record["bytes"]
                    or sha256_file(path) != record["sha256"]):
                raise ValueError(f"Native training dataset metadata changed: {name}")
        self._verified = {}
        for episode, row in self.records.items():
            for path_key, hash_key in (("relative_path", "sha256"),
                                       ("selection_relative_path", "selection_sha256")):
                path = bank_file(self.root, row[path_key])
                before = _fingerprint(path)
                if sha256_file(path) != row[hash_key] or _fingerprint(path) != before:
                    raise ValueError(f"Native training shard identity mismatch: {path}")
                self._verified[episode, path_key] = before
        self._episode_cache = OrderedDict()

    def _episode(self, episode):
        row = self.records[episode]
        paths = {}
        for key in ("relative_path", "selection_relative_path"):
            paths[key] = bank_file(self.root, row[key])
            if _fingerprint(paths[key]) != self._verified[episode, key]:
                raise ValueError("Native training shard changed after verification")
        if episode in self._episode_cache:
            self._episode_cache.move_to_end(episode)
            return self._episode_cache[episode]
        features = _mapped_features(paths["relative_path"])
        selection = read_tensors(paths["selection_relative_path"])
        frames = int(row["raw_frames"])
        # Full finite-value validation belongs to release verification. Check
        # shape, clock and layout here; selected features are checked on assembly.
        if (features["features"].shape != (frames, TOKENS_PER_FRAME, FEATURE_DIM)
                or features["features"].dtype != torch.bfloat16
                or features["raw_frame_indices"].dtype != torch.int64
                or not torch.equal(features["raw_frame_indices"], torch.arange(frames))
                or features["source_grid_hw"].dtype != torch.int64
                or not torch.equal(features["source_grid_hw"], torch.tensor(GRID_HW).expand(frames, -1))):
            raise ValueError("Native training shard violates its shape/raw-clock/grid contract")
        for prefix in ("vcs", "uniform"):
            validate_selection(selection[f"{prefix}_raw_frame_indices"], selection[f"{prefix}_frame_mask"], frames)
        self._episode_cache[episode] = (features, selection)
        if len(self._episode_cache) > 256:   # mapped features: a cache entry holds no feature copy
            self._episode_cache.popitem(last=False)
        return features, selection

    def memory_for_frame(self, episode_index, current_frame_index):
        if type(episode_index) is not int or type(current_frame_index) is not int:
            raise TypeError("Native memory episode/frame identifiers must be integers")
        memory = self.memory_for_query(episode_index, current_frame_index, selector=self.selector)
        result = {name: memory[name] for name in FRAMESAMP_MEMORY_FIELDS}
        raw = result["memory_raw_frame_indices"][result["memory_mask"]]
        result["selected_raw_frame_indices"] = raw.reshape(-1, TOKENS_PER_FRAME)[:, 0].clone()
        return result


class PaddedCollatorForNativeKeyframe(PaddedCollatorForTogether):
    MEMORY_FIELDS = FRAMESAMP_MEMORY_FIELDS
    AUDIT_FIELDS = ("framesamp_episode_index", "framesamp_frame_index")

    def __init__(self, *args, memory_budget=MEMORY_BUDGET, feature_dim=FEATURE_DIM, **kwargs):
        if memory_budget != MEMORY_BUDGET or feature_dim != FEATURE_DIM:
            raise ValueError("Native memory collator requires 539 tokens with 2048 features")
        super().__init__(*args, **kwargs)
        self.memory_budget, self.feature_dim = memory_budget, feature_dim

    def __call__(self, instances):
        for sample in instances:
            for name in self.MEMORY_FIELDS + self.AUDIT_FIELDS:
                if name not in sample:
                    raise KeyError(f"Native memory sample is missing {name}")
                if not isinstance(sample[name], torch.Tensor):
                    raise TypeError("Native memory and audit fields must be tensors")
            features, mask, raw, spatial = (sample[name] for name in self.MEMORY_FIELDS)
            if features.dtype != torch.bfloat16 or mask.dtype != torch.bool:
                raise TypeError("Native memory requires bfloat16 features and bool masks")
            if any(sample[name].dtype != torch.int64 for name in self.MEMORY_FIELDS[2:] + self.AUDIT_FIELDS):
                raise TypeError("Native memory identifiers must be int64")
            if any(sample[name].ndim != 0 for name in self.AUDIT_FIELDS):
                raise ValueError("Native memory audit identifiers must be scalar")
            current = int(sample["framesamp_frame_index"])
            if current < 0 or int(sample["framesamp_episode_index"]) < 0:
                raise ValueError("Native memory audit identifiers must be nonnegative")
            # Flexible keyframes: 1..min(7, t+1) whole right-padded frames (fixed K = the old exact count)
            cap = min(MAX_FRAMES, current + 1) * TOKENS_PER_FRAME
            if (features.shape != (MEMORY_BUDGET, FEATURE_DIM)
                    or any(item.shape != (MEMORY_BUDGET,) for item in (mask, raw, spatial))):
                raise ValueError("Native memory shape differs from its seven-frame budget")
            count = int(mask.sum())
            if (count % TOKENS_PER_FRAME or not TOKENS_PER_FRAME <= count <= cap
                    or not torch.equal(mask, torch.arange(MEMORY_BUDGET) < count)):
                raise ValueError("Native memory padding must hold 1..min(7,t+1) whole right-padded frames")
            if (not torch.isfinite(features).all() or torch.count_nonzero(features[~mask])
                    or not torch.all(raw[~mask] == -1) or not torch.all(spatial[~mask] == -1)):
                raise ValueError("Native memory values or padding are invalid")
            grouped_raw = raw[:count].reshape(-1, TOKENS_PER_FRAME)
            selected = grouped_raw[:, 0]
            if (not torch.equal(grouped_raw, selected[:, None].expand_as(grouped_raw))
                    or not torch.equal(spatial[:count], torch.arange(TOKENS_PER_FRAME).repeat(len(selected)))
                    or selected[0] < 0 or selected[-1] != current
                    or not torch.all(torch.diff(selected) > 0)):
                raise ValueError("Native memory must contain chronological causal frames including current")
        result = super().__call__(instances)
        for name in self.MEMORY_FIELDS + self.AUDIT_FIELDS:
            result[name] = torch.stack([sample[name] for sample in instances])
        return result

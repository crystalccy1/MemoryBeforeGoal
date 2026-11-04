"""Data contract for seven causal frames with native 7x11 visual tokens.

This bank is deliberately distinct from the frozen 16-token FrameSamp bank.
Features are stored once per execution frame; selection tables reference them
for every training query, without duplicating a 539-token tensor per query.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from psi.data.lerobot.framesamp_features import assemble_memory_tokens, sha256_file


BANK_SCHEMA = "psi0-keyframe-native-bank/v1"
PLAN_SCHEMA = "psi0-keyframe-native-plan/v1"
MAX_FRAMES = 7
GRID_HW = (7, 11)
TOKENS_PER_FRAME = 77
FEATURE_DIM = 2048
MEMORY_BUDGET = MAX_FRAMES * TOKENS_PER_FRAME


def validate_selection(indices: torch.Tensor, mask: torch.Tensor, frames: int) -> None:
    if indices.shape != (frames, MAX_FRAMES) or mask.shape != indices.shape:
        raise ValueError("Selection must contain one seven-slot row per raw frame")
    if indices.dtype != torch.int64 or mask.dtype != torch.bool:
        raise TypeError("Selection indices/mask must be int64/bool")
    # Flexible keyframes: a row may hold FEWER than min(7, t+1) frames (at least the current one); it stays right
    # padded. Fixed-K selections (exactly min(7, t+1)) are a special case and validate as before.
    cap = torch.arange(1, frames + 1).clamp(max=MAX_FRAMES)
    expected_count = mask.sum(dim=1)
    expected_mask = torch.arange(MAX_FRAMES)[None] < expected_count[:, None]
    if not torch.equal(mask, expected_mask) or torch.any(expected_count < 1) or torch.any(expected_count > cap):
        raise ValueError("Selection must be right padded with 1..min(7,t+1) valid frames")
    if torch.any(indices[~mask] != -1) or torch.any(indices[mask] < 0):
        raise ValueError("Selection requires nonnegative frame indices and -1 padding")
    query = torch.arange(frames)[:, None]
    if torch.any((indices > query) & mask):
        raise ValueError("Selection contains a future frame")
    if torch.any((indices[:, 1:] <= indices[:, :-1]) & mask[:, 1:]):
        raise ValueError("Selected frames must be chronological and unique")
    current = indices[torch.arange(frames), expected_count - 1]
    if not torch.equal(current, torch.arange(frames)):
        raise ValueError("Every selection must include its current frame")


def uniform_selection(frames: int) -> tuple[torch.Tensor, torch.Tensor]:
    if frames < 1:
        raise ValueError("Episode must have positive length")
    indices = torch.full((frames, MAX_FRAMES), -1, dtype=torch.int64)
    mask = torch.zeros((frames, MAX_FRAMES), dtype=torch.bool)
    for t in range(frames):
        count = min(MAX_FRAMES, t + 1)
        indices[t, :count] = torch.from_numpy(np.linspace(0, t, count, dtype=np.int64))
        mask[t, :count] = True
    validate_selection(indices, mask, frames)
    return indices, mask


def validate_features(tensors: dict[str, torch.Tensor], frames: int) -> None:
    features = tensors["features"]
    if features.shape != (frames, TOKENS_PER_FRAME, FEATURE_DIM):
        raise ValueError("Native features must have shape [raw_frames,77,2048]")
    if features.dtype != torch.bfloat16:
        raise TypeError("Native features must be bfloat16")
    raw = tensors["raw_frame_indices"]
    grid = tensors["source_grid_hw"]
    if raw.dtype != torch.int64 or not torch.equal(raw, torch.arange(frames)):
        raise ValueError("Native features must cover every raw frame in order")
    if grid.dtype != torch.int64 or grid.shape != (frames, 2) or not torch.equal(
        grid, torch.tensor(GRID_HW).expand(frames, -1)
    ):
        raise ValueError("Native features must retain the 7x11 source grid")
    for start in range(0, frames, 32):
        if not torch.isfinite(features[start:start + 32]).all():
            raise ValueError("Native features contain non-finite values")


def read_tensors(path: Path, *, expected_sha256: str | None = None,
                 expected_metadata: dict[str, str] | None = None) -> dict[str, torch.Tensor]:
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError(f"Shard hash mismatch: {path}")
    with safe_open(path, framework="pt", device="cpu") as handle:
        if expected_metadata is not None and handle.metadata() != expected_metadata:
            raise ValueError(f"Shard metadata mismatch: {path}")
        return {name: handle.get_tensor(name) for name in handle.keys()}


def bank_file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve(strict=True)
    path.relative_to(root.resolve(strict=True))
    return path


class NativeKeyframeBank:
    """Verified data reader; adapting the action model is a separate change."""

    def __init__(self, root: str | Path, *, expected_manifest_sha256: str, require_ready: bool = True):
        self.root = Path(root).resolve(strict=True)
        manifest_path = self.root / "manifest.json"
        ready = (json.loads((self.root / "READY").read_text()) if require_ready else
                 {"schema_version": BANK_SCHEMA, "manifest_sha256": expected_manifest_sha256})
        self.manifest = json.loads(manifest_path.read_text())
        actual_sha = sha256_file(manifest_path)
        if expected_manifest_sha256 != actual_sha or ready.get("manifest_sha256") != actual_sha:
            raise ValueError("Native bank manifest is not the pinned completed input")
        m = self.manifest
        if (m.get("schema_version") != BANK_SCHEMA or ready.get("schema_version") != BANK_SCHEMA
                or m.get("status") != "complete" or m.get("max_memory_frames") != MAX_FRAMES
                or m.get("feature_shape_tail") != [TOKENS_PER_FRAME, FEATURE_DIM]
                or m.get("memory_budget") != MEMORY_BUDGET or m.get("include_current") is not True
                or m.get("source_grid_hw") != list(GRID_HW)):
            raise ValueError("Unsupported native memory bank contract")
        plan_path = self.root / "plan.json"
        if sha256_file(plan_path) != m.get("plan_sha256"):
            raise ValueError("Native bank plan identity mismatch")
        self.plan = json.loads(plan_path.read_text())
        if self.plan.get("schema_version") != PLAN_SCHEMA:
            raise ValueError("Unsupported native bank plan schema")
        outputs = m.get("outputs", [])
        self.records = {int(row["episode_index"]): row for row in outputs}
        expected = {int(row["episode_index"]): int(row["raw_frames"]) for row in self.plan["outputs"]}
        actual = {key: int(row["raw_frames"]) for key, row in self.records.items()}
        if (len(expected) != len(self.plan["outputs"]) or len(outputs) != len(self.records) or actual != expected
                or len(outputs) != m["episodes"] or sum(actual.values()) != m["raw_frames"]):
            raise ValueError("Native bank episode/frame coverage is incomplete")
        self._loaded: tuple[int, dict, dict] | None = None

    def _episode(self, episode: int) -> tuple[dict, dict]:
        if self._loaded is not None and self._loaded[0] == episode:
            return self._loaded[1:]
        row = self.records[episode]
        frames = row["raw_frames"]
        features = read_tensors(bank_file(self.root, row["relative_path"]),
                                expected_sha256=row["sha256"])
        selection = read_tensors(bank_file(self.root, row["selection_relative_path"]),
                                 expected_sha256=row["selection_sha256"])
        validate_features(features, frames)
        for prefix in ("vcs", "uniform"):
            validate_selection(selection[f"{prefix}_raw_frame_indices"],
                               selection[f"{prefix}_frame_mask"], frames)
        self._loaded = (episode, features, selection)
        return features, selection

    def memory_for_query(self, episode: int, current_frame: int, *, selector: str = "vcs") -> dict:
        if selector not in ("vcs", "uniform"):
            raise ValueError("selector must be vcs or uniform")
        if not 0 <= current_frame < self.records[episode]["raw_frames"]:
            raise ValueError("Query lies outside the episode")
        features, selection = self._episode(episode)
        ids = selection[f"{selector}_raw_frame_indices"][current_frame]
        mask = selection[f"{selector}_frame_mask"][current_frame]
        raw = ids[mask]
        result = assemble_memory_tokens(features["features"][raw], raw,
                                        max_memory_frames=MAX_FRAMES, tokens_per_frame=TOKENS_PER_FRAME,
                                        feature_dim=FEATURE_DIM)
        spatial = result["memory_spatial_indices"]
        valid = result["memory_mask"]
        # Explicit coordinates prevent accidental decoding through the old 4x4 grid.
        result["memory_spatial_rows"] = torch.where(valid, spatial // GRID_HW[1], -1)
        result["memory_spatial_columns"] = torch.where(valid, spatial % GRID_HW[1], -1)
        return result

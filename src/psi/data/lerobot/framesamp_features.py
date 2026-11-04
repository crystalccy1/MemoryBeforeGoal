from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


FEATURE_BANK_SCHEMA_VERSION = "psi0-framesamp-feature-bank/v3"
FEATURE_CONTRACT_SCHEMA_VERSION = "psi0-framesamp-feature-contract/v3"


def raw_frame_indices(num_frames: int) -> np.ndarray:
    """Return every execution-frame index in one episode."""

    if isinstance(num_frames, bool) or num_frames <= 0:
        raise ValueError("num_frames must be a positive integer")
    return np.arange(int(num_frames), dtype=np.int64)


def sha256_file(path: str | Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def even_sample_positions(num_available: int, max_frames: int) -> np.ndarray:
    if num_available <= 0:
        raise ValueError("num_available must be positive")
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    if num_available <= max_frames:
        return np.arange(num_available, dtype=np.int64)
    # Match the released RoboMME integer linspace behavior: retain the first
    # and latest available query frame while spreading the remaining samples.
    return np.linspace(0, num_available - 1, max_frames, dtype=np.int64)


def memory_raw_frame_indices(
    current_frame_index: int,
    episode_frames: int,
    max_frames: int,
) -> np.ndarray:
    """Uniformly select at most ``max_frames`` from the visible raw prefix.

    This is the strict RoboMME FrameSamp clock: every raw execution frame is
    available, the current frame is included, and selection is recomputed for
    every query rather than being tied to a policy-query stride.
    """

    if not 0 <= current_frame_index < episode_frames:
        raise ValueError("current_frame_index lies outside the episode")
    available = raw_frame_indices(current_frame_index + 1)
    return available[even_sample_positions(len(available), max_frames)]


def assemble_memory_tokens(
    selected_features: torch.Tensor,
    selected_raw_indices: torch.Tensor,
    *,
    max_memory_frames: int = 32,
    tokens_per_frame: int = 16,
    feature_dim: int = 2048,
) -> dict[str, torch.Tensor]:
    """Build the identical right-padded token contract offline and online."""

    if selected_features.ndim != 3:
        raise ValueError("selected_features must have shape [frames,tokens,features]")
    frame_count = int(selected_features.shape[0])
    if tuple(selected_features.shape[1:]) != (tokens_per_frame, feature_dim):
        raise ValueError("selected_features has the wrong token/feature shape")
    if selected_features.dtype != torch.bfloat16:
        raise TypeError("selected_features must use bfloat16")
    if selected_raw_indices.shape != (frame_count,) or selected_raw_indices.dtype != torch.int64:
        raise ValueError("selected_raw_indices must be int64 with one value per frame")
    if frame_count < 1 or frame_count > max_memory_frames:
        raise ValueError("selected frame count lies outside the memory budget")
    if int(selected_raw_indices[0]) < 0 or not torch.all(
        torch.diff(selected_raw_indices) > 0
    ):
        raise ValueError("selected raw frame indices must be strictly increasing")
    if not torch.isfinite(selected_features.float()).all():
        raise ValueError("selected features must be finite")

    memory_budget = max_memory_frames * tokens_per_frame
    valid_tokens = frame_count * tokens_per_frame
    memory_features = torch.zeros(
        (memory_budget, feature_dim), dtype=torch.bfloat16
    )
    memory_mask = torch.zeros(memory_budget, dtype=torch.bool)
    memory_raw_indices = torch.full((memory_budget,), -1, dtype=torch.int64)
    memory_spatial_indices = torch.full((memory_budget,), -1, dtype=torch.int64)
    memory_features[:valid_tokens] = selected_features.reshape(
        valid_tokens, feature_dim
    )
    memory_mask[:valid_tokens] = True
    memory_raw_indices[:valid_tokens] = selected_raw_indices.repeat_interleave(
        tokens_per_frame
    )
    memory_spatial_indices[:valid_tokens] = torch.arange(
        tokens_per_frame, dtype=torch.int64
    ).repeat(frame_count)
    return {
        "memory_features": memory_features,
        "memory_mask": memory_mask,
        "memory_raw_frame_indices": memory_raw_indices,
        "memory_spatial_indices": memory_spatial_indices,
    }


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def write_json_atomic(path: str | Path, payload: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / f".{destination.name}.{os.getpid()}.incomplete"
    if os.path.lexists(stage):
        raise FileExistsError(f"Refusing stale JSON stage: {stage}")
    encoded = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    try:
        with stage.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(stage, destination)
    finally:
        if stage.exists():
            stage.unlink()


def parse_episode_selection(raw: str | None, available: Iterable[int]) -> list[int]:
    available_list = sorted(int(value) for value in available)
    available_set = set(available_list)
    if raw is None or not raw.strip():
        return available_list
    selected = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("episode selection must contain unique integers")
    missing = sorted(set(selected) - available_set)
    if missing:
        raise ValueError(f"Unknown episode indices: {missing}")
    return sorted(selected)

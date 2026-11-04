"""Shared offline/online inputs for the original A/B/C Gist experiment.

Spatial pooling belongs to the trainable model.  In particular, Learned-All
must retain the sixteen visual cells and pool the projected position branch
with the very same attention weights used for the visual branch.
"""
from __future__ import annotations

import numpy as np
import torch

GIST_BANK_SCHEMA = "psi0-gist-prefix-bank/v1"
GIST_DETAIL_FRAMES = 16
GIST_MEMORY_FIELDS = (
    "memory_frame_features", "memory_frame_mask",
    "memory_frame_raw_indices", "memory_detail_mask",
)


def detailed_frame_indices(num_frames: int) -> torch.Tensor:
    """Inclusive causal prefix, matching FrameSamp's integer-linspace clock."""
    if isinstance(num_frames, bool) or not isinstance(num_frames, int) or num_frames < 1:
        raise ValueError("num_frames must be a positive integer")
    return torch.from_numpy(np.linspace(
        0, num_frames - 1, min(num_frames, GIST_DETAIL_FRAMES), dtype=np.int64
    ))


def detail_index_table(num_frames: int) -> torch.Tensor:
    table = torch.full((num_frames, GIST_DETAIL_FRAMES), -1, dtype=torch.int64)
    for frame in range(num_frames):
        indices = detailed_frame_indices(frame + 1)
        table[frame, :indices.numel()] = indices
    return table


def assemble_gist_inputs(
    prefix_features: torch.Tensor,
    raw_indices: torch.Tensor | None = None,
    *,
    detail_indices: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return unpadded [0,t] inputs; batch padding is performed by the collator.

    The caller supplies only the prefix visible at the current query.  Full
    episode banks must slice before calling this function.  No history cap or
    stride is applied: all non-detailed frames reach the model's Gist branch.
    """
    if prefix_features.ndim != 3 or prefix_features.shape[1] != 16:
        raise ValueError("Gist features must have shape [frames,16,features]")
    n = int(prefix_features.shape[0])
    if n < 1 or prefix_features.shape[2] < 1:
        raise ValueError("Gist requires a non-empty visible prefix")
    if prefix_features.dtype != torch.bfloat16:
        raise TypeError("Gist source features must use bfloat16")
    device = prefix_features.device
    expected_raw = torch.arange(n, dtype=torch.int64, device=device)
    if raw_indices is None:
        raw_indices = expected_raw
    if raw_indices.dtype != torch.int64 or not torch.equal(raw_indices.to(device), expected_raw):
        raise ValueError("Gist requires every raw frame from zero through current")
    expected_detail = detailed_frame_indices(n).to(device)
    if detail_indices is not None:
        selected = detail_indices[detail_indices >= 0].to(device)
        if selected.dtype != torch.int64 or not torch.equal(selected, expected_detail):
            raise ValueError("Gist detailed frames differ from uniform inclusive selection")
    detail_mask = torch.zeros(n, dtype=torch.bool, device=device)
    detail_mask[expected_detail] = True
    return {
        "memory_frame_features": prefix_features,
        "memory_frame_mask": torch.ones(n, dtype=torch.bool, device=device),
        "memory_frame_raw_indices": raw_indices.to(device),
        "memory_detail_mask": detail_mask,
    }

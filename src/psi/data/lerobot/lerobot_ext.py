# Derived from Psi0 (https://github.com/physical-superintelligence-lab/Psi0, Apache License 2.0) and modified; see NOTICE.
from __future__ import annotations
from typing import Any, Dict, TYPE_CHECKING
from pathlib import Path
if TYPE_CHECKING:
    from psi.config.data_lerobot import LerobotDataConfig
    # from psi.config.data_simple import SimpleDataConfig

import os
import numpy as np
import torch
from psi.data.lerobot.compat import (
    LEROBOT_LAYOUT,
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)
from psi.utils import resolve_path
from psi.config.transform import LerobotRepackTransform
from psi.data.lerobot.symbolic_memory import SymbolicMemorySidecar


def decord_frame_indices(
    video_path: str | os.PathLike[str], timestamps: list[float] | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve timestamps exactly as the released Decord loader did."""
    import decord

    reader = decord.VideoReader(str(video_path))
    frame_timestamps = np.asarray(
        reader.get_frame_timestamp(range(len(reader))), dtype=np.float64
    )[:, 0]
    query_timestamps = np.asarray(timestamps, dtype=np.float64)
    indices = np.abs(
        frame_timestamps[:, None] - query_timestamps[None, :]
    ).argmin(axis=0)
    deltas = np.abs(frame_timestamps[indices] - query_timestamps)
    return indices, deltas


def decode_video_frames_decord(
    video_path: str | os.PathLike[str],
    timestamps: list[float],
    tolerance_s: float,
) -> torch.Tensor:
    """Decode timestamp-nearest RGB frames with pinned Decord semantics."""
    import decord

    reader = decord.VideoReader(str(video_path))
    frame_timestamps = np.asarray(
        reader.get_frame_timestamp(range(len(reader))), dtype=np.float64
    )[:, 0]
    query_timestamps = np.asarray(timestamps, dtype=np.float64)
    indices = np.abs(
        frame_timestamps[:, None] - query_timestamps[None, :]
    ).argmin(axis=0)
    deltas = np.abs(frame_timestamps[indices] - query_timestamps)
    if not np.all(deltas < tolerance_s):
        raise AssertionError(
            "One or more Decord frames violate the timestamp tolerance: "
            f"deltas={deltas.tolist()}, tolerance_s={tolerance_s}, "
            f"video={video_path}"
        )

    frames = reader.get_batch(indices).asnumpy()
    del reader
    return (
        torch.from_numpy(np.ascontiguousarray(frames))
        .permute(0, 3, 1, 2)
        .to(dtype=torch.float32)
        .div_(255.0)
    )


def _install_decord_decoder() -> None:
    """Add the released Decord backend to the public LeRobot 0.3 adapter."""
    if LEROBOT_LAYOUT != "datasets":
        # Older LeRobot layouts used by the internal release already carry
        # their own backend implementation.
        return

    from lerobot.datasets import lerobot_dataset, video_utils

    current = video_utils.decode_video_frames
    if getattr(current, "_psi_supports_decord", False):
        return

    def decode_video_frames(
        video_path, timestamps, tolerance_s, backend=None
    ):
        if backend == "decord":
            return decode_video_frames_decord(
                video_path, timestamps, tolerance_s
            )
        return current(video_path, timestamps, tolerance_s, backend)

    decode_video_frames._psi_supports_decord = True
    video_utils.decode_video_frames = decode_video_frames
    # lerobot_dataset imported the function directly, so its binding must be
    # replaced as well as the source module attribute.
    lerobot_dataset.decode_video_frames = decode_video_frames


# DataLoader workers are created with a fresh interpreter in the released
# training setup. Patching only while the parent constructs the dataset leaves
# those workers with LeRobot's stock decoder, which rejects ``backend=decord``.
# The backend selection is inherited in the worker environment, so install the
# adapter during module import in every process as well as during construction.
if os.environ.get("LEROBOT_VIDEO_BACKEND") == "decord":
    _install_decord_decoder()

class LeRobotDatasetWrapper(torch.utils.data.Dataset):
    """ A wrapper around LeRobotDataset to support multiple datasets.
    """

    def __init__(
        self, 
        data_cfg: LerobotDataConfig, 
        split: str = "train"
    ):
        repo_ids = data_cfg.train_repo_ids if split == "train" else data_cfg.val_repo_ids
        first_repo = repo_ids[0] if isinstance(repo_ids, list) else repo_ids
        dataset_meta = LeRobotDatasetMetadata(first_repo, resolve_path(f"{data_cfg.root_dir}/{first_repo}"))
        assert isinstance(data_cfg.transform.repack, LerobotRepackTransform)
        delta_timestamps = data_cfg.transform.repack.delta_timestamps(dataset_meta.fps)

        if len(repo_ids) > 1:
            root_dir = data_cfg.root_dir
            lerobot_dataset_class = MultiLeRobotDataset
        else:
            repo_ids = first_repo
            root_dir = resolve_path(f"{data_cfg.root_dir}/{first_repo}")
            lerobot_dataset_class = LeRobotDataset

        configured_backend = data_cfg.video_backend
        video_backend = os.environ.get(
            "LEROBOT_VIDEO_BACKEND", configured_backend
        )
        if video_backend == "decord":
            _install_decord_decoder()

        self.base_dataset = lerobot_dataset_class(
            repo_ids,# type: ignore
            root=root_dir,
            delta_timestamps=delta_timestamps, # type: ignore
            image_transforms=None,
            video_backend=video_backend,
        )
        self.symbolic_memory: SymbolicMemorySidecar | None = None
        self.symbolic_use_phase_action_mask = data_cfg.symbolic_use_phase_action_mask
        if data_cfg.symbolic_memory_mode != "disabled":
            if len(data_cfg.train_repo_ids) != 1:
                raise ValueError(
                    "Symbolic memory v1 supports exactly one LeRobot dataset"
                )
            rows_path = Path(str(data_cfg.symbolic_memory_rows_path)).expanduser()
            if not rows_path.is_absolute():
                rows_path = Path(data_cfg.root_dir) / rows_path
            self.symbolic_memory = SymbolicMemorySidecar(
                rows_path,
                expected_task=str(first_repo),
                expected_rows=len(self.base_dataset),
                expected_action_horizon=data_cfg.transform.repack.action_chunk_size,
                mode=data_cfg.symbolic_memory_mode,
                seed=data_cfg.symbolic_memory_seed,
            )
        self._cache = {}

    def __getitem__(self, idx) -> dict:
        sample = self.base_dataset[idx]
        if self.symbolic_memory is not None:
            sample = self.symbolic_memory.augment_sample(sample)
            if not self.symbolic_use_phase_action_mask:
                sample.pop("phase_action_mask", None)
            return sample
        return sample
    
    def __len__(self):
        return len(self.base_dataset)

    @property
    def episode_data_index(self):
        return self.base_dataset.episode_data_index # type: ignore

    @property
    def num_episodes(self):
        return self.base_dataset.num_episodes
    
    @property
    def num_frames(self):
        return self.base_dataset.num_frames
    
    @property
    def meta(self):
        return self.base_dataset.meta # type: ignore

    @property
    def stats(self):
        return self.base_dataset.stats if type(self.base_dataset) == MultiLeRobotDataset else self.base_dataset.meta.stats # type: ignore

from __future__ import annotations

from typing import Any

import torch

from psi.config.transform import DataTransform


FRAMESAMP_MEMORY_FIELDS = (
    "memory_features",
    "memory_mask",
    "memory_raw_frame_indices",
    "memory_spatial_indices",
)


def scalar_integer(value: Any, name: str) -> int:
    if value is None:
        raise KeyError(f"FrameSamp raw sample is missing {name}")
    tensor = torch.as_tensor(value)
    if tensor.numel() != 1 or tensor.dtype == torch.bool:
        raise TypeError(f"{name} must be one integer scalar")
    if tensor.is_floating_point():
        raise TypeError(f"{name} must use an integer dtype")
    return int(tensor.item())


class FrameSampDataTransform(DataTransform):
    """Run the clean transform, then attach episode-local memory.

    Capturing the two LeRobot identifiers before repacking avoids deriving
    episode membership from a global dataset row and leaves the current-image
    Qwen path byte-for-byte identical to the clean Psi0 transform.
    """

    def __call__(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        feature_bank = kwargs.pop("framesamp_feature_bank", None)
        if feature_bank is None:
            raise RuntimeError("FrameSampDataTransform requires a verified feature bank")
        episode_index = scalar_integer(data.get("episode_index"), "episode_index")
        frame_index = scalar_integer(data.get("frame_index"), "frame_index")
        result = super().__call__(data, **kwargs)
        memory = feature_bank.memory_for_frame(episode_index, frame_index)
        memory.pop("selected_raw_frame_indices")
        overlap = set(result).intersection(memory)
        if overlap:
            raise RuntimeError(f"Clean transform returned reserved memory fields: {overlap}")
        result.update(memory)
        result["framesamp_episode_index"] = torch.tensor(
            episode_index, dtype=torch.int64
        )
        result["framesamp_frame_index"] = torch.tensor(
            frame_index, dtype=torch.int64
        )
        return result

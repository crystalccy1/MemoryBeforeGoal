from __future__ import annotations

import torch

from psi.config.transform import DataTransform
from psi.config.transform_framesamp import scalar_integer


class GistDataTransform(DataTransform):
    def __call__(self, data, **kwargs):
        bank = kwargs.pop("gist_feature_bank", None)
        if bank is None:
            raise RuntimeError("GistDataTransform requires its verified feature bank")
        episode = scalar_integer(data.get("episode_index"), "episode_index")
        frame = scalar_integer(data.get("frame_index"), "frame_index")
        result = super().__call__(data, **kwargs)
        memory = bank.memory_for_frame(episode, frame)
        if set(memory).intersection(result):
            raise RuntimeError("Clean transform returned reserved Gist memory fields")
        result.update(memory)
        result["gist_episode_index"] = torch.tensor(episode, dtype=torch.int64)
        result["gist_frame_index"] = torch.tensor(frame, dtype=torch.int64)
        return result

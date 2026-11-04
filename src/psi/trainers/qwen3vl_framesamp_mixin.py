from __future__ import annotations

import torch

from psi.data.lerobot.framesamp_features import memory_raw_frame_indices
from psi.trainers.qwen3vl_mixin import PaddedCollatorForTogether


class PaddedCollatorForFrameSamp(PaddedCollatorForTogether):
    MEMORY_FIELDS = (
        "memory_features",
        "memory_mask",
        "memory_raw_frame_indices",
        "memory_spatial_indices",
    )
    AUDIT_FIELDS = ("framesamp_episode_index", "framesamp_frame_index")

    def __init__(
        self,
        *args,
        memory_budget: int = 512,
        feature_dim: int = 2048,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.memory_budget = memory_budget
        self.feature_dim = feature_dim

    @staticmethod
    def _tensor(value) -> torch.Tensor:
        return value if isinstance(value, torch.Tensor) else torch.as_tensor(value)

    def __call__(self, instances):
        for index, instance in enumerate(instances):
            missing = [
                field
                for field in self.MEMORY_FIELDS + self.AUDIT_FIELDS
                if field not in instance
            ]
            if missing:
                raise KeyError(f"FrameSamp instance {index} is missing {missing}")
            if not all(
                isinstance(instance[field], torch.Tensor)
                for field in self.MEMORY_FIELDS
            ):
                raise TypeError("FrameSamp collator requires tensor memory fields")
            if instance["memory_features"].dtype != torch.bfloat16:
                raise TypeError("FrameSamp features must remain bfloat16")
            if instance["memory_mask"].dtype != torch.bool:
                raise TypeError("FrameSamp mask must remain boolean")
            if any(
                instance[field].dtype != torch.int64
                for field in self.MEMORY_FIELDS[2:]
            ):
                raise TypeError("FrameSamp token metadata must remain int64")
        output = super().__call__(instances)
        for field in self.MEMORY_FIELDS:
            output[field] = torch.stack(
                [self._tensor(instance[field]) for instance in instances]
            )
        for field in self.AUDIT_FIELDS:
            output[field] = torch.stack(
                [self._tensor(instance[field]) for instance in instances]
            ).to(torch.int64)
            if output[field].shape != (len(instances),):
                raise RuntimeError(f"FrameSamp audit field {field} must be scalar")
        if output["memory_features"].shape != (
            len(instances),
            self.memory_budget,
            self.feature_dim,
        ):
            raise RuntimeError("FrameSamp feature batch shape mismatch")
        if output["memory_mask"].shape != (len(instances), self.memory_budget):
            raise RuntimeError("FrameSamp mask batch shape mismatch")
        output["memory_features"] = output["memory_features"].to(torch.bfloat16)
        output["memory_mask"] = output["memory_mask"].to(torch.bool)
        for field in self.MEMORY_FIELDS[2:]:
            output[field] = output[field].to(torch.int64)

        padding = ~output["memory_mask"]
        valid_counts = output["memory_mask"].sum(dim=1)
        expected_mask = (
            torch.arange(self.memory_budget)[None, :]
            < valid_counts[:, None]
        )
        if not torch.equal(output["memory_mask"].cpu(), expected_mask):
            raise RuntimeError("FrameSamp mask must be a contiguous left prefix")
        if torch.count_nonzero(output["memory_features"][padding]) != 0:
            raise RuntimeError("FrameSamp padded features must be zero")
        if not torch.isfinite(output["memory_features"].float()).all():
            raise RuntimeError("FrameSamp features must be finite")
        for field in self.MEMORY_FIELDS[2:]:
            if not torch.all(output[field][padding] == -1):
                raise RuntimeError(f"FrameSamp padded {field} values must be -1")
        if not torch.all(valid_counts % 16 == 0):
            raise RuntimeError("FrameSamp valid token count must contain whole frames")
        if torch.any(valid_counts == 0):
            raise RuntimeError("Strict FrameSamp memory must include the current raw frame")
        for sample_index, count in enumerate(valid_counts.tolist()):
            valid_slice = slice(0, count)
            raw = output["memory_raw_frame_indices"][sample_index, valid_slice]
            spatial = output["memory_spatial_indices"][sample_index, valid_slice]
            if count and (
                not torch.all(raw.reshape(-1, 16).eq(raw.reshape(-1, 16)[:, :1]))
                or not torch.equal(
                    spatial.reshape(-1, 16),
                    torch.arange(16, dtype=torch.int64)[None, :].expand(count // 16, -1),
                )
            ):
                raise RuntimeError("FrameSamp token metadata does not form complete frames")
            if count:
                frame_raw = raw.reshape(-1, 16)[:, 0]
                current_frame = int(output["framesamp_frame_index"][sample_index])
                expected_raw = torch.from_numpy(
                    memory_raw_frame_indices(
                        current_frame,
                        current_frame + 1,
                        self.memory_budget // 16,
                    )
                )
                if (
                    not torch.all(torch.diff(frame_raw) > 0)
                    or int(frame_raw[0]) < 0
                    or not torch.equal(frame_raw.cpu(), expected_raw)
                ):
                    raise RuntimeError("FrameSamp selected frames violate the raw-frame clock")
        return output

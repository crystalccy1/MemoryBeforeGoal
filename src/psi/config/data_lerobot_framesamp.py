from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Literal

from pydantic import model_validator

from psi.config.data_lerobot import LerobotDataConfig


class LerobotFrameSampDataConfig(LerobotDataConfig):
    """LeRobot data config bound to one immutable FrameSamp feature bank."""

    framesamp_feature_root: str
    framesamp_feature_manifest_sha256: str
    framesamp_dataset_revision: str
    framesamp_feature_scope: Literal["smoke", "full"] = "full"

    @model_validator(mode="after")
    def validate_framesamp_data_contract(self) -> "LerobotFrameSampDataConfig":
        if len(self.train_repo_ids) != 1:
            raise ValueError("FrameSamp v2 requires exactly one training task")
        if len(self.val_repo_ids) != 1 or self.val_repo_ids != self.train_repo_ids:
            raise ValueError("FrameSamp v2 validation must use the same task bank")
        if len(self.framesamp_feature_manifest_sha256) != 64:
            raise ValueError("FrameSamp manifest SHA-256 must contain 64 hex digits")
        if not re.fullmatch(
            r"[0-9a-f]{64}", self.framesamp_feature_manifest_sha256
        ):
            raise ValueError("FrameSamp manifest SHA-256 must be lowercase hexadecimal")
        return self

    def __call__(
        self, split: str = "train", transform_kwargs: dict | None = None, **kwargs
    ) -> Any:
        from psi.data.lerobot.framesamp_dataset import FrameSampFeatureBank

        repo_ids = self.train_repo_ids if split == "train" else self.val_repo_ids
        task = str(repo_ids[0])
        bank = FrameSampFeatureBank(
            self.framesamp_feature_root,
            expected_task=task,
            expected_dataset_revision=self.framesamp_dataset_revision,
            expected_scope=self.framesamp_feature_scope,
            expected_manifest_sha256=self.framesamp_feature_manifest_sha256,
        )
        expected_dataset_root = (Path(self.root_dir) / task).resolve(strict=True)
        bound_dataset_root = Path(bank.contract["dataset"]["root"]).resolve(strict=True)
        if expected_dataset_root != bound_dataset_root:
            raise RuntimeError(
                "FrameSamp feature bank is bound to a different extracted dataset: "
                f"{bound_dataset_root} != {expected_dataset_root}"
            )
        forwarded_kwargs = dict(transform_kwargs or {})
        if "framesamp_feature_bank" in forwarded_kwargs:
            raise ValueError("framesamp_feature_bank is reserved by the data config")
        forwarded_kwargs["framesamp_feature_bank"] = bank
        return super().__call__(
            split=split,
            transform_kwargs=forwarded_kwargs,
            **kwargs,
        )

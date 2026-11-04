"""Bind the native 7x77 bank to its original LeRobot dataset and provenance."""
from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import re
from typing import Literal

from pydantic import PrivateAttr, model_validator

from psi.config.data_lerobot import LerobotDataConfig
from psi.config.data_lerobot_framesamp import LerobotFrameSampDataConfig


@lru_cache(maxsize=8)
def _training_bank(root, digest, task, revision, selector, process_id):
    from psi.data.lerobot.keyframe_native_training import NativeKeyframeTrainingBank
    return NativeKeyframeTrainingBank(root, expected_manifest_sha256=digest,
                                     expected_task=task, expected_dataset_revision=revision,
                                     selector=selector)


class LerobotNativeKeyframeDataConfig(LerobotFrameSampDataConfig):
    native_keyframe_root: str
    native_keyframe_manifest_sha256: str
    native_keyframe_selector: Literal["vcs", "uniform"] = "vcs"
    _native_bank: object = PrivateAttr(default=None)
    _native_bank_key: tuple | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def validate_native_keyframe(self):
        if not re.fullmatch(r"[0-9a-f]{64}", self.native_keyframe_manifest_sha256):
            raise ValueError("Native keyframe manifest requires a lowercase SHA-256")
        if self.framesamp_feature_scope != "full":
            raise ValueError("Native keyframe training requires the complete task dataset")
        return self

    def feature_bank(self):
        key = (str(Path(self.native_keyframe_root).resolve(strict=True)),
               self.native_keyframe_manifest_sha256, self.train_repo_ids[0],
               self.framesamp_dataset_revision, self.native_keyframe_selector, os.getpid())
        if key != self._native_bank_key:
            self._native_bank = _training_bank(*key)
            self._native_bank_key = key
        bank = self._native_bank
        if (bank.plan["source_manifest_sha256"] != self.framesamp_feature_manifest_sha256
                or Path(bank.plan["source_feature_root"]).resolve() != Path(self.framesamp_feature_root).resolve()
                or Path(bank.contract["dataset"]["root"]).resolve() !=
                   (Path(self.root_dir) / self.train_repo_ids[0]).resolve()):
            raise ValueError("Native config and source feature/dataset identities differ")
        return bank

    def __call__(self, split="train", transform_kwargs=None, **kwargs):
        if split not in {"train", "val"}:
            raise ValueError("Native keyframe split must be train or val")
        forwarded = dict(transform_kwargs or {})
        if {"native_keyframe_bank", "framesamp_feature_bank"}.intersection(forwarded):
            raise ValueError("Native/legacy feature bank arguments are reserved")
        forwarded["native_keyframe_bank"] = self.feature_bank()
        return LerobotDataConfig.__call__(self, split=split, transform_kwargs=forwarded, **kwargs)

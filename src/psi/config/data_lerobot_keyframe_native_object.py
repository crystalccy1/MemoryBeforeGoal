"""Bind the native 7x77 bank and its SAM3 object sidecar to the LeRobot dataset."""
from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import re
from typing import Literal, Optional

from pydantic import model_validator

from psi.config.data_lerobot_keyframe_native import LerobotNativeKeyframeDataConfig


@lru_cache(maxsize=8)
def _object_training_bank(root, digest, task, revision, object_root, object_digest, process_id,
                          decision_labels=None, decision_digest=None):
    from psi.data.lerobot.sam3_object_memory import Sam3ObjectDecisionTrainingBank, Sam3ObjectNativeTrainingBank
    common = dict(expected_manifest_sha256=digest, expected_task=task, expected_dataset_revision=revision,
                  object_root=object_root, object_manifest_sha256=object_digest)
    if decision_labels is None:
        return Sam3ObjectNativeTrainingBank(root, **common)
    return Sam3ObjectDecisionTrainingBank(root, **common, decision_labels=decision_labels,
                                          decision_labels_sha256=decision_digest)


class LerobotNativeKeyframeObjectDataConfig(LerobotNativeKeyframeDataConfig):
    native_keyframe_selector: Literal["sam3"] = "sam3"
    object_memory_root: str
    object_memory_manifest_sha256: str
    # Decision-head supervision (optional): JSON of per-episode answer + first supervised frame.
    decision_labels_path: Optional[str] = None
    decision_labels_sha256: Optional[str] = None

    @model_validator(mode="after")
    def validate_object_memory(self):
        if not re.fullmatch(r"[0-9a-f]{64}", self.object_memory_manifest_sha256):
            raise ValueError("object_memory_manifest_sha256 must be a lowercase SHA-256")
        if (self.decision_labels_path is None) != (self.decision_labels_sha256 is None):
            raise ValueError("decision_labels_path and decision_labels_sha256 go together")
        if self.decision_labels_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", self.decision_labels_sha256):
            raise ValueError("decision_labels_sha256 must be a lowercase SHA-256")
        return self

    def feature_bank(self):
        key = (str(Path(self.native_keyframe_root).resolve(strict=True)), self.native_keyframe_manifest_sha256,
               self.train_repo_ids[0], self.framesamp_dataset_revision,
               str(Path(self.object_memory_root).resolve(strict=True)), self.object_memory_manifest_sha256, os.getpid(),
               None if self.decision_labels_path is None else str(Path(self.decision_labels_path).resolve(strict=True)),
               self.decision_labels_sha256)
        if key != self._native_bank_key:
            self._native_bank = _object_training_bank(*key)
            self._native_bank_key = key
        bank = self._native_bank
        if (bank.plan["source_manifest_sha256"] != self.framesamp_feature_manifest_sha256
                or Path(bank.plan["source_feature_root"]).resolve() != Path(self.framesamp_feature_root).resolve()
                or Path(bank.contract["dataset"]["root"]).resolve() !=
                   (Path(self.root_dir) / self.train_repo_ids[0]).resolve()):
            raise ValueError("Object-memory config and source feature/dataset identities differ")
        return bank

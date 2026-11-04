from __future__ import annotations

from pathlib import Path
import re

from pydantic import model_validator

from psi.config.data_lerobot import LerobotDataConfig
from psi.config.data_lerobot_framesamp import LerobotFrameSampDataConfig


class LerobotGistDataConfig(LerobotFrameSampDataConfig):
    gist_root: str
    gist_manifest_sha256: str

    @model_validator(mode="after")
    def validate_gist(self):
        if not re.fullmatch(r"[0-9a-f]{64}", self.gist_manifest_sha256):
            raise ValueError("Gist manifest requires a lowercase SHA-256")
        if self.framesamp_feature_scope != "full":
            raise ValueError("A/B/C Gist requires the full task dataset")
        return self

    def feature_bank(self):
        from psi.data.lerobot.gist_dataset import GistFeatureBank
        bank = GistFeatureBank(
            self.gist_root, expected_manifest_sha256=self.gist_manifest_sha256,
            expected_task=self.train_repo_ids[0],
            expected_dataset_revision=self.framesamp_dataset_revision,
        )
        if (bank.manifest["source_feature_manifest_sha256"] != self.framesamp_feature_manifest_sha256
                or Path(bank.manifest["source_feature_root"]).resolve() != Path(self.framesamp_feature_root).resolve()
                or Path(bank.manifest["dataset_root"]).resolve() != (Path(self.root_dir) / self.train_repo_ids[0]).resolve()):
            raise ValueError("Gist data config and source feature/dataset identities differ")
        return bank

    def __call__(self, split="train", transform_kwargs=None, **kwargs):
        if split not in {"train", "val"}:
            raise ValueError("Gist split must be train or val")
        forwarded = dict(transform_kwargs or {})
        if any(name in forwarded for name in ("gist_feature_bank", "framesamp_feature_bank")):
            raise ValueError("Feature-bank transform arguments are reserved")
        forwarded["gist_feature_bank"] = self.feature_bank()
        return LerobotDataConfig.__call__(self, split=split, transform_kwargs=forwarded, **kwargs)

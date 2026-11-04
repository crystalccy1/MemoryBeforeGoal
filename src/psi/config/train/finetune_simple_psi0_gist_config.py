import os
from typing import Literal

from pydantic import model_validator

from psi.config.config import LaunchConfig, TrainConfig
from psi.config.data_lerobot_gist import LerobotGistDataConfig
from psi.config.model_psi0_gist import Psi0GistModelConfig
from psi.config.transform import ActionStateTransform, Psi0ModelTransform, SimpleRepackTransform
from psi.config.transform_gist import GistDataTransform


def require_wandb(report_to):
    # Our runs all logged to W&B. Logging does not change training, so the released configs log nothing by default
    # (log.report_to=null); with log.report_to=wandb, W&B must not be disabled.
    if report_to == "wandb" and os.environ.get("WANDB_DISABLED", "false").lower() not in ("false", "0", ""):
        raise ValueError("log.report_to=wandb but WANDB_DISABLED is set")


class DynamicDataTransform(GistDataTransform):
    repack: SimpleRepackTransform
    field: ActionStateTransform
    model: Psi0ModelTransform


class DynamicDataConfig(LerobotGistDataConfig):
    transform: DynamicDataTransform


class GistTrainConfig(TrainConfig):
    name: Literal["finetune_gist"] = "finetune_gist"


class DynamicLaunchConfig(LaunchConfig):
    data: DynamicDataConfig
    model: Psi0GistModelConfig
    train: GistTrainConfig

    @model_validator(mode="after")
    def validate_gist_launch(self):
        require_wandb(self.log.report_to)
        if self.train.lora:
            raise ValueError("Gist uses a frozen backbone and does not use LoRA")
        return self

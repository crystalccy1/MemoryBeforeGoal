"""Independent native seven-keyframe training configuration."""
from pathlib import Path
from typing import Literal

from pydantic import model_validator

from psi.config.config import LaunchConfig, TrainConfig
from psi.config.data_lerobot_keyframe_native import LerobotNativeKeyframeDataConfig
from psi.config.model_psi0_keyframe_native import Psi0KeyframeNativeModelConfig
from psi.config.train.finetune_simple_psi0_gist_config import require_wandb
from psi.config.transform import ActionStateTransform, Psi0ModelTransform, SimpleRepackTransform
from psi.config.transform_keyframe_native import NativeKeyframeDataTransform


class DynamicDataTransform(NativeKeyframeDataTransform):
    repack: SimpleRepackTransform
    field: ActionStateTransform
    model: Psi0ModelTransform


class DynamicDataConfig(LerobotNativeKeyframeDataConfig):
    transform: DynamicDataTransform


class NativeKeyframeTrainConfig(TrainConfig):
    name: Literal["finetune_keyframe_native"] = "finetune_keyframe_native"


class DynamicLaunchConfig(LaunchConfig):
    data: DynamicDataConfig
    model: Psi0KeyframeNativeModelConfig
    train: NativeKeyframeTrainConfig

    @model_validator(mode="after")
    def validate_native_launch(self):
        require_wandb(self.log.report_to)
        if self.train.lora or self.model.tune_vlm or self.model.framesamp_entry_probe_only:
            raise ValueError("Native keyframes require the full memory path and frozen VLM without LoRA")
        for name in ("native_keyframe_root", "native_keyframe_manifest_sha256", "native_keyframe_selector",
                     "framesamp_feature_manifest_sha256", "framesamp_dataset_revision"):
            if getattr(self.model, name) != getattr(self.data, name):
                raise ValueError(f"Native model/data binding differs: {name}")
        if Path(self.model.framesamp_feature_manifest).resolve() != (
                Path(self.data.framesamp_feature_root) / "manifest.json").resolve():
            raise ValueError("Native model/data binding differs: original feature manifest path")
        return self

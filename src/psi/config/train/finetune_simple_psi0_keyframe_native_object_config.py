"""Native 7x77 SAM3-selected keyframes plus gated SAM3 object memory."""
from pathlib import Path
from typing import Literal

from pydantic import model_validator

from psi.config.config import LaunchConfig, TrainConfig
from psi.config.data_lerobot_keyframe_native_object import LerobotNativeKeyframeObjectDataConfig
from psi.config.model_psi0_keyframe_native_object import Psi0KeyframeNativeObjectModelConfig
from psi.config.train.finetune_simple_psi0_gist_config import require_wandb
from psi.config.train.finetune_simple_psi0_keyframe_native_config import DynamicDataTransform


class DynamicDataConfig(LerobotNativeKeyframeObjectDataConfig):
    transform: DynamicDataTransform


class NativeKeyframeObjectTrainConfig(TrainConfig):
    name: Literal["finetune_keyframe_native_object"] = "finetune_keyframe_native_object"


class DynamicLaunchConfig(LaunchConfig):
    data: DynamicDataConfig
    model: Psi0KeyframeNativeObjectModelConfig
    train: NativeKeyframeObjectTrainConfig

    @model_validator(mode="after")
    def validate_object_launch(self):
        require_wandb(self.log.report_to)
        if self.train.lora or self.model.tune_vlm or self.model.framesamp_entry_probe_only:
            raise ValueError("Object memory requires the full memory path and frozen VLM without LoRA")
        for name in ("native_keyframe_root", "native_keyframe_manifest_sha256", "native_keyframe_selector",
                     "framesamp_feature_manifest_sha256", "framesamp_dataset_revision",
                     "object_memory_root", "object_memory_manifest_sha256"):
            if getattr(self.model, name) != getattr(self.data, name):
                raise ValueError(f"Object-memory model/data binding differs: {name}")
        if Path(self.model.framesamp_feature_manifest).resolve() != (
                Path(self.data.framesamp_feature_root) / "manifest.json").resolve():
            raise ValueError("Object-memory model/data binding differs: original feature manifest path")
        return self

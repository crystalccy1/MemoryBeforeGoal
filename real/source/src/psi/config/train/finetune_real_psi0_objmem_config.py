"""finetune_real_psi0_config (the moving_dining_cart recipe) with the SAM3-v2 memory model.

Data transform, stats and every training flag stay those of finetune_real_psi0_config; only the model
config gains the memory fields and the trainer (train.name=sonic_objmem) attaches memory to each sample.
"""
from typing import Any

from pydantic import model_validator

from psi.config.config import LaunchConfig
from psi.config.data_lerobot import LerobotDataConfig
from psi.config.model_psi0_objmem import Psi0ObjMemModelConfig
from psi.config.transform import DataTransform
from psi.config import transform as pt


class DynamicDataTransform(DataTransform):
    repack: pt.RealRepackTransform
    field: pt.ActionStateTransform
    model: pt.Psi0ModelTransform


class DynamicDataConfig(LerobotDataConfig):
    transform: DynamicDataTransform


class DynamicLaunchConfig(LaunchConfig):
    data: DynamicDataConfig
    model: Psi0ObjMemModelConfig

    @model_validator(mode="after")
    def check_observation_dim(self, __context: Any) -> None:
        assert self.data.transform.repack.pad_action_dim == self.data.transform.field.pad_action_dim, "inconsistent action dim"
        assert self.data.transform.repack.pad_state_dim == self.data.transform.field.pad_state_dim, "inconsistent state dim"
        if self.data.transform.repack.pad_state_dim is not None:
            assert self.model.odim == self.data.transform.repack.pad_state_dim, "inconsitent odim"
        assert self.train.name == "sonic_objmem", "the SAM3-v2 memory model needs the sonic_objmem trainer"
        assert not self.train.lora, "LoRA is not part of the cart recipe"
        return self

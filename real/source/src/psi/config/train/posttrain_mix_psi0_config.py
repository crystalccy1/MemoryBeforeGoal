from typing import Union, Annotated
from pydantic import BaseModel, Field, model_validator
# from tyro.conf import subcommand as cmd
from psi.config.config import LaunchConfig
from psi.config.model_psi0 import Psi0ModelConfig
# from psi.config.data_he import HERawDataConfig
from psi.config.data_mix import MixedDataConfig
from psi.config import transform as pt
from psi.config.transform import DataTransform

class DynamicDataTransform(DataTransform):
    repack: pt.MixedRepackTransform
    field: pt.ActionStateTransform
    model: pt.Psi0ModelTransform

class DynamicDataConfig(MixedDataConfig):
    transform: DynamicDataTransform

class DynamicLaunchConfig(LaunchConfig):
    data: DynamicDataConfig
    model: Psi0ModelConfig
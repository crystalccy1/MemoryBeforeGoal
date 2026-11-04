from typing import Union, Annotated
from pydantic import BaseModel, Field, model_validator

from psi.config.config import LaunchConfig
# from psi.config.data_simple import SimpleDataConfig
from psi.config.data_lerobot import LerobotDataConfig
from psi.config.model_psi0 import Psi0ModelConfig
from psi.config.transform import DataTransform
from psi.config import transform as pt

class DynamicDataTransform(DataTransform):
    repack: pt.SimpleRepackTransform
    field: pt.ActionStateTransform
    model: pt.Psi0ModelTransform

class DynamicDataConfig(LerobotDataConfig):
    transform: DynamicDataTransform

class DynamicLaunchConfig(LaunchConfig):
    data: DynamicDataConfig
    model: Psi0ModelConfig
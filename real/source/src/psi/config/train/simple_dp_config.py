from typing import Union, Annotated
from pydantic import BaseModel, Field, model_validator

from psi.config.config import LaunchConfig
from psi.config.data_lerobot import LerobotDataConfig
from psi.config.model_dp import DiffusionPolicyModelConfig
from psi.config.transform import DataTransform
from psi.config import transform as pt

class DynamicDataTransform(DataTransform):
    repack: pt.SimpleRepackTransform
    field: pt.ActionStateTransform
    model: pt.DiffusionPolicyModelTransform

class DynamicDataConfig(LerobotDataConfig):
    transform: DynamicDataTransform
    
class DynamicLaunchConfig(LaunchConfig):
    data: DynamicDataConfig
    model: DiffusionPolicyModelConfig
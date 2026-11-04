from tyro.conf import subcommand as cmd
from typing import Union, Annotated, Optional, List
from psi.config.config import ModelConfig
from pydantic import Field

class DiffusionPolicyModelConfig(ModelConfig):
    num_diffusion_iters: int = 100
    action_chunk_size: int = 16 # Tp, pred_horizon
    num_cameras: int = 1 
    share_vision_encoder: bool = False
    obs_horizon: int = 1
    action_exec_horizon: int = 6
    action_dim: int = 2
    obs_dim: int = 15 
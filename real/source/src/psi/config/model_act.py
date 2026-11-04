from tyro.conf import subcommand as cmd
from typing import Union, Annotated, Optional, List
from psi.config.config import ModelConfig
from pydantic import Field

class ACTModelConfig(ModelConfig):
    n_obs_steps: int = 1
    chunk_size: int = 100  # action chunk size (prediction horizon)
    n_action_steps: int = 100  # number of action steps to execute
    
    # Dimensions
    action_dim: int = 7  # action dimension
    state_dim: int = 15  # robot state dimension (proprioception)
    
    # Transformer architecture
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1  # Original ACT has a bug, only 1 layer is used
    pre_norm: bool = False
    dropout: float = 0.1
    
    # VAE configuration
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4
    kl_weight: float = 10.0
    
    # Inference
    temporal_ensemble_coeff: float | None = None  # set to 0.01 to enable
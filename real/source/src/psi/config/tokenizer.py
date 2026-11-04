from pydantic import BaseModel, Field, model_validator
from typing import Optional

class ActionTokenizerConfig(BaseModel):
    ...

# class BinActionTokenizerConfig(ActionTokenizerConfig):
#     bins: int = 256
#     min_action: float = -1.0
#     max_action: float = 1.0


# #### FSQModel
# class VQVaeActionTokenizerConfig(ActionTokenizerConfig):
#     ## vqvae arch
#     quantizer_name: str = "fsq" # choices=['residualvq', 'group_residualvq', 'part_group_residualvq', 'lfq', 'fsq']
#     dim_motion: int = 48
#     code_dim: int = 512
#     nb_code: int = 512
#     mu: float = 0.99
#     down_t: int = 2
#     stride_t: int = 2
#     width: int = 512
#     depth: int = 3
#     dilation_growth_rate: int = 3
#     output_emb_width: int = 512
#     activate: str = 'relu' # choices=['relu', 'silu', 'gelu']
#     norm: Optional[str] = None
#     num_conv_layers: int = 1
#     num_quantizers: int = 1
#     shared_codebook: bool = False


#     # loss related
#     commit_weight: float = 0.02
#     vel_weight: float = 0.5
#     recons_loss: str = 'l1_smooth'


#     # for openvla_rlds_vqvae
#     pretrained_checkpoint: str | None = None
#     freeze_vqvae: bool = True  # Whether to freeze VQ-VAE parameters during training


### ResidualVQModel
class VQVaeActionTokenizerConfig(ActionTokenizerConfig):
    ## vqvae arch
    quantizer_name: str = "residualvq" # choices=['residualvq', 'group_residualvq', 'part_group_residualvq', 'lfq', 'fsq']
    dim_motion: int = 48
    code_dim: int = 512
    nb_code: int = 512
    mu: float = 0.99
    down_t: int = 2
    stride_t: int = 2
    width: int = 512
    depth: int = 3
    dilation_growth_rate: int = 3
    output_emb_width: int = 512
    activate: str = 'relu' # choices=['relu', 'silu', 'gelu']
    norm: Optional[str] = None
    num_conv_layers: int = 1
    num_quantizers: int = 8
    shared_codebook: bool = False


    # loss related
    commit_weight: float = 0.02
    vel_weight: float = 0.5
    recons_loss: str = 'l1_smooth'


    # for openvla_rlds_vqvae
    pretrained_checkpoint: str | None = None
    freeze_vqvae: bool = True  # Whether to freeze VQ-VAE parameters during training


class FastActionTokenizerConfig(ActionTokenizerConfig):
    bins: int = 2048
    pretrained_checkpoint: str | None = None

class VQActionTokenizerConfig(ActionTokenizerConfig):
    bins: int = 256
    pretrained_checkpoint: str = ""

class TextActionTokenizerConfig(ActionTokenizerConfig):
    pass

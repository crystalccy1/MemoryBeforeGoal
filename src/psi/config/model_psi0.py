# Derived from Psi0 (https://github.com/physical-superintelligence-lab/Psi0, Apache License 2.0) and modified; see NOTICE.
from tyro.conf import subcommand as cmd
from typing import Union, Annotated, Optional, List
from psi.config.config import ModelConfig
from pydantic import Field, model_validator

class Psi0ModelConfig(ModelConfig):
    ######################### hfm_action ############################
    # pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-3-medium-diffusers"
    resnet_store_path: str | None = None # = "cache/checkpoints/resnet18/IN_1M_resnet18.pth"
    pretrained_action_header_path: str | None = None
    # precomputed_text_encodings: str = "cache/precomputed_text_encodings.pkl"
    # ckpt_step: Optional[int] = None

    rtc: bool = False
    max_delay: int = 8

    action_dim: int = 7 #36 # Da
    action_chunk_size: int = 6 #30 # Tp
    action_exec_horizon: int = 6 #30 # Ta
    observation_horizon: int = 1 # To past frames

    img_chunk: int = 1 # ?
    n_cams: int = 1 
    use_obs: str = "add_token"
    dropout: float = 0.1
    noise_scheduler: str = "flow"
    train_diffusion_steps: int = 1000
    eval_diffusion_steps: int = 10
    share_cam_features: bool = False
    early_fusion: bool = False

    # observations
    odim: int = 15 # 32

    # conditions
    n_conditions: Optional[int] = 1
    token_fusion: Optional[str] = "concat"  # "concat", "cross", "perceiver"

    # tune_vision_backbone: bool = False
    # vision_backbone_lr: float = 1e-5

    """
        training loss weight for xyz, rpy, and gripper
        should all sumed to 1 in total, eg 0.1x3+0.2x3+0.1x1=1
    """
    loss_w: List[float] = Field(default_factory=lambda: [0.1, 0.2, 0.1])

    # noise nets
    time_dim: int = 256
    hidden_dim: int = 1536
    num_blocks: int = 6
    dim_feedforward: int = 2048
    nhead: int = 24
    activation: str = "gelu"

    view_feature_dim: int = 1920 # views feature dim for views and/ or vlm token dim
    use_film: bool = False
    combined_temb: bool = False

    use_dit: bool = False

    ################### qwen3vl ####################
    # Training Schedule
    weight_decay: float = 0.01 # L2 regularization strength

    # Core Arguments
    model_name_or_path: str = "Qwen/Qwen3-VL-2B-Instruct"  # First load, for initalization
    # vlm_run_dir: str | None = None
    vlm_ckpt_step: str | None = None
    tune_vlm: bool = False


    tune_mm_llm: bool = False                   # [TrainingArguments] Train LLM or not
    tune_mm_vision: bool = False                # [TrainingArguments] Train VIT or not
    tune_mm_mlp: bool = False                   # [TrainingArguments] Train MLP or not
    # dataset_use: str = "egodex"               # [DataArguments] Dataset specification
    # output_dir: str = "./outputs/hfm_qwen3vl" # Output directory for checkpoints
    # cache_dir: str = "./cache/models"         # [TrainingArguments] Model cache location
    gradient_checkpointing: bool = True  # [TrainingArguments] Enable gradient checkpointing

    # Learning Rate Configuration
    lang_backbone_lr: float = 1e-5 
    mm_projector_lr: float = 1e-5   # [TrainingArguments] Projector-specific LR
    vision_tower_lr: float = 1e-6   # [TrainingArguments] Vision encoder LR
    optim: str = "adamw_torch"      # [TrainingArguments] Optimizer selection

    ################### FastWAM world expert ####################
    # Wan2.2 VAE is a frozen trainer-owned encoder. Keeping it outside the
    # Psi0FastWAMModel means Accelerator checkpoints contain only Qwen + the
    # trainable action/world experts, rather than duplicating the VAE weights.
    video_vae_path: str | None = None
    video_vae_repo_id: str = "Wan-AI/Wan2.2-TI2V-5B"
    video_vae_filename: str = "Wan2.2_VAE.pth"
    video_vae_tiled: bool = False

    video_num_frames: int = 9
    video_height: int = 160
    video_width: int = 320
    video_latent_channels: int = 48
    video_latent_frames: int = 3
    world_patch_size: int = 2
    max_world_tokens: int = 256

    # The action expert keeps the regular training LR. The newly initialized
    # world expert can optionally use a separate LR without changing old runs.
    world_loss_weight: float = 1.0
    world_learning_rate: float | None = None

    @model_validator(mode="after")
    def validate_fastwam_settings(self):
        if self.video_num_frames <= 1 or self.video_num_frames % 4 != 1:
            raise ValueError("video_num_frames must be 4n+1 for the causal Wan VAE.")
        if self.video_latent_channels != 48:
            raise ValueError(
                "WanVideoVAE38 requires video_latent_channels=48."
            )
        expected_latent_frames = (self.video_num_frames + 3) // 4
        if self.video_latent_frames != expected_latent_frames:
            raise ValueError(
                "video_latent_frames must match the Wan temporal compression: "
                f"expected {expected_latent_frames}, got {self.video_latent_frames}."
            )
        if self.video_height <= 0 or self.video_width <= 0:
            raise ValueError("video_height and video_width must be positive.")
        if self.world_patch_size <= 0:
            raise ValueError("world_patch_size must be positive.")
        spatial_multiple = 16 * self.world_patch_size
        if (
            self.video_height % spatial_multiple != 0
            or self.video_width % spatial_multiple != 0
        ):
            raise ValueError(
                "FastWAM video height/width must be divisible by "
                f"{spatial_multiple} (Wan 16x VAE followed by world patching)."
            )
        if self.world_loss_weight < 0:
            raise ValueError("world_loss_weight must be non-negative.")
        if self.max_world_tokens <= 0:
            raise ValueError("max_world_tokens must be positive.")
        required_world_tokens = (
            self.video_latent_frames
            * (self.video_height // spatial_multiple)
            * (self.video_width // spatial_multiple)
        )
        if self.max_world_tokens < required_world_tokens:
            raise ValueError(
                "max_world_tokens is too small for the configured video: "
                f"need at least {required_world_tokens}, got "
                f"{self.max_world_tokens}."
            )
        if self.world_learning_rate is not None and self.world_learning_rate <= 0:
            raise ValueError("world_learning_rate must be positive when provided.")
        return self

    # Sequence Configuration
    model_max_length: int = 4096 # [TrainingArguments] Max sequence length
    data_flatten: bool = True    # [DataArguments] Concatenate batch sequences
    data_packing: bool = True    # [DataArguments] Using packing data

    # Image Processing
    max_pixels: int = 576 * 28 * 28  # [DataArguments] Max image pixels (H*W) for image
    min_pixels: int = 16 * 28 * 28   # [DataArguments] Min image pixels for image


    # lora_r: int = 8                          # [TrainingArguments] LoRA r
    # lora_alpha: int= 16                      # [TrainingArguments] LoRA alpha 
    # lora_dropout: float = 0.0                 # [TrainingArguments] LoRA dropout

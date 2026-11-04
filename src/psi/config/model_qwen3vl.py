from tyro.conf import subcommand as cmd
from typing import Union, Annotated
from psi.config.config import ModelConfig

from .tokenizer import (
    FastActionTokenizerConfig, 
    VQActionTokenizerConfig, 
    TextActionTokenizerConfig
)

class Qwen3VLModelConfig(ModelConfig):
    # Training Schedule
    weight_decay: float = 0.01 # L2 regularization strength

    # Core Arguments
    model_name_or_path: str = "Qwen/Qwen3-VL-2B-Instruct"  # [ModelArguments] Model identifier
    tune_mm_llm: bool = True                    # [TrainingArguments] Train LLM or not
    tune_mm_vision: bool = False                # [TrainingArguments] Train VIT or not
    tune_mm_mlp: bool = True                   # [TrainingArguments] Train MLP or not
    # dataset_use: str = "egodex"               # [DataArguments] Dataset specification
    # output_dir: str = "./outputs/hfm_qwen3vl" # Output directory for checkpoints
    # cache_dir: str = "./cache/models"         # [TrainingArguments] Model cache location
    gradient_checkpointing: bool = True  # [TrainingArguments] Enable gradient checkpointing

    # Learning Rate Configuration
    mm_projector_lr: float = 1e-5   # [TrainingArguments] Projector-specific LR
    vision_tower_lr: float = 1e-6   # [TrainingArguments] Vision encoder LR
    optim: str = "adamw_torch"      # [TrainingArguments] Optimizer selection

    # Sequence Configuration
    model_max_length: int = 8192 # [TrainingArguments] Max sequence length

    lora_r: int = 8                          # [TrainingArguments] LoRA r
    lora_alpha: int= 16                      # [TrainingArguments] LoRA alpha 
    lora_dropout: float = 0.0                 # [TrainingArguments] LoRA dropout
    
    action_tokenizer: Union[
        Annotated[FastActionTokenizerConfig, cmd("fast")],
        Annotated[VQActionTokenizerConfig, cmd("vq")],
        Annotated[TextActionTokenizerConfig, cmd("text")],
    ]
    deepspeed: str = "scripts/deepspeed/zero3.json"  # DeepSpeed configuration name
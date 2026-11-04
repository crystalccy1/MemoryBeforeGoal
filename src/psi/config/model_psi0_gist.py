"""Original A/B/C frame-gist experiment, independent of TokenDrop selection."""

from typing import Literal

from pydantic import Field, model_validator

from psi.config.model_psi0 import Psi0ModelConfig


class Psi0GistModelConfig(Psi0ModelConfig):
    noise_scheduler: Literal["flow"] = "flow"
    gist_enabled: Literal[True] = True
    gist_variant: Literal["A", "B", "C"] = "A"
    gist_tokens_per_frame: Literal[16] = 16
    gist_detailed_frames: Literal[16] = 16
    gist_fixed_gist_budget: Literal[256] = 256
    gist_feature_dim: int = Field(default=2048, gt=0)
    gist_position_dim: int = Field(default=768, ge=12)
    gist_memory_heads: int = Field(default=6, gt=0)
    gist_memory_head_dim: int = Field(default=256, gt=0)
    gist_memory_kv_heads: Literal[1] = 1
    gist_modulation_init: Literal["robomme_normal_std_0_002"] = "robomme_normal_std_0_002"
    gist_online_memory_view: Literal["rgb_head_stereo_left"] = "rgb_head_stereo_left"
    gist_feature_representation: Literal["qwen3vl_vision_merger"] = "qwen3vl_vision_merger"
    gist_history_policy: Literal["all_raw_frames_le_current"] = "all_raw_frames_le_current"

    @model_validator(mode="after")
    def validate_gist(self):
        if self.use_dit or self.tune_vlm:
            raise ValueError("Gist requires the standard action header and frozen Qwen")
        if self.gist_position_dim % 6 or self.gist_memory_head_dim % 2:
            raise ValueError("Gist positions require dim % 6 == 0 and even RoPE head width")
        if self.hidden_dim != self.gist_memory_heads * self.gist_memory_head_dim:
            raise ValueError("Gist memory heads must span the action hidden dimension")
        return self

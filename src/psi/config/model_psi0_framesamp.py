from __future__ import annotations

from typing import Literal
import re

from pydantic import model_validator

from psi.config.model_psi0 import Psi0ModelConfig


class Psi0FrameSampModelConfig(Psi0ModelConfig):
    """Configuration boundary for the Psi0 FrameSamp+Modul policy.

    The first implementation stage is deliberately an entry probe.  It proves
    that the dedicated model and trainer are active before any expensive
    perceptual features are generated.  The later feature stage switches
    ``framesamp_entry_probe_only`` off and binds an immutable feature manifest.
    """

    framesamp_enabled: bool = True
    framesamp_entry_probe_only: bool = True
    framesamp_feature_manifest: str | None = None
    framesamp_feature_manifest_sha256: str | None = None
    framesamp_dataset_revision: str | None = None
    framesamp_feature_scope: Literal["smoke", "full"] = "full"
    framesamp_memory_clock: Literal["execution_frame"] = "execution_frame"
    # Kept in the serialized config as an explicit migration guard.  Strict
    # v2 banks contain every raw execution frame, hence the only valid value is
    # one.  A v1 stride-24 config cannot silently instantiate this model.
    framesamp_query_stride: int = 1
    framesamp_off_clock_mapping: Literal["all_raw_frames_le_current"] = (
        "all_raw_frames_le_current"
    )
    framesamp_memory_budget: int = 512
    framesamp_tokens_per_frame: int = 16
    framesamp_feature_dim: int = 2048
    framesamp_position_dim: int = 768
    framesamp_memory_heads: int = 6
    framesamp_memory_kv_heads: int = 1
    framesamp_memory_head_dim: int = 256
    framesamp_temporal_position_source: Literal["raw_frame_index"] = (
        "raw_frame_index"
    )
    framesamp_modulation_init: Literal[
        "robomme_normal_std_0_002", "psi0_zero_ablation"
    ] = "robomme_normal_std_0_002"
    framesamp_training_origin: Literal["common_psi0_warm_start_step0"] = (
        "common_psi0_warm_start_step0"
    )
    framesamp_online_memory_view: str = "rgb_head_stereo_left"
    framesamp_pool_type: Literal["mean"] = "mean"
    framesamp_padding_side: Literal["right"] = "right"
    # Dynamic uniform sampling covers the complete raw prefix [0, current], so
    # the current raw frame is always present in memory and also remains the
    # ordinary System-2 observation.
    framesamp_include_current: bool = True

    @model_validator(mode="after")
    def validate_framesamp_contract(self) -> "Psi0FrameSampModelConfig":
        if not self.framesamp_enabled:
            raise ValueError("Psi0FrameSampModel requires framesamp_enabled=true")
        if self.framesamp_memory_budget <= 0:
            raise ValueError("framesamp_memory_budget must be positive")
        if self.framesamp_tokens_per_frame <= 0:
            raise ValueError("framesamp_tokens_per_frame must be positive")
        if self.framesamp_memory_budget % self.framesamp_tokens_per_frame != 0:
            raise ValueError(
                "framesamp_memory_budget must be divisible by framesamp_tokens_per_frame"
            )
        if self.framesamp_feature_dim <= 0:
            raise ValueError("framesamp_feature_dim must be positive")
        if self.framesamp_position_dim <= 0:
            raise ValueError("framesamp_position_dim must be positive")
        if self.framesamp_query_stride <= 0:
            raise ValueError("framesamp_query_stride must be positive")
        if self.hidden_dim != self.framesamp_memory_heads * self.framesamp_memory_head_dim:
            raise ValueError(
                "hidden_dim must equal framesamp_memory_heads * memory_head_dim"
            )
        if self.framesamp_memory_kv_heads != 1:
            raise ValueError("FrameSamp v2 uses one shared memory KV head")
        if self.framesamp_position_dim % 6:
            raise ValueError("framesamp_position_dim must be divisible by six")
        if self.framesamp_position_dim < 12:
            raise ValueError("framesamp_position_dim must be at least 12")
        if self.framesamp_memory_heads <= 0 or self.framesamp_memory_head_dim <= 0:
            raise ValueError("FrameSamp attention dimensions must be positive")
        if self.framesamp_memory_head_dim % 2:
            raise ValueError("FrameSamp attention head dimension must be even")
        if not self.framesamp_online_memory_view.strip():
            raise ValueError("framesamp_online_memory_view must be non-empty")
        if not self.framesamp_entry_probe_only:
            frozen_contract = {
                "framesamp_memory_clock": (self.framesamp_memory_clock, "execution_frame"),
                "framesamp_query_stride": (self.framesamp_query_stride, 1),
                "framesamp_memory_budget": (self.framesamp_memory_budget, 512),
                "framesamp_tokens_per_frame": (self.framesamp_tokens_per_frame, 16),
                "framesamp_feature_dim": (self.framesamp_feature_dim, 2048),
                "framesamp_position_dim": (self.framesamp_position_dim, 768),
                "framesamp_memory_heads": (self.framesamp_memory_heads, 6),
                "framesamp_memory_kv_heads": (self.framesamp_memory_kv_heads, 1),
                "framesamp_memory_head_dim": (self.framesamp_memory_head_dim, 256),
                "framesamp_padding_side": (self.framesamp_padding_side, "right"),
                "framesamp_include_current": (self.framesamp_include_current, True),
                "framesamp_pool_type": (self.framesamp_pool_type, "mean"),
                "framesamp_off_clock_mapping": (
                    self.framesamp_off_clock_mapping,
                    "all_raw_frames_le_current",
                ),
                "framesamp_temporal_position_source": (
                    self.framesamp_temporal_position_source,
                    "raw_frame_index",
                ),
                "framesamp_training_origin": (
                    self.framesamp_training_origin,
                    "common_psi0_warm_start_step0",
                ),
                "tune_vlm": (self.tune_vlm, False),
                "framesamp_online_memory_view": (
                    self.framesamp_online_memory_view,
                    "rgb_head_stereo_left",
                ),
            }
            mismatched = [
                name for name, (actual, expected) in frozen_contract.items()
                if actual != expected
            ]
            if mismatched:
                raise ValueError(f"Formal FrameSamp contract drift: {mismatched}")
            if not self.framesamp_feature_manifest:
                raise ValueError(
                    "framesamp_feature_manifest is required after the entry-probe stage"
                )
            if not self.framesamp_feature_manifest_sha256:
                raise ValueError(
                    "framesamp_feature_manifest_sha256 is required in formal mode"
                )
            if not self.framesamp_dataset_revision:
                raise ValueError("framesamp_dataset_revision is required in formal mode")
            if not self.pretrained_action_header_path:
                raise ValueError(
                    "Strict FrameSamp must start from the declared common Psi-0 warm start"
                )
            if self.framesamp_feature_scope != "full":
                raise ValueError("Formal FrameSamp training requires a full feature bank")
            if len(self.framesamp_feature_manifest_sha256) != 64:
                raise ValueError("feature manifest SHA-256 must contain 64 hex digits")
            if not re.fullmatch(
                r"[0-9a-f]{64}", self.framesamp_feature_manifest_sha256
            ):
                raise ValueError("feature manifest SHA-256 must be lowercase hexadecimal")
        return self

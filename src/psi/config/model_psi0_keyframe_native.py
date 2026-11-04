"""Independent training boundary for seven native 7x11 merger-token frames."""
from __future__ import annotations

import re
from typing import Literal

from pydantic import model_validator

from psi.config.model_psi0_framesamp import Psi0FrameSampModelConfig


class Psi0KeyframeNativeModelConfig(Psi0FrameSampModelConfig):
    native_keyframe_enabled: Literal[True] = True
    native_keyframe_root: str
    native_keyframe_manifest_sha256: str
    native_keyframe_selector: Literal["vcs", "uniform"] = "vcs"
    native_keyframe_grid_hw: tuple[Literal[7], Literal[11]] = (7, 11)
    native_keyframe_contract_version: Literal[1] = 1
    native_keyframe_spatial_coordinates: Literal["cell_centers_normalized_to_16"] = (
        "cell_centers_normalized_to_16"
    )
    framesamp_entry_probe_only: Literal[False] = False
    framesamp_memory_budget: Literal[539] = 539
    framesamp_tokens_per_frame: Literal[77] = 77
    framesamp_pool_type: Literal["none"] = "none"
    action_dim: int = 36
    action_chunk_size: int = 30
    action_exec_horizon: int = 30

    @model_validator(mode="after")
    def validate_framesamp_contract(self):
        # Override only this subclass's validator. Strict FrameSamp16 retains
        # its original frozen 512-token/4x4 contract.
        expected = {
            "framesamp_enabled": True, "framesamp_entry_probe_only": False,
            "framesamp_memory_budget": 539, "framesamp_tokens_per_frame": 77,
            "framesamp_feature_dim": 2048, "framesamp_position_dim": 768,
            "framesamp_memory_heads": 6, "framesamp_memory_kv_heads": 1,
            "framesamp_memory_head_dim": 256, "framesamp_query_stride": 1,
            "framesamp_include_current": True, "framesamp_padding_side": "right",
            "framesamp_pool_type": "none", "framesamp_memory_clock": "execution_frame",
            "framesamp_off_clock_mapping": "all_raw_frames_le_current",
            "framesamp_temporal_position_source": "raw_frame_index",
            "framesamp_training_origin": "common_psi0_warm_start_step0",
            "framesamp_feature_scope": "full", "framesamp_online_memory_view": "rgb_head_stereo_left",
            "tune_vlm": False, "use_dit": False, "hidden_dim": 1536,
            "action_dim": 36, "action_chunk_size": 30, "action_exec_horizon": 30,
            "observation_horizon": 1,
        }
        mismatches = [name for name, value in expected.items() if getattr(self, name) != value]
        if mismatches:
            raise ValueError(f"Native keyframe model contract drift: {mismatches}")
        for field in ("framesamp_feature_manifest_sha256", "native_keyframe_manifest_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", getattr(self, field) or ""):
                raise ValueError(f"Missing or invalid {field}")
        if not all((self.native_keyframe_root, self.framesamp_feature_manifest,
                    self.framesamp_dataset_revision, self.pretrained_action_header_path)):
            raise ValueError("Native keyframe training requires both pinned banks and the common Psi0 warm start")
        return self

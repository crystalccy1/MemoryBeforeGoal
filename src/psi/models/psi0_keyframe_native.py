"""Psi0 memory modulation with seven unpooled 7x11 Qwen merger grids.

Spatial positions preserve the original FrameSamp coordinate extent [0,16]:
each native cell uses ((row+.5)*16/7, (column+.5)*16/11). Temporal positions
remain absolute raw execution-frame indices, not selected-frame ordinals.
Current-observation encoding and the action/memory attention blocks are shared
with FrameSamp; native feature geometry has its own strict checkpoint ABI.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from psi.data.lerobot.keyframe_native import NativeKeyframeBank
from psi.models.psi0 import Psi0Model
from psi.models.psi0_framesamp import (
    FrameSampActionTransformerModel,
    FrameSampMemoryEncoder,
    FrameSampPosEmb3D,
    FrameSampVLATransformerBlock,
    Psi0FrameSampModel,
)


NATIVE_KEYFRAME_ABI = (1, 7, 77, 7, 11)


class NativeKeyframePosEmb3D(FrameSampPosEmb3D):
    """Raw-time sinusoidal encoding plus 7x11 cell centers in a 16x16 extent."""

    def forward(self, raw_frame_indices, spatial_indices, memory_mask):
        raw_time = raw_frame_indices.clamp_min(0).float()
        spatial = spatial_indices.clamp_min(0)
        row = (torch.div(spatial, 11, rounding_mode="floor").float() + 0.5) * (16.0 / 7)
        column = ((spatial % 11).float() + 0.5) * (16.0 / 11)
        temporal = raw_time[..., None] * self.temporal_omega
        y = row[..., None] * self.spatial_omega
        x = column[..., None] * self.spatial_omega
        encoded = torch.cat((temporal.sin(), temporal.cos(), y.sin(), y.cos(), x.sin(), x.cos()), dim=-1)
        return encoded.masked_fill(~memory_mask[..., None], 0)


class NativeKeyframeMemoryEncoder(FrameSampMemoryEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.memory_budget != 539 or self.feature_dim != 2048:
            raise ValueError("Native keyframe memory requires [539,2048] features")
        self.position = NativeKeyframePosEmb3D(self.position_proj.in_features)

    def forward(self, memory_features, memory_mask, memory_raw_frame_indices, memory_spatial_indices):
        if memory_features.ndim != 3 or memory_features.shape[1:] != (539, 2048):
            raise ValueError("Native memory_features must have shape [B,539,2048]")
        shape = memory_features.shape[:2]
        if any(value.shape != shape for value in (memory_mask, memory_raw_frame_indices, memory_spatial_indices)):
            raise ValueError("Native keyframe mask/index shape mismatch")
        if memory_mask.dtype != torch.bool:
            raise TypeError("Native keyframe memory_mask must be boolean")
        if memory_raw_frame_indices.dtype != torch.int64 or memory_spatial_indices.dtype != torch.int64:
            raise TypeError("Native keyframe indices must use int64")
        if not memory_features.is_floating_point():
            raise TypeError("Native keyframe features must have a floating-point dtype")
        if any(value.device != memory_features.device for value in (memory_mask, memory_raw_frame_indices, memory_spatial_indices)):
            raise ValueError("Native keyframe features and indices must share a device")
        batch = memory_features.shape[0]
        mask = memory_mask.reshape(batch, 7, 77)
        raw = memory_raw_frame_indices.reshape(batch, 7, 77)
        spatial = memory_spatial_indices.reshape(batch, 7, 77)
        frame_mask = mask[..., 0]
        if not torch.equal(mask, frame_mask[..., None].expand_as(mask)):
            raise ValueError("Native keyframe mask must cover whole 77-token frames")
        if torch.any(frame_mask[:, 1:] & ~frame_mask[:, :-1]):
            raise ValueError("Native keyframe frames must use right padding")
        expected_spatial = torch.arange(77, device=spatial.device)[None, None, :].expand_as(spatial)
        expected_spatial = expected_spatial.masked_fill(~mask, -1)
        if not torch.equal(spatial, expected_spatial):
            raise ValueError("Native keyframe spatial indices must preserve the ordered 7x11 grid")
        frame_raw = raw[..., 0]
        if (not torch.equal(raw, frame_raw[..., None].expand_as(raw))
                or torch.any(frame_raw[frame_mask] < 0) or torch.any(frame_raw[~frame_mask] != -1)):
            raise ValueError("Native keyframe raw indices must be constant per frame with -1 padding")
        if torch.any((frame_raw[:, 1:] <= frame_raw[:, :-1]) & frame_mask[:, 1:]):
            raise ValueError("Native keyframe raw frames must be chronological and unique")
        position = self.position(memory_raw_frame_indices, memory_spatial_indices, memory_mask)
        position = F.silu(self.position_proj(position))
        # Clear padding before projection so NaNs/large values in unused slots
        # cannot leak through masked attention or poison parameter gradients.
        features = memory_features.masked_fill(~memory_mask[..., None], 0).to(position.dtype)
        if self.normalize_features:
            features = F.rms_norm(features, (self.feature_dim,), eps=self.feature_norm_eps)
        encoded = self.memory_proj(torch.cat((features, position), dim=-1))
        return encoded.masked_fill(~memory_mask[..., None], 0)


class NativeKeyframeActionTransformerModel(FrameSampActionTransformerModel):
    """Shared FrameSamp action computation, with native-grid encoding and ABI."""

    def __init__(self, action_header, model_cfg):
        # Construct once from the clean Psi0 header, preserving all transferred
        # parameter prefixes and the original memory initialization order.
        nn.Module.__init__(self)
        self.config = action_header.config
        self.combined_temb = action_header.combined_temb
        self.time_ins_embed = action_header.time_ins_embed
        self.obs_proj = action_header.obs_proj
        self.action_proj_in = action_header.action_proj_in
        self.action_proj_out = action_header.action_proj_out
        self.register_buffer("framesamp_contract_version", torch.tensor(2, dtype=torch.int64), persistent=True)
        self.register_buffer("keyframe_native_contract_version", torch.tensor(NATIVE_KEYFRAME_ABI, dtype=torch.int64), persistent=True)
        self.hidden_size = int(model_cfg.hidden_dim)
        self.memory_encoder = NativeKeyframeMemoryEncoder(
            feature_dim=model_cfg.framesamp_feature_dim,
            position_dim=model_cfg.framesamp_position_dim,
            hidden_size=self.hidden_size,
            memory_budget=model_cfg.framesamp_memory_budget,
        )
        self.transformer_blocks = nn.ModuleList([
            FrameSampVLATransformerBlock(
                block, hidden_size=self.hidden_size,
                memory_heads=model_cfg.framesamp_memory_heads,
                memory_head_dim=model_cfg.framesamp_memory_head_dim,
                modulation_init=model_cfg.framesamp_modulation_init,
            ) for block in action_header.transformer_blocks
        ])
        self._active_memory = None

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        native = state_dict.get(prefix + "keyframe_native_contract_version")
        expected = torch.tensor(NATIVE_KEYFRAME_ABI, dtype=torch.int64)
        if (not isinstance(native, torch.Tensor) or native.dtype != torch.int64
                or native.shape != expected.shape or not torch.equal(native.cpu(), expected)):
            raise RuntimeError("Checkpoint is missing or incompatible with the native 7x77 keyframe ABI")
        frame_version = state_dict.get(prefix + "framesamp_contract_version")
        if (not isinstance(frame_version, torch.Tensor) or frame_version.dtype != torch.int64
                or frame_version.shape != torch.Size([]) or int(frame_version.item()) != 2):
            raise RuntimeError("Native keyframe checkpoint requires raw-frame FrameSamp contract v2")
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class _NativeRestoreFeatureBank(NativeKeyframeBank):
    """Adapt the native bank to the existing audited full-checkpoint restorer."""

    def __init__(self, root, *, expected_task, expected_dataset_revision, expected_scope, expected_manifest_sha256):
        super().__init__(root, expected_manifest_sha256=expected_manifest_sha256)
        if (self.manifest["task"] != expected_task
                or self.manifest["dataset_revision"] != expected_dataset_revision
                or self.manifest["scope"] != expected_scope or expected_scope != "full"):
            raise ValueError("Native checkpoint deployment bank binding mismatch")
        self.contract = json.loads((self.root / "source_contract.json").read_text())


class Psi0KeyframeNativeModel(Psi0FrameSampModel):
    ENTRY_SCHEMA_VERSION = "psi0-keyframe-native-model/v1"
    FEATURE_BANK_CLASS = _NativeRestoreFeatureBank

    def __init__(self, model_cfg, vlm_model):
        if (not getattr(model_cfg, "native_keyframe_enabled", False)
                or getattr(model_cfg, "native_keyframe_contract_version", None) != 1
                or model_cfg.framesamp_entry_probe_only or model_cfg.use_dit):
            raise ValueError("Native keyframe model requires its independent formal configuration")
        Psi0Model.__init__(self, model_cfg=model_cfg, vlm_model=vlm_model)
        self.framesamp_model_cfg = model_cfg
        self._framesamp_forward_seen = False
        self._framesamp_entry_probe_grad_seen = False
        self._framesamp_memory_grad_seen = False
        self.action_header = NativeKeyframeActionTransformerModel(self.action_header, model_cfg)
        self.vlm_model.requires_grad_(False)
        self.vlm_model.eval()
        def record_gradient(gradient):
            if not self._framesamp_memory_grad_seen:
                if torch.isfinite(gradient).all() and gradient.abs().sum() > 0:
                    self._framesamp_memory_grad_seen = True
            return gradient
        for block in self.action_header.transformer_blocks:
            block.memory_to_scale_shift.weight.register_hook(record_gradient)
        self._emit_marker("constructed")

    def train(self, mode=True):
        super().train(mode)
        self.vlm_model.eval()
        return self

    @classmethod
    def from_pretrained(cls, run_dir, ckpt_step, launch_config, device, **kwargs):
        model_cfg, data_cfg = launch_config.model, launch_config.data
        if (Path(model_cfg.native_keyframe_root).resolve() != Path(data_cfg.native_keyframe_root).resolve()
                or model_cfg.native_keyframe_manifest_sha256 != data_cfg.native_keyframe_manifest_sha256
                or model_cfg.native_keyframe_selector != data_cfg.native_keyframe_selector):
            raise ValueError("Native model/data checkpoint bindings disagree")
        # The parent restorer audits the checkpoint, VLM and full action state.
        # Redirect only its bank lookup; preserve caller configs and original
        # pooled-bank provenance used to produce causal selection tables.
        adapted = copy.deepcopy(launch_config)
        adapted.model.framesamp_feature_manifest = str(Path(model_cfg.native_keyframe_root) / "manifest.json")
        adapted.model.framesamp_feature_manifest_sha256 = model_cfg.native_keyframe_manifest_sha256
        adapted.data.framesamp_feature_root = data_cfg.native_keyframe_root
        adapted.data.framesamp_feature_manifest_sha256 = data_cfg.native_keyframe_manifest_sha256
        model = super().from_pretrained(run_dir, ckpt_step, adapted, device, **kwargs)
        model.framesamp_model_cfg = model_cfg
        return model

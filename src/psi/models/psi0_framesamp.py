from __future__ import annotations

import json
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration

from psi.models.psi0 import (
    ActionTransformerModelOutput,
    HumanFoundationModelOutput,
    Psi0Condition,
    Psi0Model,
)
from psi.data.lerobot.framesamp_dataset import FrameSampFeatureBank
from psi.data.lerobot.framesamp_features import sha256_file

MEMORY_FIELD_NAMES = (
    "memory_features",
    "memory_mask",
    "memory_raw_frame_indices",
    "memory_spatial_indices",
)

# Flax/JAX ``lecun_normal`` is variance_scaling(scale=1, mode="fan_in",
# distribution="truncated_normal"). JAX compensates for truncation at two
# standard deviations with this constant so the resulting variance remains
# exactly 1 / fan_in.
_TRUNCATED_NORMAL_STDDEV_FACTOR = 0.87962566103423978


@torch.no_grad()
def _normal_kernel_(linear: nn.Linear, *, std: float) -> None:
    """Match RoboMME's Normal kernels and Flax's zero bias default."""

    nn.init.normal_(linear.weight, mean=0.0, std=std)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


@torch.no_grad()
def _lecun_normal_kernel_(linear: nn.Linear) -> None:
    """Match JAX LeCun-normal for a PyTorch Linear [out, in] kernel."""

    fan_in = int(linear.weight.shape[1])
    target_std = fan_in**-0.5
    sampling_std = target_std / _TRUNCATED_NORMAL_STDDEV_FACTOR
    nn.init.trunc_normal_(
        linear.weight,
        mean=0.0,
        std=sampling_std,
        a=-2.0 * sampling_std,
        b=2.0 * sampling_std,
    )
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


@dataclass(frozen=True)
class Psi0FrameSampCondition:
    views: torch.Tensor
    states: torch.Tensor
    traj2ds: Optional[torch.Tensor]
    attention_mask: Optional[torch.Tensor]
    vision_segment_ids: Optional[torch.Tensor]
    memory_hidden_states: torch.Tensor
    memory_mask: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.states.shape[0])


class FrameSampPosEmb3D(nn.Module):
    """Deterministic 768-D RoboMME temporal/spatial sinusoidal encoding."""

    def __init__(
        self,
        dim: int = 768,
        temporal_base: int = 10_000,
        spatial_base: int = 1_000,
    ) -> None:
        super().__init__()
        if dim % 6:
            raise ValueError("FrameSamp positional dimension must be divisible by six")
        width = dim // 6
        exponent = torch.arange(width, dtype=torch.float32) / (width - 1)
        self.register_buffer(
            "temporal_omega",
            torch.pow(torch.tensor(float(temporal_base)), -exponent),
            persistent=False,
        )
        self.register_buffer(
            "spatial_omega",
            torch.pow(torch.tensor(float(spatial_base)), -exponent),
            persistent=False,
        )

    def forward(
        self,
        raw_frame_indices: torch.Tensor,
        spatial_indices: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> torch.Tensor:
        raw_time = raw_frame_indices.clamp_min(0).float()
        spatial = spatial_indices.clamp_min(0)
        row = torch.div(spatial, 4, rounding_mode="floor").float() * 4 + 2
        column = (spatial % 4).float() * 4 + 2
        temporal = raw_time[..., None] * self.temporal_omega
        y = row[..., None] * self.spatial_omega
        x = column[..., None] * self.spatial_omega
        encoding = torch.cat(
            [
                temporal.sin(),
                temporal.cos(),
                y.sin(),
                y.cos(),
                x.sin(),
                x.cos(),
            ],
            dim=-1,
        )
        return encoding * memory_mask[..., None].to(encoding.dtype)


class FrameSampMemoryEncoder(nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int,
        position_dim: int,
        hidden_size: int,
        memory_budget: int,
        normalize_features: bool = False,
        feature_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_size = int(hidden_size)
        self.memory_budget = int(memory_budget)
        self.normalize_features = bool(normalize_features)
        self.feature_norm_eps = float(feature_norm_eps)
        if self.feature_norm_eps <= 0:
            raise ValueError("feature_norm_eps must be positive")
        self.position = FrameSampPosEmb3D(position_dim)
        self.position_proj = nn.Linear(position_dim, position_dim)
        self.memory_proj = nn.Linear(feature_dim + position_dim, hidden_size)
        _normal_kernel_(self.position_proj, std=0.02)
        _normal_kernel_(self.memory_proj, std=0.02)

    def forward(
        self,
        memory_features: torch.Tensor,
        memory_mask: torch.Tensor,
        memory_raw_frame_indices: torch.Tensor,
        memory_spatial_indices: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = memory_features.shape[0]
        if memory_features.shape != (
            batch_size,
            self.memory_budget,
            self.feature_dim,
        ):
            raise ValueError("memory_features has the wrong shape")
        expected_index_shape = (batch_size, self.memory_budget)
        if (
            memory_mask.shape != expected_index_shape
            or memory_raw_frame_indices.shape != expected_index_shape
            or memory_spatial_indices.shape != expected_index_shape
        ):
            raise ValueError("FrameSamp mask/index shape mismatch")
        if memory_mask.dtype != torch.bool:
            raise TypeError("memory_mask must be boolean")
        if any(
            value.dtype != torch.int64
            for value in (
                memory_raw_frame_indices,
                memory_spatial_indices,
            )
        ):
            raise TypeError("FrameSamp indices must use int64")
        valid = memory_mask
        if os.environ.get("PSI_FRAMESAMP_RUNTIME_VALIDATE") == "1":
            invalid = ~valid
            if (
                torch.any(memory_raw_frame_indices[invalid] != -1)
                or torch.any(memory_spatial_indices[invalid] != -1)
            ):
                raise ValueError("FrameSamp padding indices must equal -1")
            if torch.any(memory_raw_frame_indices[valid] < 0):
                raise ValueError("FrameSamp raw frame indices must be non-negative")
            if torch.any(
                (memory_spatial_indices[valid] < 0)
                | (memory_spatial_indices[valid] >= 16)
            ):
                raise ValueError("FrameSamp spatial index lies outside the 4x4 grid")

        position = self.position(
            memory_raw_frame_indices,
            memory_spatial_indices,
            memory_mask,
        ).to(device=memory_features.device)
        position = F.silu(self.position_proj(position))
        features = memory_features.to(position.dtype)
        if self.normalize_features:
            features = F.rms_norm(
                features,
                (self.feature_dim,),
                eps=self.feature_norm_eps,
            )
        encoded = self.memory_proj(torch.cat([features, position], dim=-1))
        return encoded * memory_mask[..., None].to(encoded.dtype)


def _apply_rope(
    tensor: torch.Tensor, positions: torch.Tensor, base: int = 10_000
) -> torch.Tensor:
    head_dim = tensor.shape[-1]
    if head_dim % 2:
        raise ValueError("RoPE head dimension must be even")
    exponent = (
        2.0
        / head_dim
        * torch.arange(head_dim // 2, device=tensor.device, dtype=torch.float32)
    )
    omega = torch.pow(torch.tensor(float(base), device=tensor.device), -exponent)
    angle = positions.to(device=tensor.device, dtype=torch.float32)[..., None] * omega
    cosine = angle.cos().to(tensor.dtype)
    sine = angle.sin().to(tensor.dtype)
    first, second = tensor.chunk(2, dim=-1)
    return torch.cat(
        [first * cosine - second * sine, second * cosine + first * sine], dim=-1
    ).to(tensor.dtype)


class FrameSampMemoryAttention(nn.Module):
    """Action-query, single-KV-head memory attention."""

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int) -> None:
        super().__init__()
        if hidden_size != num_heads * head_dim:
            raise ValueError("memory attention heads do not span hidden_size")
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.rms = nn.RMSNorm(hidden_size, eps=1e-6)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, head_dim, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        for projection in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            _lecun_normal_kernel_(projection)

    def forward(
        self,
        action_hidden_states: torch.Tensor,
        memory_hidden_states: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, action_length, _ = action_hidden_states.shape
        memory_length = memory_hidden_states.shape[1]
        action = self.rms(action_hidden_states)
        memory = self.rms(memory_hidden_states)
        query = self.q_proj(action).reshape(
            batch_size, action_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = self.k_proj(memory).reshape(
            batch_size, memory_length, 1, self.head_dim
        ).transpose(1, 2)
        value = self.v_proj(memory).reshape(
            batch_size, memory_length, 1, self.head_dim
        ).transpose(1, 2)
        query_positions = torch.arange(
            memory_length,
            memory_length + action_length,
            device=query.device,
        )[None, None, :]
        key_positions = torch.arange(memory_length, device=key.device)[None, None, :]
        query = _apply_rope(query, query_positions)
        key = _apply_rope(key, key_positions)
        key = key.expand(-1, self.num_heads, -1, -1)
        value = value.expand(-1, self.num_heads, -1, -1)

        has_memory = memory_mask.any(dim=-1)
        allowed = memory_mask[:, None, None, :].expand(
            -1, self.num_heads, action_length, -1
        ).clone()
        # SDPA needs at least one finite key. Empty rows use a dummy key, then
        # are multiplied to exact zero below and bypassed by the block.
        allowed[..., 0] |= (~has_memory)[:, None, None]
        observer = getattr(self, "_attention_observer", None)
        if observer is not None:
            observer(query.detach(), key.detach(), allowed, action_length=action_length, valid_memory=memory_mask, value=value.detach())
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=allowed,
            dropout_p=0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(
            batch_size, action_length, self.hidden_size
        )
        attended = self.out_proj(attended)
        return attended * has_memory[:, None, None].to(attended.dtype)


class FrameSampVLATransformerBlock(nn.Module):
    """A transferred Psi0 block with FrameSamp modulation before action FFN."""

    def __init__(
        self,
        legacy_block: nn.Module,
        *,
        hidden_size: int,
        memory_heads: int,
        memory_head_dim: int,
        modulation_init: str,
    ) -> None:
        super().__init__()
        self.context_pre_only = legacy_block.context_pre_only
        self.norm1_act = legacy_block.norm1_act
        self.norm1_obs = legacy_block.norm1_obs
        self.attn = legacy_block.attn
        self.norm2_act = legacy_block.norm2_act
        self.ff_act = legacy_block.ff_act
        self.norm2_obs = legacy_block.norm2_obs
        self.ff_obs = legacy_block.ff_obs
        self._chunk_size = legacy_block._chunk_size
        self._chunk_dim = legacy_block._chunk_dim
        self.memory_attention = FrameSampMemoryAttention(
            hidden_size, memory_heads, memory_head_dim
        )
        # RoboMME applies a dedicated conditional RMSNorm to the residual
        # stream, then runs the policy block's ordinary pre-FFN norm.  Keeping
        # this as a separate norm is material; folding both modulations into
        # norm2_act changes the transferred computation.
        self.memory_ffn_rms = nn.RMSNorm(
            hidden_size, eps=1e-6, elementwise_affine=False
        )
        self.memory_to_scale_shift = nn.Linear(hidden_size, hidden_size * 2)
        if modulation_init == "robomme_normal_std_0_002":
            _normal_kernel_(self.memory_to_scale_shift, std=0.002)
        elif modulation_init == "psi0_zero_ablation":
            # Deliberately retained only as a named Psi-0-specific ablation.
            nn.init.zeros_(self.memory_to_scale_shift.weight)
            nn.init.zeros_(self.memory_to_scale_shift.bias)
        else:
            raise ValueError(f"Unknown FrameSamp modulation init: {modulation_init}")

    def forward(
        self,
        action_hidden_states: torch.Tensor,
        obs_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        memory_hidden_states: torch.Tensor,
        memory_mask: torch.Tensor,
        obs_token_mask: Optional[torch.Tensor] = None,
        joint_attention_kwargs=None,
    ):
        (
            norm_action_hidden_states,
            gate_msa_act,
            shift_mlp_act,
            scale_mlp_act,
            gate_mlp_act,
        ) = self.norm1_act(action_hidden_states, emb=temb)
        if self.context_pre_only:
            norm_obs_hidden_states = self.norm1_obs(
                obs_hidden_states, temb[:, -1] if len(temb.shape) > 2 else temb
            )
            gate_msa_obs = shift_mlp_obs = scale_mlp_obs = gate_mlp_obs = None
        else:
            (
                norm_obs_hidden_states,
                gate_msa_obs,
                shift_mlp_obs,
                scale_mlp_obs,
                gate_mlp_obs,
            ) = self.norm1_obs(
                obs_hidden_states,
                emb=temb[:, -1] if len(temb.shape) > 2 else temb,
            )

        act_attn_output, obs_attn_output = self.attn(
            hidden_states=norm_action_hidden_states,
            encoder_hidden_states=norm_obs_hidden_states,
            attention_mask=obs_token_mask,
        )
        action_hidden_states = action_hidden_states + gate_msa_act * act_attn_output

        memory_context = self.memory_attention(
            action_hidden_states, memory_hidden_states, memory_mask
        )
        memory_scale, memory_shift = self.memory_to_scale_shift(
            memory_context
        ).chunk(2, dim=-1)
        has_memory = memory_mask.any(dim=-1)[:, None, None]
        memory_scale = memory_scale * has_memory.to(memory_scale.dtype)
        memory_shift = memory_shift * has_memory.to(memory_shift.dtype)
        memory_modulated = self.memory_ffn_rms(action_hidden_states)
        memory_modulated = (
            memory_modulated * (1 + memory_scale) + memory_shift
        )
        norm_action_hidden_states = self.norm2_act(memory_modulated)
        norm_action_hidden_states = (
            norm_action_hidden_states * (1 + scale_mlp_act) + shift_mlp_act
        )
        ff_action_output = gate_mlp_act * self.ff_act(norm_action_hidden_states)
        action_hidden_states = action_hidden_states + ff_action_output

        if self.context_pre_only:
            obs_hidden_states = None
        else:
            if any(
                value is None
                for value in (
                    gate_msa_obs,
                    shift_mlp_obs,
                    scale_mlp_obs,
                    gate_mlp_obs,
                    self.norm2_obs,
                    self.ff_obs,
                )
            ):
                raise RuntimeError("Transferred observation branch is incomplete")
            obs_hidden_states = obs_hidden_states + gate_msa_obs * obs_attn_output
            norm_obs_hidden_states = self.norm2_obs(obs_hidden_states)
            norm_obs_hidden_states = (
                norm_obs_hidden_states * (1 + scale_mlp_obs) + shift_mlp_obs
            )
            obs_hidden_states = (
                obs_hidden_states + gate_mlp_obs * self.ff_obs(norm_obs_hidden_states)
            )
        return action_hidden_states, obs_hidden_states


class FrameSampActionTransformerModel(nn.Module):
    """Drop-in action header that keeps all legacy parameter key prefixes."""

    def __init__(self, action_header: nn.Module, model_cfg) -> None:
        super().__init__()
        self.config = action_header.config
        self.combined_temb = action_header.combined_temb
        self.time_ins_embed = action_header.time_ins_embed
        self.obs_proj = action_header.obs_proj
        self.action_proj_in = action_header.action_proj_in
        self.action_proj_out = action_header.action_proj_out
        self.register_buffer(
            "framesamp_contract_version",
            torch.tensor(2, dtype=torch.int64),
            persistent=True,
        )
        self.hidden_size = int(model_cfg.hidden_dim)
        self.memory_encoder = FrameSampMemoryEncoder(
            feature_dim=model_cfg.framesamp_feature_dim,
            position_dim=model_cfg.framesamp_position_dim,
            hidden_size=self.hidden_size,
            memory_budget=model_cfg.framesamp_memory_budget,
            normalize_features=getattr(
                model_cfg,
                "framesamp_normalize_memory_features",
                False,
            ),
            feature_norm_eps=getattr(
                model_cfg,
                "framesamp_memory_feature_norm_eps",
                1e-6,
            ),
        )
        self.transformer_blocks = nn.ModuleList(
            [
                FrameSampVLATransformerBlock(
                    block,
                    hidden_size=self.hidden_size,
                    memory_heads=model_cfg.framesamp_memory_heads,
                    memory_head_dim=model_cfg.framesamp_memory_head_dim,
                    modulation_init=model_cfg.framesamp_modulation_init,
                )
                for block in action_header.transformer_blocks
            ]
        )
        self._active_memory: Optional[tuple[torch.Tensor, torch.Tensor]] = None

    def memory_parameter_names(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, _ in self.named_parameters()
            if name.startswith("memory_encoder.") or ".memory_" in name
        )

    def encode_memory(self, **memory_inputs) -> torch.Tensor:
        return self.memory_encoder(**memory_inputs)

    @contextmanager
    def use_preprojected_memory(
        self, memory_hidden_states: torch.Tensor, memory_mask: torch.Tensor
    ):
        if self._active_memory is not None:
            raise RuntimeError("Nested/concurrent FrameSamp memory context is unsafe")
        self._active_memory = (memory_hidden_states, memory_mask)
        try:
            yield
        finally:
            self._active_memory = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        pooled_projections: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        block_controlnet_hidden_states: Optional[List] = None,
        joint_attention_kwargs: Optional[dict[str, Any]] = None,
        vlm_attn_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        skip_layers: Optional[List[int]] = None,
        *,
        memory_features: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        memory_raw_frame_indices: Optional[torch.Tensor] = None,
        memory_spatial_indices: Optional[torch.Tensor] = None,
        memory_hidden_states: Optional[torch.Tensor] = None,
    ) -> ActionTransformerModelOutput:
        if joint_attention_kwargs is None:
            raise ValueError("joint_attention_kwargs is required")
        raw_inputs = (
            memory_features,
            memory_raw_frame_indices,
            memory_spatial_indices,
        )
        raw_present = any(value is not None for value in raw_inputs)
        if memory_hidden_states is not None and raw_present:
            raise ValueError("provide raw or preprojected memory, never both")
        if memory_hidden_states is None and raw_present:
            if any(value is None for value in raw_inputs) or memory_mask is None:
                raise ValueError("all FrameSamp memory inputs are required")
            memory_hidden_states = self.memory_encoder(
                memory_features=memory_features,
                memory_mask=memory_mask,
                memory_raw_frame_indices=memory_raw_frame_indices,
                memory_spatial_indices=memory_spatial_indices,
            )
        if memory_hidden_states is None and self._active_memory is not None:
            memory_hidden_states, memory_mask = self._active_memory
        if memory_hidden_states is None or memory_mask is None:
            raise ValueError("FrameSamp action header cannot run without memory")
        if (
            memory_hidden_states.ndim != 3
            or memory_hidden_states.shape[1:] != (
                self.memory_encoder.memory_budget,
                self.hidden_size,
            )
            or memory_mask.shape != memory_hidden_states.shape[:2]
            or memory_mask.dtype != torch.bool
            or memory_mask.device != memory_hidden_states.device
        ):
            raise ValueError("Preprojected FrameSamp memory has an invalid contract")
        if self.config.n_conditions > 0:
            if joint_attention_kwargs["traj2ds"].shape[1] != self.config.n_conditions:
                raise ValueError("model n_conditions does not match")
        temb = (
            self.time_ins_embed(timestep, pooled_projections)
            if self.combined_temb
            else self.time_ins_embed(timestep)
        )
        noisy_action = joint_attention_kwargs.pop("action_hidden_embeds")
        if (
            memory_hidden_states.shape[0] != noisy_action.shape[0]
            or memory_hidden_states.device != noisy_action.device
        ):
            raise ValueError("FrameSamp memory/action batch or device mismatch")
        action_hidden_states = self.action_proj_in(noisy_action)
        obs_hidden_states, obs_token_mask = self.obs_proj(
            views=joint_attention_kwargs["views"],
            obs=joint_attention_kwargs["obs"],
            traj2ds=joint_attention_kwargs["traj2ds"],
            text_embeddings=pooled_projections,
            vlm_attn_mask=vlm_attn_mask,
        )
        for index, block in enumerate(self.transformer_blocks):
            if skip_layers is not None and index in skip_layers:
                continue
            action_hidden_states, obs_hidden_states = block(
                action_hidden_states=action_hidden_states,
                obs_hidden_states=obs_hidden_states,
                temb=temb,
                memory_hidden_states=memory_hidden_states,
                memory_mask=memory_mask,
                obs_token_mask=obs_token_mask,
            )
        action_output = self.action_proj_out(x=action_hidden_states, t=temb)
        return ActionTransformerModelOutput(action=action_output)


class Psi0FrameSampModel(Psi0Model):
    """Psi0 with a separately encoded FrameSamp memory modulation path."""

    ENTRY_SCHEMA_VERSION = "psi0-framesamp-entry/v2"
    FEATURE_BANK_CLASS = FrameSampFeatureBank

    @classmethod
    def from_pretrained(
        cls,
        run_dir,
        ckpt_step,
        launch_config,
        device,
        checkpoint_model_path=None,
        expected_model_sha256: Optional[str] = None,
    ):
        """Strictly restore a full FrameSamp checkpoint for deployment."""

        if launch_config.model.framesamp_entry_probe_only:
            raise RuntimeError("Entry-probe checkpoints are not deployable")
        lexical_run_dir = Path(run_dir).expanduser()
        if lexical_run_dir.is_symlink():
            raise ValueError(f"FrameSamp run_dir must not be a symlink: {run_dir}")
        run_path = lexical_run_dir.resolve(strict=True)
        if not run_path.is_dir():
            raise ValueError(f"Pretrained run directory does not exist: {run_path}")
        if checkpoint_model_path is None:
            from psi.deploy.simple_server_audit import inspect_simple_checkpoint

            checkpoint_audit = inspect_simple_checkpoint(run_path, int(ckpt_step))
            checkpoint_path = checkpoint_audit["artifacts"]["model"]["path"]
            audited_sha256 = checkpoint_audit["model_sha256"]
            if (
                expected_model_sha256 is not None
                and expected_model_sha256 != audited_sha256
            ):
                raise RuntimeError("Requested checkpoint SHA differs from COMPLETE audit")
            expected_model_sha256 = audited_sha256
        else:
            checkpoint_path = os.fspath(checkpoint_model_path)
            if expected_model_sha256 is None:
                raise ValueError(
                    "An explicit checkpoint_model_path requires expected_model_sha256"
                )
        if not re.fullmatch(r"[0-9a-f]{64}", str(expected_model_sha256)):
            raise ValueError("expected_model_sha256 must be lowercase hexadecimal")
        resolved_checkpoint = Path(checkpoint_path).expanduser().resolve(strict=True)
        if (
            not resolved_checkpoint.is_file()
            or sha256_file(resolved_checkpoint) != expected_model_sha256
        ):
            raise RuntimeError("FrameSamp checkpoint identity mismatch")
        checkpoint_path = os.fspath(resolved_checkpoint)
        manifest_input = os.fspath(launch_config.model.framesamp_feature_manifest)
        manifest_path = os.path.realpath(manifest_input)
        if (
            os.path.islink(manifest_input)
            or not os.path.isfile(manifest_path)
            or sha256_file(manifest_path)
            != launch_config.model.framesamp_feature_manifest_sha256
        ):
            raise RuntimeError("Deployment feature manifest identity mismatch")
        data_cfg = launch_config.data
        feature_bank = cls.FEATURE_BANK_CLASS(
            data_cfg.framesamp_feature_root,
            expected_task=str(data_cfg.train_repo_ids[0]),
            expected_dataset_revision=data_cfg.framesamp_dataset_revision,
            expected_scope="full",
            expected_manifest_sha256=data_cfg.framesamp_feature_manifest_sha256,
        )
        if (
            (feature_bank.root / "manifest.json").resolve(strict=True)
            != Path(manifest_path)
            or data_cfg.framesamp_feature_manifest_sha256
            != launch_config.model.framesamp_feature_manifest_sha256
        ):
            raise RuntimeError("Model/data feature-bank bindings disagree")

        from safetensors.torch import load_file

        state_dict = load_file(checkpoint_path, device="cpu")
        vlm_state_dict = {}
        action_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("vlm_model."):
                vlm_state_dict[key.removeprefix("vlm_model.")] = value
            elif key.startswith("action_header."):
                action_state_dict[key.removeprefix("action_header.")] = value
            else:
                raise RuntimeError(f"Unexpected top-level checkpoint key: {key}")
        configured_vlm_path = os.fspath(launch_config.model.model_name_or_path)
        if not os.path.isfile(os.path.join(configured_vlm_path, "config.json")):
            raise RuntimeError(
                "Formal FrameSamp restore requires a complete local Qwen checkpoint"
            )
        if Path(configured_vlm_path).resolve(strict=True) != Path(
            feature_bank.contract["encoder"]["checkpoint_dir"]
        ).resolve(strict=True):
            raise RuntimeError(
                "Online Qwen producer checkpoint differs from the offline feature bank"
            )
        config_source = configured_vlm_path
        vlm_config = AutoConfig.from_pretrained(config_source)
        # The offline feature bank is generated with SDPA. Keep the policy and
        # online producer on that exact backend to avoid backend-dependent
        # Qwen vision-merger drift.
        vlm_config._attn_implementation = "sdpa"
        vlm_config.dtype = torch.bfloat16
        vlm_config.vision_config.dtype = torch.bfloat16
        vlm_config.text_config.dtype = torch.bfloat16
        vlm_model = Qwen3VLForConditionalGeneration(vlm_config).to(
            dtype=torch.bfloat16
        )
        embedding_key = "model.language_model.embed_tokens.weight"
        if embedding_key not in vlm_state_dict:
            raise RuntimeError("Checkpoint is missing the Qwen embedding table")
        if vlm_state_dict[embedding_key].shape[0] != vlm_model.lm_head.weight.shape[0]:
            vlm_model.resize_token_embeddings(
                vlm_state_dict[embedding_key].shape[0],
                pad_to_multiple_of=192,
                mean_resizing=True,
            )
        if (
            "lm_head.weight" in vlm_state_dict
            and not torch.equal(
                vlm_state_dict["lm_head.weight"], vlm_state_dict[embedding_key]
            )
        ):
            raise RuntimeError("Checkpoint contains inconsistent tied Qwen embeddings")
        vlm_state_dict["lm_head.weight"] = vlm_state_dict[embedding_key]
        vlm_model.load_state_dict(vlm_state_dict, strict=True)
        vlm_model.tie_weights()

        model = cls(model_cfg=launch_config.model, vlm_model=vlm_model)
        model.framesamp_feature_contract = feature_bank.contract
        version = action_state_dict.get("framesamp_contract_version")
        if version is None or version.numel() != 1 or int(version.item()) != 2:
            raise RuntimeError(
                "Checkpoint is not strict FrameSamp v2; stride-24 checkpoints "
                "cannot be restored or continued"
            )
        memory_keys = set(model.action_header.memory_parameter_names())
        if not memory_keys.issubset(action_state_dict):
            raise RuntimeError(
                "Checkpoint is not a full FrameSamp model; missing memory keys: "
                f"{sorted(memory_keys - set(action_state_dict))}"
            )
        model.action_header.load_state_dict(action_state_dict, strict=True)
        model.vlm_processor = AutoProcessor.from_pretrained(config_source)

        if launch_config.model.noise_scheduler == "ddpm":
            from diffusers.schedulers.scheduling_ddim import DDIMScheduler

            scheduler = DDIMScheduler(
                num_train_timesteps=launch_config.model.train_diffusion_steps,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule="squaredcos_cap_v2",
                clip_sample=True,
                set_alpha_to_one=True,
                steps_offset=0,
                prediction_type="epsilon",
            )
        elif launch_config.model.noise_scheduler == "flow":
            from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
                FlowMatchEulerDiscreteScheduler,
            )

            scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=launch_config.model.train_diffusion_steps
            )
        else:
            raise ValueError(
                f"Unsupported noise scheduler: {launch_config.model.noise_scheduler}"
            )
        model.noise_scheduler = scheduler
        model.action_horizon = launch_config.model.action_chunk_size
        model.action_dim = launch_config.model.action_dim
        model.device = torch.device(device)
        model.to(model.device)
        model.eval()
        return model

    def __init__(self, model_cfg, vlm_model: Qwen3VLForConditionalGeneration):
        if not getattr(model_cfg, "framesamp_enabled", False):
            raise ValueError("Psi0FrameSampModel requires framesamp_enabled=true")
        if model_cfg.use_dit:
            raise ValueError("FrameSamp v2 supports ActionTransformerModel only")
        super().__init__(model_cfg=model_cfg, vlm_model=vlm_model)
        self.framesamp_model_cfg = model_cfg
        self._framesamp_forward_seen = False
        self._framesamp_entry_probe_grad_seen = False
        self._framesamp_memory_grad_seen = False
        if model_cfg.framesamp_entry_probe_only:
            self.framesamp_entry_probe = nn.Parameter(torch.zeros(()))

            def _record_probe_gradient(gradient: torch.Tensor) -> torch.Tensor:
                self._framesamp_entry_probe_grad_seen = True
                print(
                    "PSI0_FRAMESAMP_PROBE_GRAD_JSON="
                    + json.dumps(
                        {
                            "schema_version": self.ENTRY_SCHEMA_VERSION,
                            "event": "entry_probe_gradient",
                            "gradient_is_finite": bool(
                                torch.isfinite(gradient).all().item()
                            ),
                            "gradient_numel": int(gradient.numel()),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
                return gradient

            self.framesamp_entry_probe.register_hook(_record_probe_gradient)
        else:
            self.action_header = FrameSampActionTransformerModel(
                self.action_header, model_cfg
            )

            if os.environ.get("PSI_FRAMESAMP_FORMAL_EXIT_AFTER_UPDATE") == "1":
                def _record_memory_gradient(gradient: torch.Tensor) -> torch.Tensor:
                    if torch.isfinite(gradient).all() and gradient.abs().sum() > 0:
                        self._framesamp_memory_grad_seen = True
                    return gradient

                for block in self.action_header.transformer_blocks:
                    block.memory_to_scale_shift.weight.register_hook(
                        _record_memory_gradient
                    )
        self._emit_marker("constructed")

    def _emit_marker(self, event: str) -> None:
        payload = {
            "schema_version": self.ENTRY_SCHEMA_VERSION,
            "event": event,
            "model_class": type(self).__name__,
            "entry_probe_only": bool(self.framesamp_model_cfg.framesamp_entry_probe_only),
            "memory_clock": self.framesamp_model_cfg.framesamp_memory_clock,
            "raw_frame_stride": int(self.framesamp_model_cfg.framesamp_query_stride),
            "temporal_position_source": (
                self.framesamp_model_cfg.framesamp_temporal_position_source
            ),
            "modulation_init": self.framesamp_model_cfg.framesamp_modulation_init,
            "training_origin": self.framesamp_model_cfg.framesamp_training_origin,
            "memory_budget": int(self.framesamp_model_cfg.framesamp_memory_budget),
            "tokens_per_frame": int(self.framesamp_model_cfg.framesamp_tokens_per_frame),
            "feature_dim": int(self.framesamp_model_cfg.framesamp_feature_dim),
            "include_current": bool(self.framesamp_model_cfg.framesamp_include_current),
            "action_dim": int(self.framesamp_model_cfg.action_dim),
            "action_chunk_size": int(self.framesamp_model_cfg.action_chunk_size),
        }
        print(
            "PSI0_FRAMESAMP_MODEL_ACTIVE_JSON="
            + json.dumps(payload, sort_keys=True, separators=(",", ":")),
            flush=True,
        )

    @staticmethod
    def _memory_inputs(
        *,
        memory_features,
        memory_mask,
        memory_raw_frame_indices,
        memory_spatial_indices,
    ) -> dict[str, torch.Tensor]:
        values = {
            "memory_features": memory_features,
            "memory_mask": memory_mask,
            "memory_raw_frame_indices": memory_raw_frame_indices,
            "memory_spatial_indices": memory_spatial_indices,
        }
        missing = [name for name, value in values.items() if value is None]
        if missing:
            raise ValueError(f"Psi0FrameSampModel is missing {missing}")
        return values

    def forward(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        image_grid_thw,
        action_samples: torch.Tensor,
        states: torch.Tensor,
        timestep: torch.LongTensor,
        traj2ds: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        skip_layers: Optional[List[int]] = None,
        *,
        memory_features: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        memory_raw_frame_indices: Optional[torch.Tensor] = None,
        memory_spatial_indices: Optional[torch.Tensor] = None,
    ) -> HumanFoundationModelOutput:
        if not self._framesamp_forward_seen:
            self._emit_marker("first_training_forward")
            self._framesamp_forward_seen = True
        if self.framesamp_model_cfg.framesamp_entry_probe_only:
            output = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                action_samples=action_samples,
                states=states,
                timestep=timestep,
                traj2ds=traj2ds,
                return_dict=return_dict,
                skip_layers=skip_layers,
            )
            output.action = output.action + self.framesamp_entry_probe.to(
                dtype=output.action.dtype, device=output.action.device
            ) * 0.0
            return output

        memory_inputs = self._memory_inputs(
            memory_features=memory_features,
            memory_mask=memory_mask,
            memory_raw_frame_indices=memory_raw_frame_indices,
            memory_spatial_indices=memory_spatial_indices,
        )
        vlm_hidden_states = self.vlm_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            return_dict=True,
        ).hidden_states[-1]
        model_output = self.action_header(
            hidden_states=None,
            timestep=timestep,
            joint_attention_kwargs={
                "action_hidden_embeds": action_samples,
                "views": vlm_hidden_states.unsqueeze(1),
                "obs": states,
                "traj2ds": traj2ds,
            },
            vlm_attn_mask=attention_mask,
            return_dict=return_dict,
            skip_layers=skip_layers,
            **memory_inputs,
        )
        return HumanFoundationModelOutput(action=model_output.action)

    def encode_condition(
        self,
        observations: List[List[Image.Image]],
        states: torch.Tensor,
        instructions: List[str],
        traj2ds: Optional[torch.Tensor] = None,
        *,
        memory_features: torch.Tensor,
        memory_mask: torch.Tensor,
        memory_raw_frame_indices: torch.Tensor,
        memory_spatial_indices: torch.Tensor,
    ) -> Psi0FrameSampCondition:
        if self.framesamp_model_cfg.framesamp_entry_probe_only:
            raise RuntimeError("entry-probe model cannot perform FrameSamp inference")
        base: Psi0Condition = super().encode_condition(
            observations=observations,
            states=states,
            instructions=instructions,
            traj2ds=traj2ds,
        )
        memory_inputs = {
            name: value.to(self.device)
            for name, value in self._memory_inputs(
                memory_features=memory_features,
                memory_mask=memory_mask,
                memory_raw_frame_indices=memory_raw_frame_indices,
                memory_spatial_indices=memory_spatial_indices,
            ).items()
        }
        with torch.no_grad():
            memory_hidden = self.action_header.encode_memory(**memory_inputs)
        normal = lambda value: None if value is None else value.detach().clone()
        return Psi0FrameSampCondition(
            views=normal(base.views),
            states=normal(base.states),
            traj2ds=normal(base.traj2ds),
            attention_mask=normal(base.attention_mask),
            vision_segment_ids=normal(base.vision_segment_ids),
            memory_hidden_states=memory_hidden.detach().clone(),
            memory_mask=memory_inputs["memory_mask"].detach().clone(),
        )

    def _predict_velocity_from_condition(
        self,
        condition: Psi0FrameSampCondition,
        action_samples: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(condition, Psi0FrameSampCondition):
            raise TypeError("FrameSamp decoding requires Psi0FrameSampCondition")
        return self.action_header(
            hidden_states=None,
            timestep=timestep,
            joint_attention_kwargs={
                "action_hidden_embeds": action_samples,
                "views": condition.views,
                "obs": condition.states,
                "traj2ds": condition.traj2ds,
            },
            vlm_attn_mask=None,
            return_dict=True,
            memory_hidden_states=condition.memory_hidden_states,
            memory_mask=condition.memory_mask,
        ).action

    @torch.inference_mode()
    def predict_action_with_noise(
        self,
        observations,
        states,
        instructions,
        num_inference_steps,
        traj2ds,
        noise=None,
        generator=None,
        **memory_inputs,
    ):
        condition = self.encode_condition(
            observations=observations,
            states=states,
            instructions=instructions,
            traj2ds=traj2ds,
            **memory_inputs,
        )
        initial_noise = self._prepare_action_noise(
            condition.batch_size, noise, generator
        )
        actions = self.decode_noise(
            condition=condition,
            initial_noise=initial_noise,
            num_inference_steps=num_inference_steps,
        )
        return actions, initial_noise

    @torch.inference_mode()
    def predict_action(self, *args, **kwargs):
        actions, _ = self.predict_action_with_noise(*args, **kwargs)
        return actions

    def _legacy_rtc_memory_context(self, memory_inputs):
        if self.framesamp_model_cfg.framesamp_entry_probe_only:
            raise RuntimeError("entry-probe model cannot perform FrameSamp RTC")
        inputs = {
            name: value.to(self.device)
            for name, value in self._memory_inputs(**memory_inputs).items()
        }
        with torch.no_grad():
            hidden = self.action_header.encode_memory(**inputs).detach().clone()
        return self.action_header.use_preprojected_memory(hidden, inputs["memory_mask"])

    def predict_action_with_training_rtc_flow(self, *args, **kwargs):
        memory = {name: kwargs.pop(name, None) for name in MEMORY_FIELD_NAMES}
        with self._legacy_rtc_memory_context(memory):
            return super().predict_action_with_training_rtc_flow(*args, **kwargs)

    def _run_direct_rtc(self, method_name: str, args, kwargs):
        memory = {name: kwargs.pop(name, None) for name in MEMORY_FIELD_NAMES}
        with self._legacy_rtc_memory_context(memory):
            return getattr(super(), method_name)(*args, **kwargs)

    def predict_action_with_rtc_flow(self, *args, **kwargs):
        return self._run_direct_rtc("predict_action_with_rtc_flow", args, kwargs)

    def predict_action_with_rtc_flow_naive_inpaint(self, *args, **kwargs):
        return self._run_direct_rtc(
            "predict_action_with_rtc_flow_naive_inpaint", args, kwargs
        )

    def predict_action_with_rtc_flow_based_gradient(self, *args, **kwargs):
        return self._run_direct_rtc(
            "predict_action_with_rtc_flow_based_gradient", args, kwargs
        )

"""SAM3-v2 memory on the moving_dining_cart Psi0 (a7db4fb): native 7x77 keyframes + SAM3 object tokens.

Port of EmbodiedDyna sam3-objmem-20260924 (model code 49baf9f):
    FrameSampPosEmb3D / NativeKeyframePosEmb3D, NativeKeyframeMemoryEncoder, FrameSampMemoryAttention,
    FrameSampVLATransformerBlock, ObjectMemoryEncoder, ObjectGatedMemoryAttention (gate "zero_out_proj", v2).
Their math and initialisation are copied unchanged. Differences from the SIMPLE model:
  * the wrapped action header is this repository's a7db4fb ActionTransformerModel (observation-only
    positional encoding, final-LayerNorm output head, views [B,V,N,D]) exactly as used by the cart recipe;
  * object slots are fixed per SAM3 prompt (plan A) and their count is a config value (chip task: 4).

Per action block the memory context is  c = Attn(a, V) + g * Attn(a, O)  (V: 539 native tokens,
O: 7 x slots object tokens, g = 1, object out_proj zero-initialised), and c feeds a FiLM scale/shift on
the action-FFN input. Memory is never added to the residual stream.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from psi.models.psi0 import ActionTransformerModel, ActionTransformerModelOutput, HumanFoundationModelOutput, Psi0Model

MAX_FRAMES = 7
GRID_HW = (7, 11)
TOKENS_PER_FRAME = 77
FEATURE_DIM = 2048
VISUAL_BUDGET = MAX_FRAMES * TOKENS_PER_FRAME  # 539
GEOMETRY_DIM = 8
VISUAL_FIELDS = ("memory_features", "memory_mask", "memory_raw_frame_indices", "memory_spatial_indices")
OBJECT_FIELDS = ("memory_object_features", "memory_object_geometry", "memory_object_mask",
                 "memory_object_raw_frame_indices", "memory_object_slots")
MEMORY_FIELDS = VISUAL_FIELDS + OBJECT_FIELDS
OBJMEM_CONTRACT = (2, 7, 77, 7, 11)  # (object gate contract v2, frames, tokens, grid h, grid w)

_TRUNCATED_NORMAL_STDDEV_FACTOR = 0.87962566103423978


@torch.no_grad()
def _normal_kernel_(linear: nn.Linear, *, std: float) -> None:
    nn.init.normal_(linear.weight, mean=0.0, std=std)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


@torch.no_grad()
def _lecun_normal_kernel_(linear: nn.Linear) -> None:
    fan_in = int(linear.weight.shape[1])
    sampling_std = fan_in ** -0.5 / _TRUNCATED_NORMAL_STDDEV_FACTOR
    nn.init.trunc_normal_(linear.weight, mean=0.0, std=sampling_std, a=-2.0 * sampling_std, b=2.0 * sampling_std)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


class NativeKeyframePosEmb3D(nn.Module):
    """Raw-time sinusoid plus 7x11 cell centres in a 16x16 extent (768-D, RoboMME schedules)."""

    def __init__(self, dim: int = 768, temporal_base: int = 10_000, spatial_base: int = 1_000) -> None:
        super().__init__()
        if dim % 6:
            raise ValueError("positional dimension must be divisible by six")
        width = dim // 6
        exponent = torch.arange(width, dtype=torch.float32) / (width - 1)
        self.register_buffer("temporal_omega", torch.pow(torch.tensor(float(temporal_base)), -exponent), persistent=False)
        self.register_buffer("spatial_omega", torch.pow(torch.tensor(float(spatial_base)), -exponent), persistent=False)

    def forward(self, raw_frame_indices, spatial_indices, memory_mask):
        raw_time = raw_frame_indices.clamp_min(0).float()
        spatial = spatial_indices.clamp_min(0)
        row = (torch.div(spatial, GRID_HW[1], rounding_mode="floor").float() + 0.5) * (16.0 / GRID_HW[0])
        column = ((spatial % GRID_HW[1]).float() + 0.5) * (16.0 / GRID_HW[1])
        temporal = raw_time[..., None] * self.temporal_omega
        y = row[..., None] * self.spatial_omega
        x = column[..., None] * self.spatial_omega
        encoded = torch.cat((temporal.sin(), temporal.cos(), y.sin(), y.cos(), x.sin(), x.cos()), dim=-1)
        return encoded.masked_fill(~memory_mask[..., None], 0)


class NativeKeyframeMemoryEncoder(nn.Module):
    def __init__(self, *, feature_dim: int, position_dim: int, hidden_size: int, memory_budget: int) -> None:
        super().__init__()
        if memory_budget != VISUAL_BUDGET or feature_dim != FEATURE_DIM:
            raise ValueError("Native keyframe memory requires [539,2048] features")
        self.feature_dim, self.hidden_size, self.memory_budget = feature_dim, hidden_size, memory_budget
        self.position = NativeKeyframePosEmb3D(position_dim)
        self.position_proj = nn.Linear(position_dim, position_dim)
        self.memory_proj = nn.Linear(feature_dim + position_dim, hidden_size)
        _normal_kernel_(self.position_proj, std=0.02)
        _normal_kernel_(self.memory_proj, std=0.02)

    def forward(self, memory_features, memory_mask, memory_raw_frame_indices, memory_spatial_indices):
        batch = memory_features.shape[0]
        if memory_features.shape[1:] != (VISUAL_BUDGET, FEATURE_DIM):
            raise ValueError("memory_features must have shape [B,539,2048]")
        if any(t.shape != (batch, VISUAL_BUDGET) for t in (memory_mask, memory_raw_frame_indices, memory_spatial_indices)):
            raise ValueError("native memory mask/index shape mismatch")
        if memory_mask.dtype != torch.bool:
            raise TypeError("memory_mask must be boolean")
        mask = memory_mask.reshape(batch, MAX_FRAMES, TOKENS_PER_FRAME)
        frame_mask = mask[..., 0]
        if not torch.equal(mask, frame_mask[..., None].expand_as(mask)):
            raise ValueError("native memory mask must cover whole 77-token frames")
        if torch.any(frame_mask[:, 1:] & ~frame_mask[:, :-1]):
            raise ValueError("native memory frames must use right padding")
        position = self.position(memory_raw_frame_indices, memory_spatial_indices, memory_mask)
        position = F.silu(self.position_proj(position))
        features = memory_features.masked_fill(~memory_mask[..., None], 0).to(position.dtype)
        encoded = self.memory_proj(torch.cat((features, position), dim=-1))
        return encoded.masked_fill(~memory_mask[..., None], 0)


def _apply_rope(tensor: torch.Tensor, positions: torch.Tensor, base: int = 10_000) -> torch.Tensor:
    head_dim = tensor.shape[-1]
    exponent = 2.0 / head_dim * torch.arange(head_dim // 2, device=tensor.device, dtype=torch.float32)
    omega = torch.pow(torch.tensor(float(base), device=tensor.device), -exponent)
    angle = positions.to(device=tensor.device, dtype=torch.float32)[..., None] * omega
    cosine, sine = angle.cos().to(tensor.dtype), angle.sin().to(tensor.dtype)
    first, second = tensor.chunk(2, dim=-1)
    return torch.cat([first * cosine - second * sine, second * cosine + first * sine], dim=-1).to(tensor.dtype)


class MemoryAttention(nn.Module):
    """Action-query, single-KV-head memory attention (FrameSampMemoryAttention)."""

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int) -> None:
        super().__init__()
        if hidden_size != num_heads * head_dim:
            raise ValueError("memory attention heads do not span hidden_size")
        self.hidden_size, self.num_heads, self.head_dim = int(hidden_size), int(num_heads), int(head_dim)
        self.rms = nn.RMSNorm(hidden_size, eps=1e-6)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, head_dim, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        for projection in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            _lecun_normal_kernel_(projection)

    def forward(self, action_hidden_states, memory_hidden_states, memory_mask):
        batch, action_length, _ = action_hidden_states.shape
        memory_length = memory_hidden_states.shape[1]
        action = self.rms(action_hidden_states)
        memory = self.rms(memory_hidden_states)
        query = self.q_proj(action).reshape(batch, action_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(memory).reshape(batch, memory_length, 1, self.head_dim).transpose(1, 2)
        value = self.v_proj(memory).reshape(batch, memory_length, 1, self.head_dim).transpose(1, 2)
        query = _apply_rope(query, torch.arange(memory_length, memory_length + action_length, device=query.device)[None, None, :])
        key = _apply_rope(key, torch.arange(memory_length, device=key.device)[None, None, :])
        key = key.expand(-1, self.num_heads, -1, -1)
        value = value.expand(-1, self.num_heads, -1, -1)
        has_memory = memory_mask.any(dim=-1)
        allowed = memory_mask[:, None, None, :].expand(-1, self.num_heads, action_length, -1).clone()
        allowed[..., 0] |= (~has_memory)[:, None, None]  # SDPA needs one finite key; zeroed below
        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=allowed, dropout_p=0.0, is_causal=False)
        attended = self.out_proj(attended.transpose(1, 2).reshape(batch, action_length, self.hidden_size))
        return attended * has_memory[:, None, None].to(attended.dtype)


class ObjectGatedMemoryAttention(MemoryAttention):
    """Visual memory attention (same parameter names) plus a gated object branch (gate contract v2)."""

    def __init__(self, visual_attention: MemoryAttention, visual_budget: int) -> None:
        nn.Module.__init__(self)
        self.hidden_size, self.num_heads, self.head_dim = (visual_attention.hidden_size, visual_attention.num_heads,
                                                           visual_attention.head_dim)
        self.visual_budget = int(visual_budget)
        for name in ("rms", "q_proj", "k_proj", "v_proj", "out_proj"):
            setattr(self, name, getattr(visual_attention, name))
        self.object_attention = MemoryAttention(self.hidden_size, self.num_heads, self.head_dim)
        nn.init.zeros_(self.object_attention.out_proj.weight)
        self.object_gate = nn.Parameter(torch.ones(()))

    def gate_value(self) -> torch.Tensor:
        return self.object_gate

    def forward(self, action_hidden_states, memory_hidden_states, memory_mask):
        visual = super().forward(action_hidden_states, memory_hidden_states[:, :self.visual_budget],
                                 memory_mask[:, :self.visual_budget])
        objects = self.object_attention(action_hidden_states, memory_hidden_states[:, self.visual_budget:],
                                        memory_mask[:, self.visual_budget:])
        return visual + self.gate_value().to(visual.dtype) * objects


class MemoryVLATransformerBlock(nn.Module):
    """A transferred Psi0 VLATransformerBlock with memory FiLM before the action FFN (FrameSampVLATransformerBlock)."""

    def __init__(self, legacy_block: nn.Module, *, hidden_size: int, memory_heads: int, memory_head_dim: int,
                 modulation_init: str) -> None:
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
        self.memory_attention = MemoryAttention(hidden_size, memory_heads, memory_head_dim)
        self.memory_ffn_rms = nn.RMSNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.memory_to_scale_shift = nn.Linear(hidden_size, hidden_size * 2)
        if modulation_init == "robomme_normal_std_0_002":
            _normal_kernel_(self.memory_to_scale_shift, std=0.002)
        else:
            raise ValueError(f"Unknown memory modulation init: {modulation_init}")

    def forward(self, action_hidden_states, obs_hidden_states, temb, memory_hidden_states, memory_mask,
                obs_token_mask=None):
        norm_action, gate_msa_act, shift_mlp_act, scale_mlp_act, gate_mlp_act = self.norm1_act(action_hidden_states, emb=temb)
        obs_temb = temb[:, -1] if len(temb.shape) > 2 else temb
        if self.context_pre_only:
            norm_obs = self.norm1_obs(obs_hidden_states, obs_temb)
            gate_msa_obs = shift_mlp_obs = scale_mlp_obs = gate_mlp_obs = None
        else:
            norm_obs, gate_msa_obs, shift_mlp_obs, scale_mlp_obs, gate_mlp_obs = self.norm1_obs(obs_hidden_states, emb=obs_temb)
        act_attn_output, obs_attn_output = self.attn(hidden_states=norm_action, encoder_hidden_states=norm_obs,
                                                     attention_mask=obs_token_mask)
        action_hidden_states = action_hidden_states + gate_msa_act * act_attn_output

        memory_context = self.memory_attention(action_hidden_states, memory_hidden_states, memory_mask)
        memory_scale, memory_shift = self.memory_to_scale_shift(memory_context).chunk(2, dim=-1)
        has_memory = memory_mask.any(dim=-1)[:, None, None]
        memory_scale = memory_scale * has_memory.to(memory_scale.dtype)
        memory_shift = memory_shift * has_memory.to(memory_shift.dtype)
        modulated = self.memory_ffn_rms(action_hidden_states) * (1 + memory_scale) + memory_shift
        norm_action = self.norm2_act(modulated) * (1 + scale_mlp_act) + shift_mlp_act
        action_hidden_states = action_hidden_states + gate_mlp_act * self.ff_act(norm_action)

        if self.context_pre_only:
            obs_hidden_states = None
        else:
            obs_hidden_states = obs_hidden_states + gate_msa_obs * obs_attn_output
            norm_obs = self.norm2_obs(obs_hidden_states) * (1 + scale_mlp_obs) + shift_mlp_obs
            obs_hidden_states = obs_hidden_states + gate_mlp_obs * self.ff_obs(norm_obs)
        return action_hidden_states, obs_hidden_states


class ObjectMemoryEncoder(nn.Module):
    """Mask-pooled feature + raw-time/centroid sinusoid + geometry + slot identity."""

    def __init__(self, *, feature_dim: int, geometry_dim: int, slots: int, hidden_size: int, position_dim: int) -> None:
        super().__init__()
        self.feature_dim, self.geometry_dim, self.slots = feature_dim, geometry_dim, slots
        self.object_budget = MAX_FRAMES * slots
        self.position = NativeKeyframePosEmb3D(position_dim)  # only its omega buffers are used
        self.position_proj = nn.Linear(position_dim, position_dim)
        self.geometry_proj = nn.Linear(geometry_dim, position_dim)
        self.memory_proj = nn.Linear(feature_dim + 2 * position_dim, hidden_size)
        self.slot_embedding = nn.Parameter(torch.zeros(slots, hidden_size))
        for layer in (self.position_proj, self.geometry_proj, self.memory_proj):
            _normal_kernel_(layer, std=0.02)
        nn.init.normal_(self.slot_embedding, std=0.02)

    def encode_position(self, raw, geometry):
        time = raw.clamp_min(0).float()[..., None] * self.position.temporal_omega
        y = (geometry[..., 5].float() * 16)[..., None] * self.position.spatial_omega
        x = (geometry[..., 4].float() * 16)[..., None] * self.position.spatial_omega
        return torch.cat([time.sin(), time.cos(), y.sin(), y.cos(), x.sin(), x.cos()], dim=-1)

    def forward(self, memory_object_features, memory_object_geometry, memory_object_mask,
                memory_object_raw_frame_indices, memory_object_slots):
        batch = memory_object_features.shape[0]
        if (memory_object_features.shape != (batch, self.object_budget, self.feature_dim)
                or memory_object_geometry.shape != (batch, self.object_budget, self.geometry_dim)
                or any(t.shape != (batch, self.object_budget) for t in (
                    memory_object_mask, memory_object_raw_frame_indices, memory_object_slots))):
            raise ValueError("Object memory inputs have the wrong shape")
        mask = memory_object_mask[..., None]
        position = self.encode_position(memory_object_raw_frame_indices, memory_object_geometry)
        position = F.silu(self.position_proj(position.to(self.position_proj.weight.dtype)))
        geometry = F.silu(self.geometry_proj(memory_object_geometry.to(self.geometry_proj.weight.dtype)))
        features = memory_object_features.masked_fill(~mask, 0).to(position.dtype)
        encoded = self.memory_proj(torch.cat([features, position, geometry], dim=-1))
        encoded = encoded + self.slot_embedding[memory_object_slots.clamp(0, self.slots - 1)].to(encoded.dtype)
        return encoded.masked_fill(~mask, 0)


class ObjMemActionTransformerModel(nn.Module):
    """Wrap the cart ActionTransformerModel; keeps every transferred parameter name (transformer_blocks.N.*)."""

    def __init__(self, action_header: ActionTransformerModel, model_cfg) -> None:
        super().__init__()
        if action_header.state_drop_prob != 0.0 or getattr(model_cfg, "vlm_layer_indices", None) is not None:
            raise ValueError("Object memory is ported for the cart configuration (no state drop, last-layer views)")
        self.config = action_header.config
        self.combined_temb = action_header.combined_temb
        self.time_ins_embed = action_header.time_ins_embed
        self.obs_proj = action_header.obs_proj
        self.action_proj_in = action_header.action_proj_in
        self.action_proj_out = action_header.action_proj_out
        self.register_buffer("objmem_contract_version", torch.tensor(OBJMEM_CONTRACT, dtype=torch.int64), persistent=True)
        self.hidden_size = int(model_cfg.hidden_dim)
        # Same construction order as EmbodiedDyna: native encoder, per-block memory, then object branch.
        self.memory_encoder = NativeKeyframeMemoryEncoder(feature_dim=FEATURE_DIM, position_dim=model_cfg.memory_position_dim,
                                                          hidden_size=self.hidden_size, memory_budget=VISUAL_BUDGET)
        self.transformer_blocks = nn.ModuleList([
            MemoryVLATransformerBlock(block, hidden_size=self.hidden_size, memory_heads=model_cfg.memory_heads,
                                      memory_head_dim=model_cfg.memory_head_dim,
                                      modulation_init=model_cfg.memory_modulation_init)
            for block in action_header.transformer_blocks])
        self.slots = int(model_cfg.object_slots)
        self.object_budget = MAX_FRAMES * self.slots
        self.object_memory_encoder = ObjectMemoryEncoder(feature_dim=FEATURE_DIM, geometry_dim=GEOMETRY_DIM,
                                                         slots=self.slots, hidden_size=self.hidden_size,
                                                         position_dim=model_cfg.memory_position_dim)
        for block in self.transformer_blocks:
            block.memory_attention = ObjectGatedMemoryAttention(block.memory_attention, VISUAL_BUDGET)

    def memory_parameter_names(self):
        return tuple(name for name, _ in self.named_parameters()
                     if name.startswith(("memory_encoder.", "object_memory_encoder.")) or ".memory_" in name)

    def forward(self, hidden_states=None, encoder_hidden_states=None, pooled_projections=None, timestep=None,
                block_controlnet_hidden_states=None, joint_attention_kwargs=None, vlm_attn_mask=None, return_dict=True,
                skip_layers=None, *, memory_features, memory_mask, memory_raw_frame_indices, memory_spatial_indices,
                memory_object_features, memory_object_geometry, memory_object_mask, memory_object_raw_frame_indices,
                memory_object_slots) -> ActionTransformerModelOutput:
        if joint_attention_kwargs is None:
            raise ValueError("joint_attention_kwargs is required")
        visual = self.memory_encoder(memory_features=memory_features, memory_mask=memory_mask,
                                     memory_raw_frame_indices=memory_raw_frame_indices,
                                     memory_spatial_indices=memory_spatial_indices)
        objects = self.object_memory_encoder(memory_object_features=memory_object_features,
                                             memory_object_geometry=memory_object_geometry,
                                             memory_object_mask=memory_object_mask,
                                             memory_object_raw_frame_indices=memory_object_raw_frame_indices,
                                             memory_object_slots=memory_object_slots)
        packed = torch.cat([visual, objects.to(visual.dtype)], 1)
        packed_mask = torch.cat([memory_mask, memory_object_mask], 1)
        if self.config.n_conditions > 0 and joint_attention_kwargs["traj2ds"].shape[1] != self.config.n_conditions:
            raise ValueError("model n_conditions does not match")
        # Identical to the a7db4fb ActionTransformerModel default path, plus memory in every block.
        temb = self.time_ins_embed(timestep, pooled_projections) if self.combined_temb else self.time_ins_embed(timestep)
        noisy_action = joint_attention_kwargs.pop("action_hidden_embeds")
        if packed.shape[0] != noisy_action.shape[0]:
            raise ValueError("memory/action batch mismatch")
        action_hidden_states = self.action_proj_in(noisy_action)
        views = joint_attention_kwargs["views"]
        if views.shape[1] != 1:
            raise ValueError("Object memory expects last-layer views [B,1,N,D]")
        obs_hidden_states, obs_token_mask = self.obs_proj(views=views, obs=joint_attention_kwargs["obs"],
                                                          traj2ds=joint_attention_kwargs["traj2ds"],
                                                          text_embeddings=pooled_projections, vlm_attn_mask=vlm_attn_mask)
        for index, block in enumerate(self.transformer_blocks):
            if skip_layers is not None and index in skip_layers:
                continue
            action_hidden_states, obs_hidden_states = block(
                action_hidden_states=action_hidden_states, obs_hidden_states=obs_hidden_states, temb=temb,
                memory_hidden_states=packed, memory_mask=packed_mask, obs_token_mask=obs_token_mask)
        return ActionTransformerModelOutput(action=self.action_proj_out(x=action_hidden_states, t=temb))


class Psi0ObjMemModel(Psi0Model):
    """Cart Psi0 + SAM3-v2 memory. Memory tensors are passed as kwargs or through ``memory_context``."""

    def __init__(self, model_cfg, vlm_model) -> None:
        if model_cfg.use_dit or model_cfg.tune_vlm:
            raise ValueError("Object memory requires the ActionTransformerModel header and a frozen VLM")
        super().__init__(model_cfg=model_cfg, vlm_model=vlm_model)
        self.action_header = ObjMemActionTransformerModel(self.action_header, model_cfg)
        self._active_memory: Optional[dict[str, torch.Tensor]] = None

    @contextmanager
    def memory_context(self, memory: dict[str, torch.Tensor]):
        if self._active_memory is not None:
            raise RuntimeError("Nested memory context")
        missing = [name for name in MEMORY_FIELDS if name not in memory]
        if missing:
            raise KeyError(f"memory batch is missing {missing}")
        self._active_memory = {name: memory[name] for name in MEMORY_FIELDS}
        try:
            yield
        finally:
            self._active_memory = None

    def object_gates(self) -> list[float]:
        return [float(block.memory_attention.gate_value().detach()) for block in self.action_header.transformer_blocks]

    def object_out_proj_norms(self) -> list[float]:
        return [float(block.memory_attention.object_attention.out_proj.weight.detach().float().norm())
                for block in self.action_header.transformer_blocks]

    def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw, action_samples: torch.Tensor,
                states: torch.Tensor, timestep: torch.LongTensor, traj2ds: Optional[torch.Tensor] = None,
                pooled_projections: Optional[torch.Tensor] = None, return_dict: bool = True,
                skip_layers: Optional[List[int]] = None, **memory: Any) -> HumanFoundationModelOutput:
        if not memory:
            if self._active_memory is None:
                raise ValueError("Object-memory model called without memory")
            memory = self._active_memory
        if set(memory) != set(MEMORY_FIELDS):
            raise ValueError(f"unexpected memory inputs: {sorted(memory)}")
        all_hidden_states = self.vlm_model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values,
                                           image_grid_thw=image_grid_thw, output_hidden_states=True,
                                           return_dict=True).hidden_states
        vlm_hidden_states = all_hidden_states[-1]
        views = self._select_vlm_views(all_hidden_states)
        with torch.no_grad():
            tok_norm = vlm_hidden_states.detach().float().norm(dim=-1)
            if attention_mask is not None and attention_mask.shape == tok_norm.shape:
                m = attention_mask.to(tok_norm.dtype)
                vlm_feat_norm = (tok_norm * m).sum() / m.sum().clamp_min(1.0)
            else:
                vlm_feat_norm = tok_norm.mean()
        output = self.action_header(
            hidden_states=None, timestep=timestep, pooled_projections=pooled_projections,
            joint_attention_kwargs=dict(action_hidden_embeds=action_samples, views=views, obs=states, traj2ds=traj2ds),
            vlm_attn_mask=attention_mask, return_dict=return_dict, skip_layers=skip_layers,
            **{name: memory[name].to(action_samples.device) for name in MEMORY_FIELDS})
        return HumanFoundationModelOutput(action=output.action, vlm_feat_norm=vlm_feat_norm)

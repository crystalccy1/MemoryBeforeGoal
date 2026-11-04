"""Native 7x77 keyframe memory plus a zero-gated SAM3 object-memory branch.

Per action-DiT block the memory context becomes

    c = Attn(a, V) + gate * Attn(a, O)

where V are the 539 native keyframe tokens and O the SAM3 object tokens
(7 keyframes x ``slots``). Both gate modes make step 0 exactly the native 7x77
model; ``c`` feeds the existing scale/shift modulation.

* ``zero`` (contract v1): gate = tanh(g), g = 0. Deadlocks in practice: the
  branch gets no gradient while the gate is 0, so the gate only random-walks.
* ``zero_out_proj`` (contract v2): gate = g, g = 1, and the object attention's
  output projection starts at zero. That projection receives gradient on the
  first update, so the branch can learn (standard zero-init adapter).

Optional decision head (model_cfg.decision_head_enabled, default off; off = the exact model above, same state_dict):
a learned query attends over the packed memory tokens (visual and/or object, per ``decision_head_inputs``) and an MLP
gives ``decision_num_classes`` logits (which pad the remembered cue points to). Their softmax goes through a
zero-initialized linear map into the time/condition embedding ``temb`` of every action block, so step 0 is the model
above and the action head can learn to use the explicit decision. The trainer adds a cross-entropy on the answer label
(frames from the decision step on). The logits of the last forward are kept in ``last_decision_logits``.
"""
from __future__ import annotations

from typing import Any, List, Optional

import torch
from torch import nn
import torch.nn.functional as F

from psi.models.psi0 import HumanFoundationModelOutput
from psi.models.psi0_framesamp import (
    ActionTransformerModelOutput, FrameSampMemoryAttention, FrameSampPosEmb3D, _normal_kernel_,
)
from psi.models.psi0_keyframe_native import NativeKeyframeActionTransformerModel, Psi0KeyframeNativeModel

OBJECT_MEMORY_CONTRACT_VERSION = 1
GATE_CONTRACT = {"zero": 1, "zero_out_proj": 2}
VISUAL_BUDGET = 539
OBJECT_INPUTS = ("memory_object_features", "memory_object_geometry", "memory_object_mask",
                 "memory_object_raw_frame_indices", "memory_object_slots")
DECISION_HEAD_CONTRACT_VERSION = 1
DECISION_INPUTS = ("objects", "visual", "both")


class ObjectMemoryEncoder(nn.Module):
    """Mask-pooled feature + raw-time/centroid sinusoid + geometry + slot identity."""

    def __init__(self, *, feature_dim: int, geometry_dim: int, slots: int, hidden_size: int,
                 position_dim: int, object_budget: int) -> None:
        super().__init__()
        self.feature_dim, self.geometry_dim = feature_dim, geometry_dim
        self.slots, self.object_budget = slots, object_budget
        self.position = FrameSampPosEmb3D(position_dim)
        self.position_proj = nn.Linear(position_dim, position_dim)
        self.geometry_proj = nn.Linear(geometry_dim, position_dim)
        self.memory_proj = nn.Linear(feature_dim + 2 * position_dim, hidden_size)
        self.slot_embedding = nn.Parameter(torch.zeros(slots, hidden_size))
        for layer in (self.position_proj, self.geometry_proj, self.memory_proj):
            _normal_kernel_(layer, std=0.02)
        nn.init.normal_(self.slot_embedding, std=0.02)

    def encode_position(self, raw: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        # Same frequency schedules as the native memory: absolute raw time, and the
        # mask centroid expressed in the shared 16x16 spatial extent.
        time = raw.clamp_min(0).float()[..., None] * self.position.temporal_omega
        y = (geometry[..., 5].float() * 16)[..., None] * self.position.spatial_omega
        x = (geometry[..., 4].float() * 16)[..., None] * self.position.spatial_omega
        return torch.cat([time.sin(), time.cos(), y.sin(), y.cos(), x.sin(), x.cos()], dim=-1)

    def forward(self, memory_object_features, memory_object_geometry, memory_object_mask,
                memory_object_raw_frame_indices, memory_object_slots) -> torch.Tensor:
        batch = memory_object_features.shape[0]
        if (memory_object_features.shape != (batch, self.object_budget, self.feature_dim)
                or memory_object_geometry.shape != (batch, self.object_budget, self.geometry_dim)
                or any(t.shape != (batch, self.object_budget) for t in (
                    memory_object_mask, memory_object_raw_frame_indices, memory_object_slots))):
            raise ValueError("Object memory inputs have the wrong shape")
        if memory_object_mask.dtype != torch.bool:
            raise TypeError("memory_object_mask must be boolean")
        mask = memory_object_mask[..., None]
        position = self.encode_position(memory_object_raw_frame_indices, memory_object_geometry)
        position = F.silu(self.position_proj(position.to(self.position_proj.weight.dtype)))
        geometry = F.silu(self.geometry_proj(memory_object_geometry.to(self.geometry_proj.weight.dtype)))
        features = memory_object_features.masked_fill(~mask, 0).to(position.dtype)
        encoded = self.memory_proj(torch.cat([features, position, geometry], dim=-1))
        encoded = encoded + self.slot_embedding[memory_object_slots.clamp(0, self.slots - 1)].to(encoded.dtype)
        return encoded.masked_fill(~mask, 0)


class DecisionHead(nn.Module):
    """Memory -> answer logits: one learned query attends over the selected packed memory tokens (plus an always-valid
    null token, so an empty memory is well defined), then LayerNorm + MLP."""

    def __init__(self, *, hidden_size: int, num_classes: int, inputs: str, num_heads: int = 8) -> None:
        super().__init__()
        if inputs not in DECISION_INPUTS:
            raise ValueError(f"Unknown decision head inputs: {inputs}")
        self.inputs, self.num_classes = inputs, int(num_classes)
        self.norm = nn.LayerNorm(hidden_size)
        self.query = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.null = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.mlp = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size), nn.GELU(),
                                 nn.Linear(hidden_size, self.num_classes))
        nn.init.normal_(self.query, std=0.02)
        nn.init.normal_(self.null, std=0.02)

    def forward(self, packed: torch.Tensor, packed_mask: torch.Tensor) -> torch.Tensor:
        if self.inputs == "objects":
            tokens, mask = packed[:, VISUAL_BUDGET:], packed_mask[:, VISUAL_BUDGET:]
        elif self.inputs == "visual":
            tokens, mask = packed[:, :VISUAL_BUDGET], packed_mask[:, :VISUAL_BUDGET]
        else:
            tokens, mask = packed, packed_mask
        batch = tokens.shape[0]
        dtype = self.norm.weight.dtype
        tokens = torch.cat([self.null.expand(batch, -1, -1).to(dtype), tokens.to(dtype)], dim=1)
        mask = torch.cat([torch.ones((batch, 1), dtype=torch.bool, device=mask.device), mask], dim=1)
        keys = self.norm(tokens)
        pooled, _ = self.attention(self.query.expand(batch, -1, -1).to(keys.dtype), keys, keys,
                                   key_padding_mask=~mask, need_weights=False)
        return self.mlp(pooled[:, 0])


class ObjectGatedMemoryAttention(FrameSampMemoryAttention):
    """Keep the visual memory attention (and its parameter names); add a gated object branch."""

    def __init__(self, visual_attention: FrameSampMemoryAttention, visual_budget: int, gate_init: str = "zero"):
        nn.Module.__init__(self)
        if gate_init not in GATE_CONTRACT:
            raise ValueError(f"Unknown object gate init: {gate_init}")
        self.gate_init = gate_init
        self.hidden_size = visual_attention.hidden_size
        self.num_heads = visual_attention.num_heads
        self.head_dim = visual_attention.head_dim
        self.visual_budget = int(visual_budget)
        for name in ("rms", "q_proj", "k_proj", "v_proj", "out_proj"):
            setattr(self, name, getattr(visual_attention, name))
        self.object_attention = FrameSampMemoryAttention(self.hidden_size, self.num_heads, self.head_dim)
        if gate_init == "zero":
            self.object_gate = nn.Parameter(torch.zeros(()))
        else:
            nn.init.zeros_(self.object_attention.out_proj.weight)
            self.object_gate = nn.Parameter(torch.ones(()))

    def gate_value(self) -> torch.Tensor:
        return self.object_gate.tanh() if self.gate_init == "zero" else self.object_gate

    def forward(self, action_hidden_states, memory_hidden_states, memory_mask):
        if memory_hidden_states.shape[1] <= self.visual_budget:
            raise ValueError("Object-gated memory attention requires visual and object tokens")
        visual = super().forward(action_hidden_states, memory_hidden_states[:, :self.visual_budget],
                                 memory_mask[:, :self.visual_budget])
        objects = self.object_attention(action_hidden_states, memory_hidden_states[:, self.visual_budget:],
                                        memory_mask[:, self.visual_budget:])
        return visual + self.gate_value().to(visual.dtype) * objects


class NativeKeyframeObjectActionTransformerModel(NativeKeyframeActionTransformerModel):
    """Wrap an initialized native header; add the object encoder and gated branch."""

    def __init__(self, action_header: NativeKeyframeActionTransformerModel, model_cfg):
        nn.Module.__init__(self)
        if not isinstance(action_header, NativeKeyframeActionTransformerModel):
            raise TypeError("Object memory requires an initialized native keyframe header")
        for name in ("config", "combined_temb", "time_ins_embed", "obs_proj", "action_proj_in",
                     "action_proj_out", "memory_encoder", "transformer_blocks", "hidden_size"):
            setattr(self, name, getattr(action_header, name))
        self.register_buffer("framesamp_contract_version", action_header.framesamp_contract_version.clone())
        self.register_buffer("keyframe_native_contract_version",
                             action_header.keyframe_native_contract_version.clone())
        self.gate_init = getattr(model_cfg, "object_memory_gate_init", "zero")
        if self.gate_init not in GATE_CONTRACT:
            raise ValueError(f"Unknown object gate init: {self.gate_init}")
        self.register_buffer("object_memory_contract_version",
                             torch.tensor(GATE_CONTRACT[self.gate_init], dtype=torch.int64))
        slots = int(model_cfg.object_memory_slots)
        self.object_budget = 7 * slots
        self.object_memory_encoder = ObjectMemoryEncoder(
            feature_dim=model_cfg.framesamp_feature_dim, geometry_dim=model_cfg.object_memory_geometry_dim,
            slots=slots, hidden_size=self.hidden_size, position_dim=model_cfg.framesamp_position_dim,
            object_budget=self.object_budget)
        for block in self.transformer_blocks:
            block.memory_attention = ObjectGatedMemoryAttention(block.memory_attention, VISUAL_BUDGET, self.gate_init)
        self._active_memory = None
        self.decision_head = None
        self.last_decision_logits = None
        # Teacher forcing (training only): the trainer sets _decision_teacher_labels (per sample, -100 = none) before the
        # forward and clears it after; with probability decision_teacher_forcing a labelled sample gets the one-hot answer.
        self.decision_teacher_forcing = float(getattr(model_cfg, "decision_teacher_forcing", 0.0) or 0.0)
        self._decision_teacher_labels = None
        self.last_decision_forced_fraction = None
        if getattr(model_cfg, "decision_head_enabled", False):
            classes = int(model_cfg.decision_num_classes)
            self.decision_head = DecisionHead(hidden_size=self.hidden_size, num_classes=classes,
                                              inputs=model_cfg.decision_head_inputs)
            self.decision_embed = nn.Linear(classes, self.hidden_size)
            nn.init.zeros_(self.decision_embed.weight)
            nn.init.zeros_(self.decision_embed.bias)
            self.register_buffer("decision_head_contract_version",
                                 torch.tensor(DECISION_HEAD_CONTRACT_VERSION, dtype=torch.int64))

    def memory_parameter_names(self):
        return tuple(name for name, _ in self.named_parameters() if (
            name.startswith(("memory_encoder.", "object_memory_encoder.", "decision_head.", "decision_embed."))
            or ".memory_" in name))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        version = state_dict.get(prefix + "object_memory_contract_version")
        if (not isinstance(version, torch.Tensor) or version.dtype != torch.int64
                or int(version.item()) != GATE_CONTRACT[self.gate_init]):
            raise RuntimeError("Checkpoint is missing or incompatible with the object-memory contract")
        if self.decision_head is not None:
            decision = state_dict.get(prefix + "decision_head_contract_version")
            if (not isinstance(decision, torch.Tensor) or decision.dtype != torch.int64
                    or int(decision.item()) != DECISION_HEAD_CONTRACT_VERSION):
                raise RuntimeError("Checkpoint is missing or incompatible with the decision-head contract")
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def encode_object_memory(self, **inputs):
        return self.object_memory_encoder(**inputs)

    def forward(self, hidden_states, encoder_hidden_states=None, pooled_projections=None, timestep=None,
                block_controlnet_hidden_states=None, joint_attention_kwargs=None, vlm_attn_mask=None,
                return_dict=True, skip_layers=None, *, memory_features=None, memory_mask=None,
                memory_raw_frame_indices=None, memory_spatial_indices=None, memory_hidden_states=None,
                memory_object_features=None, memory_object_geometry=None, memory_object_mask=None,
                memory_object_raw_frame_indices=None, memory_object_slots=None,
                memory_object_hidden_states=None) -> ActionTransformerModelOutput:
        if joint_attention_kwargs is None:
            raise ValueError("joint_attention_kwargs is required")
        object_raw = (memory_object_features, memory_object_geometry, memory_object_raw_frame_indices,
                      memory_object_slots)
        if memory_hidden_states is None and memory_features is not None:
            memory_hidden_states = self.memory_encoder(
                memory_features=memory_features, memory_mask=memory_mask,
                memory_raw_frame_indices=memory_raw_frame_indices, memory_spatial_indices=memory_spatial_indices)
        if memory_object_hidden_states is None and any(v is not None for v in object_raw):
            if any(v is None for v in (*object_raw, memory_object_mask)):
                raise ValueError("All object memory inputs are required")
            memory_object_hidden_states = self.object_memory_encoder(
                memory_object_features=memory_object_features, memory_object_geometry=memory_object_geometry,
                memory_object_mask=memory_object_mask, memory_object_raw_frame_indices=memory_object_raw_frame_indices,
                memory_object_slots=memory_object_slots)
        if memory_hidden_states is not None:
            if memory_object_hidden_states is None or memory_mask is None or memory_object_mask is None:
                raise ValueError("Object-memory header needs both visual and object memory")
            packed = torch.cat([memory_hidden_states, memory_object_hidden_states.to(memory_hidden_states.dtype)], 1)
            packed_mask = torch.cat([memory_mask, memory_object_mask], 1)
        elif self._active_memory is not None:
            packed, packed_mask = self._active_memory
        else:
            raise ValueError("Object-memory header cannot run without memory")
        if packed.shape[1:] != (VISUAL_BUDGET + self.object_budget, self.hidden_size) or packed_mask.dtype != torch.bool:
            raise ValueError("Packed visual/object memory has an invalid contract")
        if self.config.n_conditions > 0 and joint_attention_kwargs["traj2ds"].shape[1] != self.config.n_conditions:
            raise ValueError("model n_conditions does not match")
        temb = self.time_ins_embed(timestep, pooled_projections) if self.combined_temb else self.time_ins_embed(timestep)
        if self.decision_head is not None:
            logits = self.decision_head(packed, packed_mask)
            self.last_decision_logits = logits
            probs = logits.float().softmax(-1)
            labels = self._decision_teacher_labels
            if self.training and labels is not None and self.decision_teacher_forcing > 0:
                labels = labels.to(probs.device)
                if labels.shape != probs.shape[:1]:
                    raise ValueError("Decision teacher labels do not match the batch")
                forced = (labels >= 0) & (torch.rand(labels.shape, device=probs.device) < self.decision_teacher_forcing)
                onehot = F.one_hot(labels.clamp(min=0), probs.shape[-1]).to(probs.dtype)
                probs = torch.where(forced[:, None], onehot, probs)
                self.last_decision_forced_fraction = float(forced.float().mean())
            decision = self.decision_embed(probs.to(self.decision_embed.weight.dtype))
            temb = temb + (decision[:, None, :] if temb.ndim == 3 else decision).to(temb.dtype)
        noisy_action = joint_attention_kwargs.pop("action_hidden_embeds")
        if packed.shape[0] != noisy_action.shape[0] or packed.device != noisy_action.device:
            raise ValueError("Object memory/action batch or device mismatch")
        action_hidden_states = self.action_proj_in(noisy_action)
        obs_hidden_states, obs_token_mask = self.obs_proj(
            views=joint_attention_kwargs["views"], obs=joint_attention_kwargs["obs"],
            traj2ds=joint_attention_kwargs["traj2ds"], text_embeddings=pooled_projections,
            vlm_attn_mask=vlm_attn_mask)
        for index, block in enumerate(self.transformer_blocks):
            if skip_layers is not None and index in skip_layers:
                continue
            action_hidden_states, obs_hidden_states = block(
                action_hidden_states=action_hidden_states, obs_hidden_states=obs_hidden_states, temb=temb,
                memory_hidden_states=packed, memory_mask=packed_mask, obs_token_mask=obs_token_mask)
        return ActionTransformerModelOutput(action=self.action_proj_out(x=action_hidden_states, t=temb))


class Psi0KeyframeNativeObjectModel(Psi0KeyframeNativeModel):
    ENTRY_SCHEMA_VERSION = "psi0-keyframe-native-object-model/v1"

    def __init__(self, model_cfg, vlm_model):
        if not getattr(model_cfg, "object_memory_enabled", False):
            raise ValueError("Object-memory model requires object_memory_enabled")
        super().__init__(model_cfg, vlm_model)
        self.action_header = NativeKeyframeObjectActionTransformerModel(self.action_header, model_cfg)
        self._object_gate_grad_seen = False

        def record_gate(gradient):
            if not self._object_gate_grad_seen and torch.isfinite(gradient).all() and gradient.abs().sum() > 0:
                self._object_gate_grad_seen = True
            return gradient

        # The first trainable object parameter depends on the gate mode.
        for block in self.action_header.transformer_blocks:
            attention = block.memory_attention
            target = (attention.object_gate if attention.gate_init == "zero"
                      else attention.object_attention.out_proj.weight)
            target.register_hook(record_gate)

    def decision_logits(self):
        """Logits of the last forward (None when the decision head is off)."""
        return self.action_header.last_decision_logits

    def object_gates(self) -> list[float]:
        return [float(block.memory_attention.gate_value().detach()) for block in self.action_header.transformer_blocks]

    def object_out_proj_norms(self) -> list[float]:
        return [float(block.memory_attention.object_attention.out_proj.weight.detach().float().norm())
                for block in self.action_header.transformer_blocks]

    def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw, action_samples: torch.Tensor,
                states: torch.Tensor, timestep: torch.LongTensor, traj2ds: Optional[torch.Tensor] = None,
                return_dict: bool = True, skip_layers: Optional[List[int]] = None, *, memory_features=None,
                memory_mask=None, memory_raw_frame_indices=None, memory_spatial_indices=None,
                **object_inputs: Any) -> HumanFoundationModelOutput:
        if not self._framesamp_forward_seen:
            self._emit_marker("first_training_forward")
            self._framesamp_forward_seen = True
        missing = [name for name in OBJECT_INPUTS if object_inputs.get(name) is None]
        if missing or set(object_inputs) != set(OBJECT_INPUTS):
            raise ValueError(f"Object-memory model is missing/has unexpected inputs: {missing or set(object_inputs)}")
        memory_inputs = self._memory_inputs(
            memory_features=memory_features, memory_mask=memory_mask,
            memory_raw_frame_indices=memory_raw_frame_indices, memory_spatial_indices=memory_spatial_indices)
        vlm_hidden_states = self.vlm_model(
            input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values,
            image_grid_thw=image_grid_thw, output_hidden_states=True, return_dict=True).hidden_states[-1]
        output = self.action_header(
            hidden_states=None, timestep=timestep,
            joint_attention_kwargs={"action_hidden_embeds": action_samples, "views": vlm_hidden_states.unsqueeze(1),
                                    "obs": states, "traj2ds": traj2ds},
            vlm_attn_mask=attention_mask, return_dict=return_dict, skip_layers=skip_layers,
            **memory_inputs, **object_inputs)
        return HumanFoundationModelOutput(action=output.action)

# Derived from Psi0 (https://github.com/physical-superintelligence-lab/Psi0, Apache License 2.0) and modified; see NOTICE.
import os
import copy
from tqdm import tqdm
from typing import List, Optional, Tuple, Union, Dict, Any
from typing import Any, Dict, List, Optional, Tuple, Union
from typing_extensions import Self
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from dataclasses import dataclass
from diffusers.utils.outputs import BaseOutput
from diffusers.configuration_utils import ConfigMixin, FrozenDict, register_to_config
# from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.modeling_utils import ModelMixin
# from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
from diffusers.models.embeddings import CombinedTimestepTextProjEmbeddings
# from diffusers.models.normalization import AdaLayerNorm, AdaLayerNormZero, AdaLayerNormContinuous
from diffusers.models.embeddings import CombinedTimestepLabelEmbeddings
from diffusers.models.normalization import FP32LayerNorm
from diffusers.models.attention_processor import Attention, JointAttnProcessor2_0
from diffusers.models.attention import FeedForward, _chunked_feed_forward
# from dit_policy.data4robotics.models.diffusion import _TimeNetwork
from psi.config.config import LaunchConfig
from transformers import Qwen3VLForConditionalGeneration, AutoConfig, AutoProcessor, Qwen2TokenizerFast, Qwen3VLProcessor
from qwen_vl_utils import process_vision_info

from psi.models.flow_matching import (
    FlowInversionResult,
    FlowSchedule,
    decode_flow,
    flow_euler_step,
    invert_flow_fixed_point,
)
from psi.utils import initialize_overwatch, count_parameters

# from InternVLA.model.modules.action_model.DiT_modules.models import DiT
# from InternVLA.model.modules.projector.QFormer import CrossAttentionBlock
overwatch = initialize_overwatch(__name__)

# from psi.config.model import ModelConfig, InternVLA_M1_ModelConfig
# from psi.learn.vlms import _QWen_VL_Interface
from transformers.models.siglip import SiglipModel
from transformers.models.dinov2 import Dinov2Model
# from vlt.transformers.vla import ActionTransformerModel

from diffusers.utils import logging
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name  FIXME why this not works any more?

QWEN3VL_VARIANT = "Qwen/Qwen3-VL-2B-Instruct"

@dataclass
class HumanFoundationModelOutput(BaseOutput):
    action: "torch.Tensor"  # noqa: F821

    def to_tuple(self):
        return (None, self.action)

@dataclass
class ActionTransformerModelOutput(BaseOutput):
    action: "torch.Tensor"  # noqa: F821

    def to_tuple(self):
        return (None, self.action)

@dataclass(frozen=True)
class Psi0Condition:
    """Frozen observation features reused by flow decoding and inversion."""

    views: torch.Tensor
    states: torch.Tensor
    traj2ds: Optional[torch.Tensor]
    attention_mask: Optional[torch.Tensor] = None
    vision_segment_ids: Optional[torch.Tensor] = None

    @property
    def batch_size(self) -> int:
        return int(self.states.shape[0])


def vision_segment_ids_from_input_ids(
    input_ids: torch.Tensor,
    image_token_id: int,
    expected_num_views: int | List[int] | Tuple[int, ...] | torch.Tensor,
) -> torch.Tensor:
    """Assign each contiguous image-token run to its image/view.

    Non-image tokens, including ordinary padding tokens, receive ``-1``.  Image
    runs are numbered independently for each batch row starting at zero.  The
    expected number of views is checked per row so prompt/template drift cannot
    silently corrupt downstream view-aware pooling.
    """

    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        shape = getattr(input_ids, "shape", None)
        raise ValueError(f"input_ids must have shape [B,N], got {shape}")
    if input_ids.dtype == torch.bool or input_ids.is_floating_point():
        raise TypeError("input_ids must use an integer dtype")
    if isinstance(image_token_id, bool) or not isinstance(
        image_token_id, (int, np.integer)
    ):
        raise TypeError("image_token_id must be an integer")
    image_token_id = int(image_token_id)
    if image_token_id < 0:
        raise ValueError("image_token_id must be non-negative")

    batch_size = int(input_ids.shape[0])
    if isinstance(expected_num_views, bool):
        raise TypeError("expected_num_views must contain integers")
    if isinstance(expected_num_views, (int, np.integer)):
        expected = torch.full(
            (batch_size,),
            int(expected_num_views),
            dtype=torch.long,
            device=input_ids.device,
        )
    else:
        expected = torch.as_tensor(
            expected_num_views, dtype=torch.long, device=input_ids.device
        )
        if expected.ndim != 1 or int(expected.numel()) != batch_size:
            raise ValueError(
                "expected_num_views must be a scalar or have one value per "
                f"batch row ({batch_size}), got shape {tuple(expected.shape)}"
            )
        # ``torch.as_tensor(..., dtype=long)`` would otherwise silently accept
        # fractional floating-point counts.
        raw_expected = torch.as_tensor(
            expected_num_views, device=input_ids.device
        )
        if raw_expected.dtype == torch.bool or (
            raw_expected.is_floating_point()
            and not torch.equal(raw_expected, raw_expected.round())
        ):
            raise TypeError("expected_num_views must contain integers")
    if torch.any(expected < 0):
        raise ValueError("expected_num_views must be non-negative")

    image_mask = input_ids == image_token_id
    previous_is_image = torch.zeros_like(image_mask)
    if input_ids.shape[1] > 1:
        previous_is_image[:, 1:] = image_mask[:, :-1]
    run_starts = image_mask & ~previous_is_image
    observed = run_starts.sum(dim=1, dtype=torch.long)
    mismatched = torch.nonzero(observed != expected, as_tuple=False).flatten()
    if mismatched.numel() > 0:
        rows = mismatched.detach().cpu().tolist()
        details = [
            {
                "row": int(row),
                "expected": int(expected[row].item()),
                "observed": int(observed[row].item()),
            }
            for row in rows
        ]
        raise ValueError(f"Image-token view count mismatch: {details}")

    run_ids = run_starts.cumsum(dim=1, dtype=torch.long) - 1
    return torch.where(
        image_mask,
        run_ids,
        torch.full_like(run_ids, -1),
    )


def _resolve_image_token_id(vlm_model: Any, vlm_processor: Any) -> int:
    """Resolve and cross-check the Qwen image placeholder token id."""

    candidates: list[tuple[str, int]] = []

    def add_candidate(label: str, value: Any) -> None:
        if value is None:
            return
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{label} must be an integer, got {value!r}")
        value = int(value)
        if value < 0:
            raise ValueError(f"{label} must be non-negative, got {value}")
        candidates.append((label, value))

    config = getattr(vlm_model, "config", None)
    add_candidate(
        "vlm_model.config.image_token_id",
        getattr(config, "image_token_id", None),
    )
    text_config = getattr(config, "text_config", None)
    add_candidate(
        "vlm_model.config.text_config.image_token_id",
        getattr(text_config, "image_token_id", None),
    )

    tokenizer = getattr(vlm_processor, "tokenizer", None)
    add_candidate(
        "vlm_processor.image_token_id",
        getattr(vlm_processor, "image_token_id", None),
    )
    add_candidate(
        "vlm_processor.tokenizer.image_token_id",
        getattr(tokenizer, "image_token_id", None),
    )

    image_token = getattr(vlm_processor, "image_token", None)
    if image_token is None:
        image_token = getattr(tokenizer, "image_token", None)
    if image_token is None:
        image_token = "<|image_pad|>"
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert):
        converted = convert(image_token)
        unk_token_id = getattr(tokenizer, "unk_token_id", None)
        if converted is not None and converted != unk_token_id:
            add_candidate(
                f"vlm_processor.tokenizer[{image_token!r}]", converted
            )

    if not candidates:
        raise ValueError(
            "Cannot resolve image_token_id from the VLM config or processor "
            "tokenizer"
        )
    distinct = {value for _, value in candidates}
    if len(distinct) != 1:
        raise ValueError(f"Conflicting image_token_id candidates: {candidates}")
    return candidates[0][1]


class _TimeNetwork(nn.Module):
    def __init__(self, time_dim, out_dim, learnable_w=False):
        assert time_dim % 2 == 0, "time_dim must be even!"
        half_dim = int(time_dim // 2)
        super().__init__()

        w = np.log(10000) / (half_dim - 1)
        w = torch.exp(torch.arange(half_dim) * -w).float()
        # self.register_parameter("w", nn.Parameter(w, requires_grad=learnable_w))
        self.w = nn.Parameter(w, requires_grad=learnable_w)

        self.out_net = nn.Sequential(
            nn.Linear(time_dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        # assert len(x.shape) == 1, "assumes 1d input timestep array"
        # RTC: support 2D timesteps (B,Tp)
        x = x[..., None] * self.w 
        x = torch.cat((torch.cos(x), torch.sin(x)), dim=-1) 
        return self.out_net(x) 

class RMSNorm(nn.Module):
    def __init__(self, dim, eps: float, elementwise_affine: bool = True):
        super().__init__()

        self.eps = eps

        if isinstance(dim, int):
            dim = (dim,)

        self.dim = torch.Size(dim)

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.weight = None

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        if self.weight is not None:
            # convert into half-precision if necessary
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                hidden_states = hidden_states.to(self.weight.dtype)
            hidden_states = hidden_states * self.weight
        else:
            hidden_states = hidden_states.to(input_dtype)

        return hidden_states


if torch.__version__ >= "2.1.0":
    LayerNorm = nn.LayerNorm # type:ignore
else:
    # Has optional bias parameter compared to torch layer norm
    # TODO: replace with torch layernorm once min required torch version >= 2.1
    class LayerNorm(nn.Module):
        def __init__(self, dim, eps: float = 1e-5, elementwise_affine: bool = True, bias: bool = True):
            super().__init__()

            self.eps = eps

            if isinstance(dim, int):
                dim = (dim,)

            self.dim = torch.Size(dim)

            if elementwise_affine:
                self.weight = nn.Parameter(torch.ones(dim))
                self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
            else:
                self.weight = None
                self.bias = None

        def forward(self, input):
            return F.layer_norm(input, self.dim, self.weight, self.bias, self.eps)

class AdaLayerNormContinuous(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        # NOTE: It is a bit weird that the norm layer can be configured to have scale and shift parameters
        # because the output is immediately scaled and shifted by the projected conditioning embeddings.
        # Note that AdaLayerNorm does not let the norm layer have scale and shift parameters.
        # However, this is how it was implemented in the original code, and it's rather likely you should
        # set `elementwise_affine` to False.
        elementwise_affine=True,
        eps=1e-5,
        bias=True,
        norm_type="layer_norm",
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=bias)
        if norm_type == "layer_norm":
            self.norm = LayerNorm(embedding_dim, eps, elementwise_affine, bias)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(embedding_dim, eps, elementwise_affine)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        # convert back to the original dtype in case `conditioning_embedding`` is upcasted to float32 (needed for hunyuanDiT)
        if len(conditioning_embedding.shape) == 2: # (B, D)
            conditioning_embedding = conditioning_embedding.unsqueeze(1) # (B, 1, D)
        elif len(conditioning_embedding.shape) == 3: # (B, T, D)
            pass
        else:
            raise ValueError(f"Invalid shape of conditioning_embedding: {conditioning_embedding.shape}")
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        return x

class AdaLayerNormZero(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, num_embeddings: Optional[int] = None, norm_type="layer_norm", bias=True):
        super().__init__()
        if num_embeddings is not None:
            self.emb = CombinedTimestepLabelEmbeddings(num_embeddings, embedding_dim)
        else:
            self.emb = None

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=bias)
        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        elif norm_type == "fp32_layer_norm":
            self.norm = FP32LayerNorm(embedding_dim, elementwise_affine=False, bias=False)
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm', 'fp32_layer_norm'."
            )

    def forward(
        self,
        x: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        hidden_dtype: Optional[torch.dtype] = None,
        emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.emb is not None:
            emb = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)
            assert False, "Not implemented with emb shape == 3, (B, T, D)"
        
        if len(emb.shape) == 2: # (B, D)
            emb = emb.unsqueeze(1) # (B, 1, D)
        elif len(emb.shape) == 3: # (B, T, D)
            pass
        else:
            raise ValueError(f"Invalid shape of emb: {emb.shape}")
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=-1)
        x = self.norm(x) * (1 + scale_msa) + shift_msa
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp

class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # self.register_buffer("pe", torch.clone(pe)) # <-- SONGLIN: this will cause problems when accelerate.prepare
        # self.register_parameter("pe", nn.Parameter(pe, requires_grad=False))
        self.pe = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)

        Returns:
            Tensor of shape (seq_len, batch_size, d_model) with positional encodings added
        """
        pe = self.pe[: x.shape[0]]
        pe = pe.repeat((1, x.shape[1], 1))
        return pe.detach().clone()

class FilmConditioning(nn.Module):
    """Layer that adds FiLM conditioning.

    This is intended to be applied after a convolutional layer. It will learn a
    multiplicative and an additive factor to be applied to each channel of the
    convolution's output.

    Conv layer can be rank 2 or 4.

    For further details, see: https://arxiv.org/abs/1709.07871
    """
    
    def __init__(self,
                 in_dim: int,
                 num_channels: int):
        """Constructs a FiLM conditioning layer.

        Args:
            num_channels: Number of filter channels to expect in the input.
        """
        super(FilmConditioning, self).__init__() 
        # Note that we initialize with zeros because empirically we have found
        # this works better than initializing with glorot.
        self._projection_add = nn.Linear(in_dim, num_channels)
        self._projection_mult = nn.Linear(in_dim, num_channels)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, conv_filters, conditioning):
        assert len(conditioning.shape) == 2
        projected_cond_add = self._projection_add(conditioning)
        projected_cond_mult = self._projection_mult(conditioning)

        if len(conv_filters.shape) == 4:
            # [B, D] -> [B, D, 1, 1]
            projected_cond_add = projected_cond_add.unsqueeze(-1).unsqueeze(-1)
            projected_cond_mult = projected_cond_mult.unsqueeze(-1).unsqueeze(-1)
        elif len(conv_filters.shape) == 3:
            # [B, D] -> [B, 1, D]
            projected_cond_add = projected_cond_add.unsqueeze(1)
            projected_cond_mult = projected_cond_mult.unsqueeze(1)
        else:
            assert len(conv_filters.shape) == 2
        
        # Original FiLM paper argues that 1 + gamma centers the initialization at
        # identity transform.
        result = (1 + projected_cond_mult) * conv_filters + projected_cond_add
        return result

class JointVLAAttnProcessor: #(nn.Module):
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        kv_heads: Optional[int] = None,
        dim_head: int = 64,
        out_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
        out_bias: bool = True,
        context_pre_only=None,
    ):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("JointAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        # self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        # self.inner_kv_dim = self.inner_dim if kv_heads is None else dim_head * kv_heads
        # self.out_dim = out_dim if out_dim is not None else query_dim

        # self.to_q_lang = nn.Linear(query_dim, self.inner_dim, bias=bias)
        # self.to_k_lang = nn.Linear(query_dim, self.inner_kv_dim, bias=bias)
        # self.to_v_lang = nn.Linear(query_dim, self.inner_kv_dim, bias=bias)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states

        batch_size = hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)

        assert attn.to_k is not None
        key = attn.to_k(hidden_states)

        assert attn.to_v is not None
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # `context` projections.
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)
            if attention_mask is not None:
                assert attention_mask.dtype == torch.float32
                attn_mask = torch.cat([
                    torch.ones(hidden_states.shape[0], 1, 1, hidden_states.shape[1], device=attention_mask.device).to(torch.bool),  # (B, 1, 1, S1)
                    (attention_mask == 1)[:, None, None, :]  # (B, 1, 1, S2)
                ], dim=-1)
            else:
                attn_mask = None

        observer = getattr(attn, "_attention_observer", None)
        if observer is not None:
            observer(query.detach(), key.detach(), attn_mask, action_length=residual.shape[1], value=value.detach())
        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False, attn_mask=attn_mask) # type: ignore
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim) # type: ignore
        hidden_states = hidden_states.to(query.dtype) # type: ignore
 
        # Split the attention outputs.
        hidden_states, encoder_hidden_states = (
            hidden_states[:, : residual.shape[1]],
            hidden_states[:, residual.shape[1] :],
        ) # type: ignore
        if not attn.context_pre_only:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states, encoder_hidden_states
    
        
class ObservationProjection(nn.Module): # FIXME naming
    """ VLT observation projector

    Args:
        ...
    """

    def __init__(
        self,
        action_pred_horizon: int = 50, # Tp
        action_dim: int = 7, # in dim
        output_dim: int = 512, # out dim
        hidden_dim: int = 1280,
        n_conditions: int = 1, # num image of contitions
        timestep_in_dim: int = 320,
        timestep_flip_sin_to_cos: bool = True,
        timestep_freq_shift: int = 0,
        token_fusion: str = "concat",
        resnet_store_path: str = "cache/visual_features/resnet18/IN_1M_resnet18.pth",
        odim: int = 32,
        view_feature_dim: int = 1920,
        use_film: bool = False
    ):
        super().__init__()

        # TODO 1. compare
        
        # self.time_net = _TimeNetwork(time_dim=256, out_dim=512)
        
        # self.time_proj = Timesteps(timestep_in_dim, timestep_flip_sin_to_cos, timestep_freq_shift)
        # self.time_embedding = TimestepEmbedding(timestep_in_dim, output_dim, act_fn="silu")
        self.action_pred_horizon = action_pred_horizon
        self.action_dim = action_dim
        # self.ac_proj = nn.Sequential(
        #     nn.Linear(action_dim, action_dim),
        #     nn.GELU(approximate="tanh"),
        #     nn.Linear(action_dim, output_dim),
        # )

        # _DiTNoiseNet.__init__
        self.enc_pos = _PositionalEncoding(d_model=output_dim)

        

        # agent.__init__
        n_cams = 0 # 1
        
        imgs_per_cam = 1
        self.camera_indices = [] # [0]
        self._n_cams = n_cams
        self._share_cam_features = False # model_cfg.share_cam_features
        early_fusion = False
        text_dim = 768
        # odim = 15
        self.token_fusion = token_fusion

        if self._share_cam_features:
            ...
        else:
            if n_conditions > 0:
                from dit_policy.data4robotics.models.resnet import ResNet
                features = ResNet(
                    size=18,
                    weights="IMAGENET1K_V1",
                    restore_path=resnet_store_path,
                    avg_pool=False,
                    conv_repeat=0,
                    norm_cfg={
                        "name": "group_norm",
                        "num_groups": 16,
                    }
                )
            else:
                features = None
            feat_list = [copy.deepcopy(features) for _ in range(0, n_conditions)] # + [copy.deepcopy(features) for _ in range(1, n_cams)] + [copy.deepcopy(features) for _ in range(0, n_conditions)]
            self.visual_features = nn.ModuleList(feat_list)
            self.views_proj = nn.Linear(view_feature_dim, output_dim, bias=True) # TODO configure 1920 # boqian_fix

            # TODO abl: more layers # view proj
            # self.fc1 = nn.Linear(self.vision_dim, initial_projection_dim, bias=True)
            # self.fc2 = nn.Linear(initial_projection_dim, self.llm_dim, bias=True)
            # self.fc3 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            # self.act_fn1 = nn.GELU()
            # self.act_fn2 = nn.GELU()

            # self.views_proj = nn.Sequential(
            #     nn.Conv2d(16, output_dim, kernel_size=(2, 2), stride=2, bias=True),
            #     nn.ReLU(),
            #     nn.Conv2d(output_dim, output_dim, kernel_size=(2, 2), stride=2, bias=True)
            # )
            # self.traj2d_proj = nn.ModuleList([
            #     nn.Conv2d(16, 512, kernel_size=(2, 2), stride=2, bias=True),
            #     nn.ReLU(),
            #     nn.Conv2d(512, output_dim, kernel_size=(2, 2), stride=2, bias=True),
            #     nn.ReLU(),
            #     nn.Linear(output_dim, output_dim, bias=True),
            #     nn.LayerNorm(output_dim)
            # ])
        
        self.early_fusion = early_fusion
        self.imgs_per_cam = imgs_per_cam
        imgs_per_cam = 1 if early_fusion else imgs_per_cam
        self._n_conditions = n_conditions

        self.embed_proj = nn.Identity()
        if self.token_fusion != "perceiver" and self._n_conditions > 0:
            assert features is not None
            self._token_dim = features.embed_dim
            if output_dim != self._token_dim:
                self.embed_proj = nn.Linear(self._token_dim, output_dim, bias=False)
        else:
            self._token_dim = 0

        self._n_tokens = imgs_per_cam * n_cams * self._token_dim + n_conditions * self._token_dim
        self.traj_n_token = self._token_dim
        
        self.feat_norm = nn.LayerNorm(output_dim, elementwise_affine=False, eps=1e-6) # feat_norm = None
        self.feat_traj_norm = nn.LayerNorm(output_dim, elementwise_affine=False, eps=1e-6) # feat_norm = None
        # self.traj2d_proj = nn.ModuleList([
        #     nn.Linear(output_dim, output_dim) for _ in range(2)
        # ])
        # self.traj2d_proj = nn.Conv2d(
        #     16, output_dim, kernel_size=(2, 2), stride=2, bias=True
        # )

        if self.token_fusion == "cross":
            
            self.attn = Attention(
                query_dim=self._token_dim,
                cross_attention_dim=1920,
                dim_head=64,
                out_dim=output_dim,
                heads=output_dim//64,
                bias=True,
                # pre_only=True,
                eps=1e-6
            )
        elif self.token_fusion == "perceiver":
            num_queries = 256
            self.latents1 = nn.Parameter(torch.randn(1, num_queries, hidden_dim) / hidden_dim**0.5)
            # self.proj_in1 = nn.Linear(1920, hidden_dim)  -> views_proj (see above)
            self.cross_attn1 = Attention(
                query_dim=hidden_dim,
                cross_attention_dim=hidden_dim,
                dim_head=64,
                out_dim=output_dim,
                heads=output_dim//64,
                bias=True,
                # pre_only=True,
                eps=1e-6
            )
            self.latents2 = nn.Parameter(torch.randn(1, num_queries, hidden_dim) / hidden_dim**0.5)
            self.proj_in2 = nn.Linear(self._token_dim, hidden_dim)
            self.cross_attn2 = Attention(
                query_dim=hidden_dim,
                cross_attention_dim=hidden_dim,
                dim_head=64,
                out_dim=output_dim,
                heads=output_dim//64,
                bias=True,
                # pre_only=True,
                eps=1e-6
            )

        self.film = FilmConditioning(2048, output_dim) if use_film else None

        self._obs_strat = "add_token"
        self._n_tokens += 1

        self._obs_proc = nn.Sequential(
            nn.Dropout(p=0.2), nn.Linear(odim, output_dim)
        )
        dropout = 0.1 # @see model_cfg.dropout
        linear_proj = nn.Identity() # build (optional) token feature projection layer 
        norm = nn.Identity() # feat_norm = None
        self.post_proc = nn.Sequential(linear_proj, norm, nn.Dropout(dropout))

    def forward(self, 
                # noisy_actions: torch.Tensor, 
                # timestep: torch.Tensor, 
                # temp
                views, 
                obs, 
                traj2ds=None, 
                # ac_flat = None, 
                # mask_flat = None, 
                text_embeddings = None, 
                vlm_attn_mask=None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            noisy_actions (`torch.Tensor`):
                Image features.
            timestep (`torch.Tensor`):
                Timestep in denoising process.
        Returns:
            `Tuple`[`torch.Tensor`, `torch.Tensor`]: The pair (latents, timestep_emb).
        """
        # DiffusionTransformerAgent.forward
        s_t, vlm_attn_mask = self.tokenize_obs(views, obs, traj2ds, text_embeddings=text_embeddings, vlm_attn_mask=vlm_attn_mask) # (B, S, d_model)
        # _DiTNoiseNet.forward_enc
        s_t = s_t.transpose(0, 1) # (S, B, d_model)
        pos = self.enc_pos(s_t) # (S, B, d_model)
        return (s_t + pos).transpose(0,1), vlm_attn_mask # (B,S,d_model)

    def tokenize_obs(self, views, obs, traj2ds = None, flatten=False, text_embeddings=None, vlm_attn_mask=None):
        
        view_tokens = self.views_proj(views)
        B,V,S,D = view_tokens.shape
        
        if traj2ds is not None:
            traj_tokens = self.embed({}, {f"cond{i}": traj2ds[:, i] for i in range(traj2ds.shape[1])})

            if self.token_fusion == "concat":
                traj_tokens = self.embed_proj(traj_tokens) #(B,S,d_model)
                tokens = torch.cat([view_tokens.view(B,V*S,D), traj_tokens], dim=1) # (B, tok1+tok2, 1536)
            elif self.token_fusion == "cross":  
                traj_tokens_list, view_tokens_list = [], []
                for v_idx in range(traj2ds.shape[1]):
                    # traj2d --> cross attend --> view features
                    view_tokens = self.attn(
                        hidden_states=traj_tokens[:, v_idx*self.traj_n_token, (v_idx+1)*self.traj_n_token],
                        encoder_hidden_states=views[:, v_idx],
                    )
                    traj_tokens_list.append(traj_tokens)
                    view_tokens_list.append(view_tokens)

                traj_tokens = torch.cat(traj_tokens_list, dim=1)
                traj_tokens = self.embed_proj(traj_tokens)
                view_tokens = torch.cat(view_tokens_list, dim=1)
                tokens = torch.cat([view_tokens, traj_tokens], dim=1) # (B, tok1+tok2, 1536)
            elif self.token_fusion == "perceiver":
                latents1 = self.latents1.repeat(B, 1, 1)
                # views = self.proj_in1(views)
                # views = views + text_embeddings[:, None]
                tokens1 = self.cross_attn1(
                    hidden_states=latents1,
                    encoder_hidden_states=view_tokens.view(B, V*S, D),
                )
                latents2 = self.latents2.repeat(B, 1, 1)
                traj_tokens = self.proj_in2(traj_tokens)
                # traj2ds = traj2ds + text_embeddings[:, None]
                tokens2 = self.cross_attn2(
                    hidden_states=latents2,
                    encoder_hidden_states=traj_tokens,
                )
                tokens = torch.cat([tokens1, tokens2], dim=1) # (B, tok1+tok2, 1536)
            else:
                raise ValueError
        else:
            assert self._n_conditions == 0, "inconsistent confg"
            tokens = view_tokens.view(B,V*S,D)

        if (self.film is not None) and (text_embeddings is not None):
            tokens = self.film(tokens, text_embeddings)

        if self._obs_strat == "add_token":
            obs_token = self._obs_proc(obs)#[:, None]
            tokens = torch.cat((tokens, obs_token), 1)
            vlm_attn_mask = torch.cat([vlm_attn_mask, torch.ones((vlm_attn_mask.shape[0], 1), device=vlm_attn_mask.device)], 1) if vlm_attn_mask is not None else None
        elif self._obs_strat == "pad_img_tokens":
            obs = self._obs_proc(obs)
            obs = obs[:, None].repeat((1, tokens.shape[1], 1))
            tokens = torch.cat((obs, tokens), 2)
            vlm_attn_mask = None
        else:
            assert self._obs_strat is None

        tokens = self.post_proc(tokens)
        if flatten:
            return tokens.reshape((tokens.shape[0], -1)), vlm_attn_mask
        return tokens, vlm_attn_mask
    
    def embed(self, imgs, conditions=None):
        """ SONGLIN: DOWNSCALE == 2**5 = 32, total 5 conv2d with stride(2) """
        def embed_helper(net, im):
            # with torch.autocast(device_type='cuda', dtype=torch.bfloat16): # TODO check if this is needed
            if self.early_fusion and len(im.shape) == 5:
                T = im.shape[1]
                im = torch.cat([im[:, t] for t in range(T)], 1)
                embeds = net(im)
            elif len(im.shape) == 5:
                B, T, C, H, W = im.shape
                embeds = net(im.reshape((B * T, C, H, W)))
                embeds = embeds.reshape((B, -1, net.embed_dim))
            else:
                assert len(im.shape) == 4
                embeds = net(im)
            return embeds

        if self._share_cam_features:
            embeds = [
                embed_helper(self.visual_features, imgs[f"cam{i}"])
                for i in self.camera_indices
            ]
            if conditions is not None:
                assert self._n_conditions > 0
                embeds += [
                    embed_helper(self.visual_features, conditions[f"cond{i}"]) 
                    for i in range(self._n_conditions)
                ]
        else:
            assert conditions is not None
            # img_encoders = self.visual_features[:self._n_cams] # sum([v.sum().item() for k, v in img_encoders.named_parameters()])
            # cond_encoders = self.visual_features[self._n_cams:]
            cond_encoders = self.visual_features
            # [
            #     embed_helper(net, imgs[f"cam{i}"])
            #     for i, net in zip(self.camera_indices, img_encoders)
            # ] +
            embeds = [
                embed_helper(net, conditions[f"cond{0}"])
                for i, net in enumerate(cond_encoders)
            ]

        """ if self.training: # FIXME better handle this thing
            return torch.cat(embeds, dim=1) """
        assert conditions is not None
        in_dtype = next(iter(imgs.values())).dtype if len(imgs) > 0 else next(iter(conditions.values())).dtype
        embeds = torch.cat(embeds, dim=1).to(dtype=in_dtype) # (B, S, d_model)
        return embeds

class ActionProjectionIn(nn.Module):
    def __init__(
        self,
        action_pred_horizon,
        action_dim,
        output_dim,
    ):  
        super().__init__()
        self.action_pred_horizon = action_pred_horizon
        self.action_dim = action_dim

        self.ac_proj = nn.Sequential(
            nn.Linear(action_dim, action_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(action_dim, output_dim),
        )
        # self.time_net = _TimeNetwork(time_dim=256, out_dim=output_dim)

        # self.register_parameter(
        #     "dec_pos",
        #     nn.Parameter(torch.empty(action_pred_horizon, output_dim), requires_grad=True),
        # ) # (1, Tp, hidden_dim)
        self.dec_pos = nn.Parameter(torch.empty(action_pred_horizon, output_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.dec_pos.data)

    def forward(self, noisy_actions: torch.Tensor, 
        # timestep
        ) -> torch.Tensor:
        B = noisy_actions.shape[0]
        # action_dim = self._dit_obs_proj.action_dim
        # time_enc = self.time_net(timestep)
        noise_acs = noisy_actions.reshape(B, -1, self.action_dim) # DiffusionTransformerAgent.forward
        ac_tokens = self.ac_proj(noise_acs) # noisy_actions, _DiTNoiseNet.forward
        # ac_tokens = ac_tokens.transpose(0, 1) # (B, L, d_model) _DiTNoiseNet.forward_dec
        action_hidden_states = ac_tokens + self.dec_pos.unsqueeze(0)
        return action_hidden_states#, time_enc
    

class ActionProjectionOut(nn.Module): # _FinalLayer
    def __init__(self, hidden_size, action_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, action_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        # self.reset_parameters()

    def forward(self, x, t, cond=None):
        # process the conditioning vector first
        # cond = torch.mean(cond, dim=1)
        # cond = cond + t

        if len(t.shape) == 2: # (B, D)
            t = t.unsqueeze(1) # (B, 1, D)
        elif len(t.shape) == 3: # (B, T, D)
            pass
        else:
            raise ValueError(f"Invalid shape of t: {t.shape}")
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=-1) # cond -> t
        # x = self.norm_final(x) * (1+scale[None]) + shift[None] # TODO be careful!
        x = x * scale + shift
        x = self.linear(x)
        # return x.transpose(0, 1)
        return x
    

# @maybe_allow_in_graph
class VLATransformerBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        context_pre_only: bool = False,
        qk_norm: Optional[str] = None,
        use_dual_attention: bool = False,
        training_phase: str = "joint",  # "joint", "action", "traj2d", "action_mini"
    ):
        super().__init__()

        # self.training_phase = training_phase

        # from models.dit_policy.data4robotics.models.diffusion import  _SelfAttnEncoder #_TransformerEncoder,
        # self.encoder_module = _SelfAttnEncoder(
        #     d_model=dim,
        #     nhead=num_attention_heads,
        #     dim_feedforward=2048,
        #     dropout=0.1,
        #     activation="gelu",
        # )
        # self.encoder_module.reset_parameters() 

        # # decoder blocks
        # from models.dit_policy.data4robotics.models.diffusion import _DiTDecoder #_TransformerDecoder, 
        # self.decoder_module = _DiTDecoder(
        #     d_model=dim,
        #     nhead=num_attention_heads,
        #     dim_feedforward=2048,
        #     dropout=0.1,
        #     activation="gelu",
        # )
        # self.decoder_module.reset_parameters() # _TransformerDecoder.__init__

        self.context_pre_only = context_pre_only
        context_norm_type = "ada_norm_continous" if context_pre_only else "ada_norm_zero"

        # self.norm1 = AdaLayerNormZero(dim)
        self.norm1_act = AdaLayerNormZero(dim)
        logger.debug(f"AdaLN act parameters: {count_parameters(self.norm1_act):,}")
        if context_norm_type == "ada_norm_continous":
            """ self.norm1_lang = AdaLayerNormContinuous(dim, dim, elementwise_affine=False, eps=1e-6, bias=True, norm_type="layer_norm") """
            self.norm1_obs = AdaLayerNormContinuous(dim, dim, elementwise_affine=False, eps=1e-6, bias=True, norm_type="layer_norm")
            logger.debug(f"AdaLN obs parameters: {count_parameters(self.norm1_obs):,}")
        elif context_norm_type == "ada_norm_zero":
            """ self.norm1_lang = AdaLayerNormZero(dim) """
            self.norm1_obs = AdaLayerNormZero(dim)
            logger.debug(f"AdaLN obs parameters: {count_parameters(self.norm1_obs):,}")
        else:
            raise ValueError(
                f"Unknown context_norm_type: {context_norm_type}, currently only support `ada_norm_continous`, `ada_norm_zero`"
            )
        
        if hasattr(F, "scaled_dot_product_attention"):
            processor = JointVLAAttnProcessor(
                query_dim=dim,
                dim_head=attention_head_dim,
                heads=num_attention_heads,
                out_dim=dim,
                bias=True,
                context_pre_only=context_pre_only,
            )
        else:
            raise ValueError(
                "The current PyTorch version does not support the `scaled_dot_product_attention` function."
            )

        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=context_pre_only,
            bias=True,
            processor=processor, # type: ignore
            qk_norm=qk_norm,
            eps=1e-6,
        )
        logger.debug(f"Joint Attention parameters: {count_parameters(self.attn):,}")

        self.norm2_act = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_act = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
        logger.debug(f"FF act parameters: {count_parameters(self.ff_act):,}")

        if not context_pre_only:
            """ self.norm2_lang = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.ff_lang = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate") """

            self.norm2_obs = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.ff_obs = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
            logger.debug(f"FF obs parameters: {count_parameters(self.ff_obs):,}")
        else:
            """ self.norm2_lang = None
            self.ff_lang = None """

            self.norm2_obs= None
            self.ff_obs = None

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def forward(
        self, 
        # hidden_states: torch.FloatTensor,  #  V
        # lang_hidden_states: torch.FloatTensor,  # L
        action_hidden_states: torch.Tensor, # A
        obs_hidden_states: torch.Tensor, # O
        temb: torch.Tensor,
        obs_token_mask: Optional[torch.Tensor] = None,
        # obs_pos_emb: Optional[torch.FloatTensor] = None,
        # time_enc: Optional[torch.FloatTensor] = None,
        joint_attention_kwargs=None,
    ):
        # obs_hidden_states = self.encoder_module(src=obs_hidden_states, pos=obs_pos_emb)
        # action_hidden_states = self.decoder_module(x=action_hidden_states, t=time_enc, cond=obs_hidden_states) # recursive: _TransformerDecoder.forward
        
        norm_action_hidden_states, gate_msa_act, shift_mlp_act, scale_mlp_act, gate_mlp_act = self.norm1_act(action_hidden_states, emb=temb)

        if self.context_pre_only:
            """ norm_lang_hidden_states = self.norm1_lang(lang_hidden_states, temb) """
            norm_obs_hidden_states = self.norm1_obs(obs_hidden_states, temb[:,-1] if len(temb.shape) > 2 else temb)

            gate_msa_obs, shift_mlp_obs, scale_mlp_obs, gate_mlp_obs = None, None, None, None
        else:
            """ norm_lang_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_lang(
                lang_hidden_states, emb=temb
            ) """
            norm_obs_hidden_states, gate_msa_obs, shift_mlp_obs, scale_mlp_obs, gate_mlp_obs = self.norm1_obs(obs_hidden_states, emb=temb[:,-1] if len(temb.shape) > 2 else temb)

        act_attn_output, obs_attn_output = self.attn( ## indirectly calls JointAttnProcessor2_0.__call__
            # hidden_states=None,
            # lang_hidden_states=norm_lang_hidden_states,
            hidden_states=norm_action_hidden_states,
            encoder_hidden_states=norm_obs_hidden_states,
            attention_mask=obs_token_mask
            # **joint_attention_kwargs,
        )

        # Action feed forward
        act_attn_output = gate_msa_act * act_attn_output
        action_hidden_states = action_hidden_states + act_attn_output

        norm_action_hidden_states = self.norm2_act(action_hidden_states)
        norm_action_hidden_states = norm_action_hidden_states * (1 + scale_mlp_act) + shift_mlp_act
        ff_action_output = self.ff_act(norm_action_hidden_states)

        ff_action_output = gate_mlp_act * ff_action_output
        action_hidden_states = action_hidden_states + ff_action_output

        # attention outputs for the `obs_hidden_states`.
        if self.context_pre_only:
            # encoder_hidden_states = None
            obs_hidden_states = None # type:ignore
        else:
            assert gate_msa_obs is not None and \
                shift_mlp_obs is not None and \
                    scale_mlp_obs is not None and \
                        gate_mlp_obs is not None
            """ # lang forward
            lang_attn_output = c_gate_msa.unsqueeze(1) * lang_attn_output
            lang_hidden_states = lang_hidden_states + lang_attn_output
            norm_lang_hidden_states = self.norm2_lang(lang_hidden_states)
            norm_lang_hidden_states = norm_lang_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                context_ff_output = _chunked_feed_forward(
                    self.ff_lang, norm_lang_hidden_states, self._chunk_dim, self._chunk_size
                )
            else:
                context_ff_output = self.ff_lang(norm_lang_hidden_states)
            lang_hidden_states = lang_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output """
            
            # obs feedforward
            assert self.norm2_obs is not None

            obs_attn_output = gate_msa_obs * obs_attn_output
            obs_hidden_states = obs_hidden_states + obs_attn_output

            norm_obs_hidden_states = self.norm2_obs(obs_hidden_states)
            norm_obs_hidden_states = norm_obs_hidden_states * (1 + scale_mlp_obs) + shift_mlp_obs

            assert self.ff_obs is not None
            ff_obs_output = self.ff_obs(norm_obs_hidden_states)
            ff_obs_output = gate_mlp_obs * ff_obs_output
            obs_hidden_states = obs_hidden_states + ff_obs_output

        return action_hidden_states, obs_hidden_states
    
class ActionTransformerModel(
    ModelMixin, ConfigMixin #, VLTTransformerActionMixin
    # PeftAdapterMixin, FromOriginalModelMixin, ReprodMixin, SD3Transformer2DLoadersMixin, 
):  
    """  An action transformer model """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["ActionTransformerBlock"]
    # _skip_layerwise_casting_patterns = ["pos_embed", "norm"]

    @classmethod
    def from_pretrained(cls,  pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        # model = cls.from_config(config=None, **kwargs)
        model = ActionTransformerModel()
        torch_dtype = kwargs.pop("torch_dtype", torch.float32)
        if pretrained_model_name_or_path is not None and "checkpoints" in str(pretrained_model_name_or_path):
            if os.path.exists(pretrained_model_name_or_path):
                 # NOTE use accelerator.checkpointing 
                from safetensors.torch import load_file
                file_path = f"{pretrained_model_name_or_path}/model.safetensors"
                assert os.path.exists(file_path), f"Checkpoint {file_path} does not exist."
                model_state_dict = load_file(file_path)
                missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
                
                """ # load all checkpoint data, including weights, optimizer, etc.
                load_dict = torch.load(pretrained_model_name_or_path, weights_only=False) 
                from torch.nn.parallel import DistributedDataParallel as DDP
                model = model.module if isinstance(model, DDP) else model
                missing_keys, unexpected_keys = model.load_state_dict(load_dict["model"], strict=False)  """

                if len(unexpected_keys) > 0:
                    logger.info(f"Unexpected keys in the checkpoint: {unexpected_keys}. ")
                if len(missing_keys) > 0:
                    logger.info(f"Missing keys in the checkpoint: {missing_keys}.")

                logger.info(f"Loaded model from {pretrained_model_name_or_path}")
            else:
                raise ValueError(f"Checkpoint {pretrained_model_name_or_path} does not exist. ")
        
        model.to(torch_dtype)
        return model

    @register_to_config
    def __init__(
        self,
        # sample_size: int = 128,
        # patch_size: int = 2,
        # in_channels: int = 16,
        # num_layers: int = 24,
        attention_head_dim: int = 64,
        num_attention_heads: int = 24,
        # joint_attention_dim: int = 4096,
        caption_projection_dim: int = 1536,
        pooled_projection_dim: int = 2048,
        # out_channels: int = 16,
        pos_embed_max_size: int = 192,
        dual_attention_layers: Tuple[
            int, ...
        ] = (),  # () for sd3.0; (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12) for sd3.5
        qk_norm: Optional[str] = None,
        # VLA parameters
        training_phase: str = "joint",  # "joint", "action", "traj2d", "action_mini"
        # action head parameters
        action_num_blocks:int = 6,
        action_pred_horizon: int = 6,
        action_dim: int = 7,
        action_hidden_dim: int = 1536,
        action_nheads: int = 24,
        # action_attention_head_dim: Optional[int] = 64,
        n_conditions: int = 0,
        token_fusion: str = "concat",  # "concat", "cross"
        resnet_store_path: str = "cache/visual_features/resnet18/IN_1M_resnet18.pth",
        odim: int = 32,
        view_feature_dim: int = 1920,
        use_film: bool = False,
        combined_temb: bool = False
    ):
        super().__init__()
        # self.inner_dim = num_attention_heads * attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim 

        self.combined_temb = combined_temb
        if self.combined_temb:
            self.time_ins_embed = CombinedTimestepTextProjEmbeddings(
                embedding_dim=self.inner_dim, pooled_projection_dim=pooled_projection_dim
            ) # used
        else:
            self.time_ins_embed = _TimeNetwork(time_dim=256, out_dim=action_hidden_dim)
        """ self.lang_embedder = nn.Linear(joint_attention_dim, caption_projection_dim) """

        # Set the observation projection
        self.obs_proj = ObservationProjection(
            action_pred_horizon=action_pred_horizon, 
            action_dim=action_dim, 
            hidden_dim=self.inner_dim, # not used
            output_dim=action_hidden_dim,
            n_conditions=n_conditions,
            token_fusion=token_fusion,
            resnet_store_path=resnet_store_path,
            odim=odim,
            view_feature_dim=view_feature_dim,
            use_film=use_film
        )

        total_params = sum(p.numel() for p in self.obs_proj.parameters() if p.requires_grad)
        logger.debug(f"ObservationEncoder parameters: {total_params:,}")
        # Set the action projection
        self.action_proj_in = ActionProjectionIn(
            action_pred_horizon=action_pred_horizon, 
            action_dim=action_dim, 
            output_dim=action_hidden_dim,
        )
        total_params = sum(p.numel() for p in self.action_proj_in.parameters() if p.requires_grad)
        logger.debug(f"ActionProjectionIn parameters: {total_params:,}")

        self.transformer_blocks = nn.ModuleList(
            [
                # IdentityBlock()
                VLATransformerBlock( # JointTransformerBlock
                    dim=action_hidden_dim,
                    num_attention_heads=action_nheads,
                    attention_head_dim=attention_head_dim,
                    context_pre_only=i == action_num_blocks - 1,
                    qk_norm=qk_norm,
                    # use_dual_attention=True if i in dual_attention_layers else False,
                    training_phase=training_phase,  # "joint", "action", "traj2d", "action_mini"
                )
                for i in range(action_num_blocks)
            ]
        )
        total_params = sum(p.numel() for p in self.transformer_blocks[0].parameters() if p.requires_grad)
        logger.debug(f"ActionTransformerBlock parameters: {total_params:,}")

        self.action_proj_out = ActionProjectionOut(
            hidden_size=action_hidden_dim, 
            action_dim=action_dim
        )
        total_params = sum(p.numel() for p in self.action_proj_out.parameters())
        logger.debug(f"ActionProjectionOut parameters: {total_params:,}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        pooled_projections: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        block_controlnet_hidden_states: Optional[List] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        vlm_attn_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        skip_layers: Optional[List[int]] = None,
    ) -> Union[List[torch.Tensor], ActionTransformerModelOutput]: 
        assert joint_attention_kwargs is not None

        if self.config.n_conditions > 0: # type: ignore
            assert joint_attention_kwargs["traj2ds"].shape[1] == self.config.n_conditions, "model n_conditions does not match" # type: ignore

        # L
        if self.combined_temb:
            temb = self.time_ins_embed(timestep, pooled_projections)
        else: 
            temb = self.time_ins_embed(timestep)

        """ lang_hidden_states = self.lang_embedder(encoder_hidden_states) """

        # A
        noisy_action = joint_attention_kwargs.pop("action_hidden_embeds")
        action_hidden_states = self.action_proj_in(noisy_action)

        # Obs: V+Proprio
        obs_hidden_states, obs_token_mask = self.obs_proj(
            views=joint_attention_kwargs["views"],
            obs=joint_attention_kwargs["obs"],
            traj2ds=joint_attention_kwargs["traj2ds"],
            text_embeddings=pooled_projections,
            vlm_attn_mask=vlm_attn_mask
        ) # S, B, d_model
        
        for index_block, block in enumerate(self.transformer_blocks):
            is_skip = True if skip_layers is not None and index_block in skip_layers else False
            if is_skip:
                continue

            action_hidden_states, obs_hidden_states = block(
                # hidden_states=hidden_states, # V
                # lang_hidden_states=lang_hidden_states, # L
                action_hidden_states=action_hidden_states, # A
                obs_hidden_states=obs_hidden_states, # O
                temb=temb,
                obs_token_mask=obs_token_mask
                # time_enc=time_enc,
                # obs_pos_emb=obs_pos_emb,
                # joint_attention_kwargs=joint_attention_kwargs,
            )

        action_output = self.action_proj_out(
            x=action_hidden_states,
            t=temb,
            # cond=obs_hidden_states,
        )
        return ActionTransformerModelOutput(action=action_output)


class DiTActionTransformerModel(
    ModelMixin, ConfigMixin #, VLTTransformerActionMixin
    # PeftAdapterMixin, FromOriginalModelMixin, ReprodMixin, SD3Transformer2DLoadersMixin, 
):  
    """  An action transformer model """

    _supports_gradient_checkpointing = True
    # _no_split_modules = ["ActionTransformerBlock"]
    # _skip_layerwise_casting_patterns = ["pos_embed", "norm"]

    @register_to_config
    def __init__(
        self,
        # sample_size: int = 128,
        # patch_size: int = 2,
        # in_channels: int = 16,
        # num_layers: int = 24,
        attention_head_dim: int = 64,
        num_attention_heads: int = 24,
        # joint_attention_dim: int = 4096,
        caption_projection_dim: int = 1536,
        pooled_projection_dim: int = 2048,
        # out_channels: int = 16,
        pos_embed_max_size: int = 192,
        dual_attention_layers: Tuple[
            int, ...
        ] = (),  # () for sd3.0; (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12) for sd3.5
        qk_norm: Optional[str] = None,
        # VLA parameters
        training_phase: str = "joint",  # "joint", "action", "traj2d", "action_mini"
        # action head parameters
        action_num_blocks:int = 6,
        action_pred_horizon: int = 6,
        action_dim: int = 7,
        action_hidden_dim: int = 1536, #this
        action_nheads: int = 24,
        # action_attention_head_dim: Optional[int] = 64,
        n_conditions: int = 0,
        token_fusion: str = "concat",  # "concat", "cross"
        resnet_store_path: str = "cache/visual_features/resnet18/IN_1M_resnet18.pth",
        odim: int = 32,
        view_feature_dim: int = 1920,
        use_film: bool = False,
        combined_temb: bool = False
    ):
        super().__init__()
        # self.inner_dim = num_attention_heads * attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim

        # Set the observation projection
        self.obs_proj = ObservationProjection(
            action_pred_horizon=action_pred_horizon, 
            action_dim=action_dim, 
            hidden_dim=self.inner_dim, # not used
            output_dim=action_hidden_dim,
            n_conditions=n_conditions,
            token_fusion="concat",
            resnet_store_path=resnet_store_path,
            odim=odim,
            view_feature_dim=view_feature_dim,
            use_film=use_film
        )

        total_params = sum(p.numel() for p in self.obs_proj.parameters() if p.requires_grad)
        logger.debug(f"ObservationEncoder parameters: {total_params:,}")


        # q_former:
        # TODO: boqian: hardcode here
        num_query_tokens = 256
        num_heads_ = 16
        self.query_tokens = nn.Parameter(torch.randn(num_query_tokens, action_hidden_dim))
        self.cross_attention = CrossAttentionBlock(action_hidden_dim, num_heads_)


        # assert action_hidden_dim == 1024
        self.DiT = DiT(
            in_channels=action_dim,
            depth=24,
            num_heads=16,
            token_size=action_hidden_dim,
            learn_sigma=False,
            class_dropout_prob=0,
            future_action_window_size=action_pred_horizon-1,
            past_action_window_size=0,
            n_conditon_token=num_query_tokens
        )
        total_params = sum(p.numel() for p in self.DiT.parameters() if p.requires_grad)
        logger.debug(f"DiT parameters: {total_params:,}")


    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        pooled_projections: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        block_controlnet_hidden_states: Optional[List] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        vlm_attn_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        skip_layers: Optional[List[int]] = None,
    ) -> Union[List[torch.Tensor], ActionTransformerModelOutput]: 
        assert joint_attention_kwargs is not None

        if self.config.n_conditions > 0: # type: ignore
            assert joint_attention_kwargs["traj2ds"].shape[1] == self.config.n_conditions, "model n_conditions does not match" # type: ignore


        """ lang_hidden_states = self.lang_embedder(encoder_hidden_states) """

        # A
        noisy_action = joint_attention_kwargs.pop("action_hidden_embeds")

        # Obs: V+Proprio
        condition_tokens, vlm_attn_mask = self.obs_proj(
            views=joint_attention_kwargs["views"],
            obs=joint_attention_kwargs["obs"],
            traj2ds=joint_attention_kwargs["traj2ds"],
            text_embeddings=pooled_projections,
            vlm_attn_mask=vlm_attn_mask
        ) # B, S, d_model

        # print("condition_tokens.shape:", condition_tokens.shape) # [B, 106, 1024])


        # q_former
        # TODO: forward vlm_attn_mask here 
        query = self.query_tokens.unsqueeze(0).expand(condition_tokens.shape[0], -1, -1) # (B, Q, D)
        queried = self.cross_attention(query, condition_tokens) # (B, Q, D)


        # DiT
        action_output = self.DiT(noisy_action, timestep, queried)
        

        return ActionTransformerModelOutput(action=action_output)


class HumanFoundationModel(nn.Module):

    def __init__(self, model_cfg, feature_extractors: dict[str, str]):
        super().__init__()
        
        # self.qwen = _QWen_VL_Interface(config=model_cfg)
        # self.layer_qformer = get_layerwise_qformer(config=self.config)
        # self.action_model = get_action_model(config=self.config)

        self.action_header = ActionTransformerModel(
            resnet_store_path=model_cfg.resnet_store_path,
            odim=model_cfg.odim,
            action_dim=model_cfg.action_dim,
            action_pred_horizon=model_cfg.action_chunk_size,
            view_feature_dim=model_cfg.view_feature_dim,
            use_film=model_cfg.use_film,
            combined_temb=model_cfg.combined_temb,
            action_hidden_dim=model_cfg.hidden_dim,
            )

        total_params = sum(p.numel() for p in self.action_header.parameters())
        overwatch.info(f"Total ActionTransformerModel parameters: {total_params:,}")

        vision_backbones = {}
        for backbone, model_path in feature_extractors.items():
            if backbone == "siglip":
                siglip = SiglipModel.from_pretrained(
                    model_path
                )  # , resume_download=None, local_files_only=True
                vision_backbones[backbone] = siglip.vision_model
                vision_backbones[backbone].head = nn.Identity() # remove classification head
                del siglip
            elif backbone == "dinov2":
                vision_backbones[backbone] = Dinov2Model.from_pretrained(
                    model_path
                )  # , resume_download=None, local_files_only=True
                vision_backbones[backbone].embeddings.mask_token = None
            else:
                raise ValueError(f"Unsupported vision backbone: {backbone}")
            
        self.vision_backbones = nn.ModuleDict(vision_backbones)

        for name, backbone in self.vision_backbones.items():
            num_parameters = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
            overwatch.info(f"Vision backbone {name} has {num_parameters*1e-6:,}M parameters")
            
    def tune_vision_backbones(self, backbone_names: List[str]):
        for name, backbone in self.vision_backbones.items():
            if name in backbone_names:
                overwatch.info(f"Tuning vision backbone: {name}")
                for param in backbone.parameters():
                    param.requires_grad = True
            else:
                overwatch.info(f"Freezing vision backbone: {name}")
                for param in backbone.parameters():
                    param.requires_grad = False
    
    def extract_vision_features(self, batch):
        features = []
        for vision_backbone, model in self.vision_backbones.items():
            assert vision_backbone in batch
            B, V, C, H, W = batch[vision_backbone].shape
            if vision_backbone == "siglip":
                vision_outputs = model(
                    pixel_values=batch[vision_backbone].reshape(B * V, C, H, W),
                    return_dict=True,
                )
                siglip_features = vision_outputs.last_hidden_state
                features.append(siglip_features.view(B, V, *siglip_features.shape[1:]))
            elif vision_backbone == "dinov2":
                dinov2_outputs = model(
                    pixel_values=batch[vision_backbone].view(B * V, C, H, W)
                )
                dinov2_features = dinov2_outputs.last_hidden_state[:, 1:, :]
                features.append(dinov2_features.view(B, V, *dinov2_features.shape[1:]))
            else:
                raise ValueError(f"Unsupported vision backbone: {vision_backbone}")
        return torch.cat(features, dim=3)  # (B,V,N,D)
    
    def forward(
        self,
        action_samples: torch.Tensor,
        views: dict[str, torch.Tensor],
        states: torch.Tensor,
        # hidden_states: torch.Tensor,
        lang_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        timestep: torch.LongTensor,
        # block_controlnet_hidden_states: List,
        # joint_attention_kwargs: Optional[Dict[str, Any]],
        traj2ds: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        skip_layers: Optional[List[int]] = None,
    ) -> Union[List[torch.Tensor], HumanFoundationModelOutput]: 

        view_features = self.extract_vision_features(views)

        model_output = self.action_header(
            hidden_states=None,
            encoder_hidden_states=lang_hidden_states, # (B, 333, 4096)
            timestep=timestep,
            pooled_projections=pooled_projections, # (B, 2048)
            joint_attention_kwargs=dict(
                action_hidden_embeds=action_samples, # (B,Tp,Da)
                views=view_features,  # (B,V,N,D)
                obs=states,  # (B,1,M)
                traj2ds=traj2ds,  # (B, C, 3, H, W)
                # text_embeddings=text_embeddings
            ),
            return_dict=return_dict,
        )
        return HumanFoundationModelOutput(action=model_output.action)

    @torch.inference_mode()
    def predict_action(
        self,
        observations: List[List[Image.Image]],  # B * List of PIL Image as [view1, view2]
        states: List[Any],
        instructions: List[str],
        **kwargs: str
    ) -> np.ndarray:
        # with torch.autocast("cuda", dtype=torch.float32):
        ...

    @torch.inference_mode()
    def chat(self):
        ...
    


class Psi0Model(nn.Module):

    def __init__(self, model_cfg, vlm_model: Qwen3VLForConditionalGeneration):
        super().__init__()
        
        if model_cfg.use_dit:
            self.action_header = DiTActionTransformerModel(
                resnet_store_path=model_cfg.resnet_store_path,
                odim=model_cfg.odim,
                action_dim=model_cfg.action_dim,
                action_pred_horizon=model_cfg.action_chunk_size,
                view_feature_dim=model_cfg.view_feature_dim,
                use_film=model_cfg.use_film,
                combined_temb=model_cfg.combined_temb,
                action_hidden_dim=model_cfg.hidden_dim,
            )
        else:
            self.action_header = ActionTransformerModel(
                resnet_store_path=model_cfg.resnet_store_path,
                odim=model_cfg.odim,
                action_dim=model_cfg.action_dim,
                action_pred_horizon=model_cfg.action_chunk_size,
                view_feature_dim=model_cfg.view_feature_dim,
                use_film=model_cfg.use_film,
                combined_temb=model_cfg.combined_temb,
                action_hidden_dim=model_cfg.hidden_dim,
            )


        total_params = sum(p.numel() for p in self.action_header.parameters())
        overwatch.info(f"Total ActionTransformerModel parameters: {total_params:,}")

        self.vlm_model = vlm_model
        total_params = sum(p.numel() for p in self.vlm_model.parameters())
        overwatch.info(f"Total VLM Backbone parameters: {total_params:,}")

    # load pretrained vlm+action head, and all the modules needed for predict_action
    @classmethod
    def from_pretrained(
        cls,
        run_dir,
        ckpt_step,
        launch_config: LaunchConfig,
        device,
        checkpoint_model_path=None,
    ):
        if not os.path.exists(run_dir):
            raise ValueError(f"Pretrained model path {run_dir} does not exist.")

        from safetensors.torch import load_file
        file_path = (
            os.fspath(checkpoint_model_path)
            if checkpoint_model_path is not None
            else f"{run_dir}/checkpoints/ckpt_{ckpt_step}/model.safetensors"
        )
        if not os.path.exists(file_path):
            raise ValueError(f"Checkpoint file {file_path} does not exist.")
        state_dict = load_file(file_path, device="cpu")

        # Init an empty VLM backbone from config only (skip loading the base
        # weights).  Evaluation often runs on offline compute nodes, while the
        # training config already points at a complete local backbone snapshot.
        # Prefer that snapshot so loading a local checkpoint never requires a
        # Hugging Face network lookup.
        configured_vlm_path = os.fspath(launch_config.model.model_name_or_path)
        vlm_config_source = (
            configured_vlm_path
            if os.path.isfile(os.path.join(configured_vlm_path, "config.json"))
            else QWEN3VL_VARIANT
        )
        vlm_config = AutoConfig.from_pretrained(vlm_config_source)
        # Attention backend: flash_attention_2 on CUDA (if installed), sdpa on XPU/CPU.
        try:
            from transformers.utils import is_flash_attn_2_available
            vlm_config._attn_implementation = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
        except Exception:
            vlm_config._attn_implementation = "sdpa"
        vlm_config.dtype = torch.bfloat16
        vlm_config.vision_config.dtype = torch.bfloat16
        vlm_config.text_config.dtype = torch.bfloat16
        vlm_model = Qwen3VLForConditionalGeneration(vlm_config)
        vlm_model = vlm_model.to(dtype=torch.bfloat16) # type: ignore

        vlm_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("action_header."):
                continue
            elif k.startswith('vlm_model.'):
                vlm_state_dict[k.replace("vlm_model.", "")] = v
            else:
                assert False, "check here"
        vlm_state_dict["lm_head.weight"] = vlm_state_dict["model.language_model.embed_tokens.weight"] # TODO check here

        if vlm_state_dict["lm_head.weight"].shape[0] != vlm_model.lm_head.weight.shape[0]:
            vlm_model.resize_token_embeddings(
                vlm_state_dict["lm_head.weight"].shape[0], 
                pad_to_multiple_of = 192,
                mean_resizing = True
            )
            overwatch.info(f"Resized model token embeddings to {vlm_model.lm_head.weight.shape[0]}")

        vlm_model.load_state_dict(vlm_state_dict, strict=True)
        overwatch.info("loaded vlm_backbone checkpoint successfully.")

        # init hfm-together model with vlm backbone
        model = Psi0Model(
            model_cfg=launch_config.model,
            vlm_model = vlm_model,
        )

        # load action head
        action_head_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("vlm_model."):
                continue
            elif k.startswith('action_header.'):
                action_head_state_dict[k.replace("action_header.", "")] = v
            else:
                assert False, "check here"
        model.action_header.load_state_dict(action_head_state_dict, strict=True)
        overwatch.info("loaded action head checkpoint successfully.")

        # load necessary modules
        model.vlm_processor = AutoProcessor.from_pretrained(vlm_config_source)
        # model.tokenizer = model.vlm_processor.tokenizer

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
            from diffusers.schedulers.scheduling_flow_match_euler_discrete import (FlowMatchEulerDiscreteScheduler)
            scheduler= FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=launch_config.model.train_diffusion_steps, # MUST be 1000 as per pretrained SD3
            )
        else:
            raise ValueError(f"Unsupported noise scheduler: {launch_config.model.noise_scheduler}")

        model.noise_scheduler = scheduler
        model.action_horizon = launch_config.model.action_chunk_size
        model.action_dim = launch_config.model.action_dim
        model.device = device
        return model
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
    ) -> Union[List[torch.Tensor], HumanFoundationModelOutput]: 

        # extract vision + language features
        vlm_hidden_states = self.vlm_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            return_dict=True
        ).hidden_states[-1]
        # print(vlm_hidden_states.shape)

        model_output = self.action_header(
            hidden_states=None,
            timestep=timestep,
            joint_attention_kwargs=dict(
                action_hidden_embeds=action_samples, # (B,Tp,Da)
                views=vlm_hidden_states.unsqueeze(1),  # (B,V,N,D)
                obs=states,  # (B,1,M)
                traj2ds=traj2ds,  # (B, C, 3, H, W)
            ),
            vlm_attn_mask=attention_mask, # (B, seq_len)
            return_dict=return_dict,
        )
        return HumanFoundationModelOutput(action=model_output.action)

    @torch.inference_mode()
    def encode_condition(
        self,
        observations: List[List[Image.Image]],
        states: torch.Tensor,
        instructions: List[str],
        traj2ds: Optional[torch.Tensor] = None,
    ) -> Psi0Condition:
        """Encode an observation once for reuse across flow evaluations."""
        bsz = int(states.shape[0])
        if len(observations) != bsz or len(instructions) != bsz:
            raise ValueError(
                "Batch mismatch: expected observations, states, and instructions "
                f"to have the same batch size, got {len(observations)}, {bsz}, "
                f"and {len(instructions)}."
            )

        batch_messages = []
        texts = []
        expected_num_views = []

        for observation, instruction in zip(observations, instructions):
            expected_num_views.append(len(observation))
            content = [{"type": "image", "image": img} for img in observation]
            content.append({"type": "text", "text": instruction})
            user_msg = {"role": "user", "content": content}
            messages = [user_msg]
            batch_messages.append(messages)
            texts.append(
                self.vlm_processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )

        # Qwen's multimodal tensors are not image-batched tensors.  The
        # processor pads ``input_ids`` across text rows, but concatenates all
        # image patches and image grids in prompt order:
        #   input_ids:     [B, N]
        #   pixel_values:  [sum(image_patches), patch_dim]
        #   image_grid_thw:[sum(images), 3]
        # Calling the processor once for the whole batch preserves that shared
        # ordering.  Stacking per-sample outputs would add a spurious batch
        # dimension to ``image_grid_thw`` for multi-view observations and would
        # fail for variable prompt/image lengths.
        image_inputs, video_inputs = process_vision_info(
            batch_messages, image_patch_size=16
        )
        inputs = self.vlm_processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        batch_input_ids = inputs["input_ids"]
        batch_attention_mask = inputs["attention_mask"]
        batch_pixel_values = inputs["pixel_values"]
        batch_image_grid_thw = inputs["image_grid_thw"]

        if tuple(batch_input_ids.shape[:1]) != (bsz,) or batch_input_ids.ndim != 2:
            raise ValueError(
                "Qwen processor input_ids must have shape [B,N], got "
                f"{tuple(batch_input_ids.shape)} for B={bsz}"
            )
        if tuple(batch_attention_mask.shape) != tuple(batch_input_ids.shape):
            raise ValueError(
                "Qwen processor attention_mask must match input_ids, got "
                f"{tuple(batch_attention_mask.shape)} and "
                f"{tuple(batch_input_ids.shape)}"
            )
        if batch_pixel_values.ndim != 2:
            raise ValueError(
                "Qwen processor pixel_values must have flattened shape "
                "[sum(image_patches),patch_dim], got "
                f"{tuple(batch_pixel_values.shape)}"
            )
        if batch_image_grid_thw.ndim != 2 or batch_image_grid_thw.shape[1] != 3:
            raise ValueError(
                "Qwen processor image_grid_thw must have flattened shape "
                f"[sum(images),3], got {tuple(batch_image_grid_thw.shape)}"
            )
        total_expected_views = sum(expected_num_views)
        if int(batch_image_grid_thw.shape[0]) != total_expected_views:
            raise ValueError(
                "Qwen processor image count does not match observations: "
                f"expected {total_expected_views}, got "
                f"{int(batch_image_grid_thw.shape[0])}"
            )
        image_token_id = _resolve_image_token_id(
            self.vlm_model, self.vlm_processor
        )
        vision_segment_ids = vision_segment_ids_from_input_ids(
            batch_input_ids,
            image_token_id,
            expected_num_views,
        )

        with torch.autocast(str(self.device).split(":")[0], dtype=torch.bfloat16):
            output = self.vlm_model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                pixel_values=batch_pixel_values,
                image_grid_thw=batch_image_grid_thw,
                output_hidden_states=True,
                return_dict=True
            )
            vlm_hidden_states = output.hidden_states[-1].unsqueeze(1)

        states = states.to(self.device)
        if traj2ds is not None:
            traj2ds = traj2ds.to(self.device)
        return Psi0Condition(
            views=vlm_hidden_states,
            states=states,
            traj2ds=traj2ds,
            attention_mask=batch_attention_mask,
            vision_segment_ids=vision_segment_ids,
        )

    def _prepare_action_noise(
        self,
        batch_size: int,
        noise: Optional[torch.Tensor],
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        expected_shape = (batch_size, self.action_horizon, self.action_dim)
        if noise is None:
            return torch.randn(
                expected_shape,
                device=self.device,
                dtype=torch.float32,
                generator=generator,
            )

        if generator is not None:
            raise ValueError("generator cannot be provided together with explicit noise")
        if tuple(noise.shape) != expected_shape:
            raise ValueError(
                f"Expected noise shape {expected_shape}, got {tuple(noise.shape)}."
            )
        if not torch.isfinite(noise).all():
            raise ValueError("Initial action noise must contain only finite values.")
        return noise.to(device=self.device, dtype=torch.float32).clone()

    def _prepare_flow_timestep(
        self,
        timestep: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        timestep = torch.as_tensor(timestep, device=self.device)
        if timestep.ndim == 0:
            return timestep.expand(batch_size)
        if timestep.shape[0] == batch_size:
            return timestep
        if timestep.shape[0] == 1:
            return timestep.expand(batch_size, *timestep.shape[1:])
        raise ValueError(
            f"Expected timestep batch dimension 1 or {batch_size}, "
            f"got shape {tuple(timestep.shape)}."
        )

    def _predict_velocity_from_condition(
        self,
        condition: Psi0Condition,
        action_samples: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        # ActionTransformerModel.forward pops action_hidden_embeds, so this dict
        # must be freshly allocated for every flow evaluation.
        return self.action_header(
            hidden_states=None,
            timestep=timestep,
            joint_attention_kwargs=dict(
                action_hidden_embeds=action_samples,
                views=condition.views,
                obs=condition.states,
                traj2ds=condition.traj2ds,
            ),
            # Preserve the legacy online sampler exactly. The mask remains in
            # Psi0Condition for adapter pooling and future padded-batch support.
            vlm_attn_mask=None,
            return_dict=True,
        ).action

    @torch.inference_mode()
    def predict_velocity(
        self,
        action_samples: torch.Tensor,
        timestep: torch.Tensor,
        condition: Psi0Condition,
    ) -> torch.Tensor:
        """Evaluate the frozen Psi0 action velocity field once."""
        expected_shape = (
            condition.batch_size,
            self.action_horizon,
            self.action_dim,
        )
        if tuple(action_samples.shape) != expected_shape:
            raise ValueError(
                f"Expected action sample shape {expected_shape}, "
                f"got {tuple(action_samples.shape)}."
            )
        action_samples = action_samples.to(self.device)
        batched_timestep = self._prepare_flow_timestep(
            timestep, condition.batch_size
        )
        with torch.autocast(str(self.device).split(":")[0], dtype=torch.bfloat16):
            return self._predict_velocity_from_condition(
                condition, action_samples, batched_timestep
            )

    @torch.inference_mode()
    def decode_noise(
        self,
        condition: Psi0Condition,
        initial_noise: torch.Tensor,
        num_inference_steps: int,
    ) -> torch.Tensor:
        """Decode explicit noise with a stateless, legacy-equivalent sampler."""
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")

        action_samples = self._prepare_action_noise(
            condition.batch_size, initial_noise
        )

        with torch.autocast(str(self.device).split(":")[0], dtype=torch.bfloat16):
            if hasattr(self.noise_scheduler, "sigmas"):
                schedule = FlowSchedule.from_scheduler(
                    self.noise_scheduler,
                    num_inference_steps,
                )

                def velocity_fn(
                    sample: torch.Tensor, timestep: torch.Tensor
                ) -> torch.Tensor:
                    batched_timestep = self._prepare_flow_timestep(
                        timestep, condition.batch_size
                    )
                    return self._predict_velocity_from_condition(
                        condition, sample, batched_timestep
                    )

                action_samples = decode_flow(
                    action_samples, schedule, velocity_fn
                )
            else:
                # Preserve the legacy DDIM path; FlowDAgger only supports the
                # flow scheduler above.
                self.noise_scheduler.set_timesteps(num_inference_steps)
                for timestep in self.noise_scheduler.timesteps:
                    batched_timestep = self._prepare_flow_timestep(
                        timestep, condition.batch_size
                    )
                    model_pred = self._predict_velocity_from_condition(
                        condition, action_samples, batched_timestep
                    )
                    action_samples = self.noise_scheduler.step(
                        model_output=model_pred,
                        timestep=timestep,
                        sample=action_samples,  # type: ignore
                    ).prev_sample
        return action_samples.float()

    @torch.inference_mode()
    def invert_action_flow(
        self,
        condition: Psi0Condition,
        clean_action: torch.Tensor,
        num_inference_steps: int,
        num_fixed_point_steps: int = 5,
    ) -> FlowInversionResult:
        """Invert a normalized clean action chunk to its initial flow noise."""
        if self.training:
            raise RuntimeError(
                "Flow inversion requires Psi0Model.eval() so dropout is disabled."
            )
        if any(parameter.requires_grad for parameter in self.parameters()):
            raise RuntimeError(
                "Flow inversion requires all Base Psi0 parameters to be frozen."
            )
        if not hasattr(self.noise_scheduler, "sigmas"):
            raise ValueError("Flow inversion requires a flow-matching scheduler")
        expected_shape = (
            condition.batch_size,
            self.action_horizon,
            self.action_dim,
        )
        if tuple(clean_action.shape) != expected_shape:
            raise ValueError(
                f"Expected clean action shape {expected_shape}, "
                f"got {tuple(clean_action.shape)}."
            )
        if not torch.isfinite(clean_action).all():
            raise ValueError("Clean action must contain only finite values.")

        schedule = FlowSchedule.from_scheduler(
            self.noise_scheduler,
            num_inference_steps,
        )
        clean_action = clean_action.to(device=self.device, dtype=torch.float32)

        with torch.autocast(str(self.device).split(":")[0], dtype=torch.bfloat16):
            def velocity_fn(
                sample: torch.Tensor, timestep: torch.Tensor
            ) -> torch.Tensor:
                batched_timestep = self._prepare_flow_timestep(
                    timestep, condition.batch_size
                )
                return self._predict_velocity_from_condition(
                    condition, sample, batched_timestep
                )

            return invert_flow_fixed_point(
                clean_action,
                schedule,
                velocity_fn,
                num_fixed_point_steps=num_fixed_point_steps,
            )

    @torch.inference_mode()
    def predict_action_with_noise(
        self,
        observations: List[List[Image.Image]],
        states: torch.Tensor,
        instructions: List[str],
        num_inference_steps: int,
        traj2ds: Optional[torch.Tensor],
        noise: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict actions and return the exact initial noise that was used."""
        condition = self.encode_condition(
            observations=observations,
            states=states,
            instructions=instructions,
            traj2ds=traj2ds,
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
    def predict_action(
        self,
        observations: List[List[Image.Image]],  # B * List of PIL Image as [view1, view2]
        states: torch.Tensor, # (B, Ts, Ds)
        instructions: List[str], # (B,)
        num_inference_steps: int,
        traj2ds,
        noise: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        actions, _ = self.predict_action_with_noise(
            observations=observations,
            states=states,
            instructions=instructions,
            num_inference_steps=num_inference_steps,
            traj2ds=traj2ds,
            noise=noise,
            generator=generator,
        )
        return actions

    @torch.inference_mode()
    def predict_action_with_training_rtc_flow(
        self,
        observations: List[List[Image.Image]],  # B * List of PIL Image as [view1, view2]
        states: torch.Tensor, # (B, Ts, Ds)
        instructions: List[str], # (B,)
        num_inference_steps: int,
        traj2ds, 
        prev_actions: torch.Tensor = None, # (1, H, D)
        inference_delay: int = 0,
        max_delay: int = 0,
        **kwargs: str
    ) -> torch.Tensor:

        ## RTC related ##
        H = self.action_horizon
        assert prev_actions is not None and inference_delay > 0 and max_delay > 0, "prev_actions, inference_delay and max_delay must be provided"
        assert prev_actions.shape[0] == 1
        prev_actions = prev_actions.to(device=self.device, dtype=torch.float32)

        # Create soft mask for inpainting
        d = inference_delay

        # Validate constraint from paper: d ≤ s ≤ H - d
        assert d < H, f"Constraint violated: d={d}, H={H}. Need d < H"
        assert d < max_delay, f"Constraint violated: d={d}, max_delay={max_delay}. Need d < max_delay"

        prefix_mask = torch.arange(H, device=self.device)[None, :] < d # shape (1, H)
        ##              ##


        bsz = states.shape[0]
        batch_input_ids = []
        batch_attention_mask = []
        batch_pixel_values = []
        batch_image_grid_thw = []


        for observation, instruction in zip(observations, instructions):
            messages = []
            content = [{"type": "image", "image": img} for img in observation]
            content.append({"type": "text", "text": instruction})
            user_msg = {"role": "user", "content": content}
            messages.append([user_msg])
            texts = [
                self.vlm_processor.apply_chat_template(
                    m, tokenize=False, add_generation_prompt=True
                )
                for m in messages
            ]
            image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
            inputs = self.vlm_processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            # input_ids torch.Size([1, 80])
            # attention_mask torch.Size([1, 80])
            # pixel_values torch.Size([256, 1536])
            # image_grid_thw torch.Size([1, 3])
            batch_input_ids.append(inputs['input_ids'].squeeze(0))
            batch_attention_mask.append(inputs['attention_mask'].squeeze(0))
            batch_pixel_values.append(inputs['pixel_values'])
            batch_image_grid_thw.append(inputs['image_grid_thw'].squeeze(0))

        batch_input_ids = torch.stack(batch_input_ids) # (B, 80)
        batch_attention_mask = torch.stack(batch_attention_mask) # (B, 80)
        batch_pixel_values = torch.stack(batch_pixel_values) # (B, 256, 1536)
        batch_image_grid_thw = torch.stack(batch_image_grid_thw) # (B, 3)

        with torch.autocast(str(self.device).split(":")[0], dtype=torch.bfloat16):
            # extract vision + language features
            output = self.vlm_model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                pixel_values=batch_pixel_values,
                image_grid_thw=batch_image_grid_thw,
                output_hidden_states=True,
                return_dict=True
            )
            vlm_hidden_states_ = output.hidden_states # len(vlm_hidden_states_) == 29
            

            # use hidden states from the last layer
            vlm_hidden_states = vlm_hidden_states_[-1] # shape (B, seq_len, D_h)  shape(16, 80, 2048)
            vlm_hidden_states = vlm_hidden_states.unsqueeze(1) # shape (B, 1, seq_len, D_h) (16, 1, 80, 2048)

            # generate action from noise
            action_samples = torch.randn(
                bsz, self.action_horizon, self.action_dim, device=self.device
            )
            self.noise_scheduler.set_timesteps(num_inference_steps)

            # self.noise_scheduler.timesteps: tensor([1000.,  889.,  778.,  667.,  556.,  445.,  334.,  223.,  112.,    1.])
            # self.noise_scheduler.sigmas: tensor([1.0000, 0.8890, 0.7780, 0.6670, 0.5560, 0.4450, 0.3340, 0.2230, 0.1120, 0.0010, 0.0000])

            for i, timestep in enumerate(self.noise_scheduler.timesteps):
                # batched_timestep = timestep.expand(bsz).to(self.device).detach()

                batched_timestep_masked = torch.where(prefix_mask, 0, timestep.to(self.device)) # shape (B, H)

                # replace action_samples with clean prev_actions when prefix_mask == True
                action_samples = torch.where(prefix_mask[:, :, None], prev_actions, action_samples)

                model_pred = self.action_header(
                    hidden_states=None,
                    timestep=batched_timestep_masked,
                    joint_attention_kwargs=dict(
                        action_hidden_embeds=action_samples, # (B,Tp,Da)
                        views=vlm_hidden_states,  # (B,V,N,D)
                        obs=states,  # (B,1,M)
                        traj2ds=traj2ds,  # (B, C, 3, H, W)
                    ),
                    return_dict=True,
                ).action

                action_samples = self.noise_scheduler.step(
                    model_output=model_pred, timestep=timestep, sample=action_samples # type: ignore
                ).prev_sample

                # if i == len(self.noise_scheduler.timesteps) - 1:
                #     action_samples = torch.where(prefix_mask[:, :, None], prev_actions, action_samples)

        return action_samples.float()

    def _prepare_rtc_inputs(
        self,
        condition: Psi0Condition,
        prev_actions: torch.Tensor,
        committed_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate and canonicalize the immutable RTC prefix contract.

        ``committed_mask`` may be ``[B, H]`` or ``[B, H, D]``.  The returned
        mask is always boolean ``[B, H, D]`` so every action dimension is
        governed explicitly.  FlowDAgger RTC targets use this exact mask in
        both decoding and inversion; a non-RTC target is therefore impossible
        to load accidentally as an RTC suffix target.
        """
        expected = (
            condition.batch_size,
            self.action_horizon,
            self.action_dim,
        )
        if tuple(prev_actions.shape) != expected:
            raise ValueError(
                f"Expected prev_actions shape {expected}, got "
                f"{tuple(prev_actions.shape)}."
            )
        prev_actions = prev_actions.to(device=self.device, dtype=torch.float32)
        if not torch.isfinite(prev_actions).all():
            raise ValueError("prev_actions must contain only finite values")

        mask = torch.as_tensor(committed_mask, device=self.device)
        if mask.dtype is not torch.bool:
            if not torch.all((mask == 0) | (mask == 1)):
                raise ValueError("committed_mask must be boolean or binary")
            mask = mask.bool()
        if tuple(mask.shape) == expected[:2]:
            mask = mask.unsqueeze(-1).expand(expected)
        elif tuple(mask.shape) != expected:
            raise ValueError(
                "committed_mask must have shape "
                f"{expected[:2]} or {expected}, got {tuple(mask.shape)}."
            )
        token_any = mask.any(dim=-1)
        token_all = mask.all(dim=-1)
        if not torch.equal(token_any, token_all):
            raise ValueError(
                "committed_mask must commit all action dimensions of a token"
            )
        if torch.any(token_all[:, 1:] & ~token_all[:, :-1]):
            raise ValueError("committed_mask must describe one contiguous prefix")
        return prev_actions, mask

    @torch.inference_mode()
    def decode_noise_rtc(
        self,
        condition: Psi0Condition,
        initial_noise: torch.Tensor,
        prev_actions: torch.Tensor,
        committed_mask: torch.Tensor,
        num_inference_steps: int,
    ) -> torch.Tensor:
        """Decode explicit noise while preserving an RTC committed prefix.

        This is deliberately a separate decoder mode (``rtc_suffix_v1``), not
        an option on :meth:`decode_noise`.  Committed values are re-applied
        before the velocity evaluation and after every Euler step, making the
        prefix invariant exact rather than scheduler-dependent.
        """
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        if not hasattr(self.noise_scheduler, "sigmas"):
            raise ValueError("RTC suffix decoding requires a flow scheduler")

        prev_actions, committed = self._prepare_rtc_inputs(
            condition, prev_actions, committed_mask
        )
        sample = self._prepare_action_noise(
            condition.batch_size, initial_noise
        )
        schedule = FlowSchedule.from_scheduler(
            self.noise_scheduler, num_inference_steps
        )
        sample = torch.where(committed, prev_actions, sample)

        with torch.autocast(str(self.device).split(":")[0], dtype=torch.bfloat16):
            for index in range(schedule.num_steps):
                timestep = schedule.timesteps[index].to(self.device)
                token_timestep = torch.full(
                    (condition.batch_size, self.action_horizon),
                    timestep,
                    device=self.device,
                    dtype=timestep.dtype,
                )
                # The action head consumes one timestep per token in RTC mode.
                token_committed = committed.all(dim=-1)
                token_timestep = torch.where(
                    token_committed,
                    torch.zeros_like(token_timestep),
                    token_timestep,
                )
                sample = torch.where(committed, prev_actions, sample)
                velocity = self._predict_velocity_from_condition(
                    condition, sample, token_timestep
                )
                sample = flow_euler_step(
                    sample,
                    velocity,
                    schedule.sigmas[index],
                    schedule.sigmas[index + 1],
                    validate_finite=False,
                )
                sample = torch.where(committed, prev_actions, sample)

        if not torch.isfinite(sample).all():
            raise ValueError("RTC decoded action contains non-finite values")
        return sample.float()

    @torch.inference_mode()
    def invert_action_flow_rtc(
        self,
        condition: Psi0Condition,
        clean_action: torch.Tensor,
        prev_actions: torch.Tensor,
        committed_mask: torch.Tensor,
        free_suffix_mask: torch.Tensor,
        num_inference_steps: int,
        num_fixed_point_steps: int = 5,
    ) -> FlowInversionResult:
        """Invert only the free RTC suffix under an exact committed prefix."""
        if self.training:
            raise RuntimeError(
                "RTC flow inversion requires Psi0Model.eval()."
            )
        if any(parameter.requires_grad for parameter in self.parameters()):
            raise RuntimeError(
                "RTC flow inversion requires all Base parameters frozen."
            )
        if num_inference_steps <= 0 or num_fixed_point_steps <= 0:
            raise ValueError("inference and fixed-point steps must be positive")
        if not hasattr(self.noise_scheduler, "sigmas"):
            raise ValueError("RTC suffix inversion requires a flow scheduler")

        prev_actions, committed = self._prepare_rtc_inputs(
            condition, prev_actions, committed_mask
        )
        expected = tuple(prev_actions.shape)
        if tuple(clean_action.shape) != expected:
            raise ValueError(
                f"Expected clean_action shape {expected}, got "
                f"{tuple(clean_action.shape)}."
            )
        clean_action = clean_action.to(device=self.device, dtype=torch.float32)
        if not torch.isfinite(clean_action).all():
            raise ValueError("clean_action must contain only finite values")

        free = torch.as_tensor(free_suffix_mask, device=self.device)
        if free.dtype is not torch.bool:
            if not torch.all((free == 0) | (free == 1)):
                raise ValueError("free_suffix_mask must be boolean or binary")
            free = free.bool()
        if tuple(free.shape) == expected[:2]:
            free = free.unsqueeze(-1).expand(expected)
        elif tuple(free.shape) != expected:
            raise ValueError(
                "free_suffix_mask must have shape "
                f"{expected[:2]} or {expected}, got {tuple(free.shape)}."
            )
        if not torch.equal(free.any(dim=-1), free.all(dim=-1)):
            raise ValueError(
                "free_suffix_mask must free all action dimensions of a token"
            )
        if torch.any(committed & free) or not torch.all(committed | free):
            raise ValueError(
                "committed_mask and free_suffix_mask must be disjoint and "
                "cover the full action chunk"
            )
        if torch.any(free.reshape(condition.batch_size, -1).sum(dim=1) == 0):
            raise ValueError("free_suffix_mask must select a non-empty suffix")
        if not torch.equal(
            clean_action.masked_select(committed),
            prev_actions.masked_select(committed),
        ):
            raise ValueError(
                "clean_action committed prefix must exactly match prev_actions"
            )

        schedule = FlowSchedule.from_scheduler(
            self.noise_scheduler, num_inference_steps
        )
        x_next = torch.where(committed, prev_actions, clean_action).clone()
        residuals = torch.empty(
            schedule.num_steps,
            num_fixed_point_steps,
            device=self.device,
            dtype=torch.float32,
        )
        per_sample_residuals = torch.empty(
            condition.batch_size,
            schedule.num_steps,
            num_fixed_point_steps,
            device=self.device,
            dtype=torch.float32,
        )
        token_committed = committed.all(dim=-1)

        with torch.autocast(str(self.device).split(":")[0], dtype=torch.bfloat16):
            for reverse_index, index in enumerate(
                range(schedule.num_steps - 1, -1, -1)
            ):
                timestep = schedule.timesteps[index].to(self.device)
                token_timestep = torch.full(
                    (condition.batch_size, self.action_horizon),
                    timestep,
                    device=self.device,
                    dtype=timestep.dtype,
                )
                token_timestep = torch.where(
                    token_committed,
                    torch.zeros_like(token_timestep),
                    token_timestep,
                )
                dt = (
                    schedule.sigmas[index + 1] - schedule.sigmas[index]
                ).to(device=self.device, dtype=torch.float32)
                z = x_next.clone()
                for iteration in range(num_fixed_point_steps):
                    z = torch.where(committed, prev_actions, z)
                    velocity = self._predict_velocity_from_condition(
                        condition, z, token_timestep
                    )
                    updated = x_next - dt * velocity.float()
                    updated = torch.where(committed, prev_actions, updated)
                    sample_residual = (
                        ((updated - z).square() * free)
                        .reshape(condition.batch_size, -1)
                        .sum(dim=1)
                        / free.reshape(condition.batch_size, -1)
                        .sum(dim=1)
                        .clamp_min(1)
                    )
                    per_sample_residuals[:, reverse_index, iteration] = (
                        sample_residual
                    )
                    residuals[reverse_index, iteration] = (
                        sample_residual.mean()
                    )
                    z = updated
                x_next = z

        if not torch.isfinite(x_next).all():
            raise ValueError("RTC inverted noise contains non-finite values")
        return FlowInversionResult(
            noise=x_next.float(),
            residuals=residuals,
            per_sample_residuals=per_sample_residuals,
        )

    def predict_action_with_rtc_flow(
        self,
        observations: List[List[Image.Image]],  # B * List of PIL Image as [view1, view2]
        states: torch.Tensor, # (B, Ts, Ds)
        instructions: List[str], # (B,)
        num_inference_steps: int,
        traj2ds, 
        prev_actions: torch.Tensor = None, # (1, H, D)
        inference_delay: int = 0,
        execution_horizon: int = 0,
        mask_schedule: str = "exponential",
        guidance_weight: float = 5.0,
        **kwargs: str
    ) -> np.ndarray:
        with torch.no_grad():

            ## RTC related ##
            H = self.action_horizon
            assert prev_actions is not None and inference_delay > 0 and execution_horizon > 0, "prev_actions, inference_delay and execution_horizon must be provided"
            assert prev_actions.shape[0] == 1
            prev_actions = prev_actions.to(device=self.device, dtype=torch.float32)

            # Create soft mask for inpainting
            d = inference_delay
            s = execution_horizon

            # Validate constraint from paper: d ≤ s ≤ H - d
            assert d <= s and s <= H - d, f"Constraint violated: d={d}, s={s}, H={H}. Need d ≤ s ≤ H-d"

            # Create soft mask [H] according to Equation 5
            mask = self._create_soft_mask(
                H, d, s, schedule=mask_schedule, device=self.device
            )
            mask_expanded = mask.view(1, -1, 1).expand(1, self.action_horizon, self.action_dim)
            ##              ##


            bsz = states.shape[0]
            batch_input_ids = []
            batch_attention_mask = []
            batch_pixel_values = []
            batch_image_grid_thw = []


            for observation, instruction in zip(observations, instructions):
                messages = []
                content = [{"type": "image", "image": img} for img in observation]
                content.append({"type": "text", "text": instruction})
                user_msg = {"role": "user", "content": content}
                messages.append([user_msg])
                texts = [
                    self.vlm_processor.apply_chat_template(
                        m, tokenize=False, add_generation_prompt=True
                    )
                    for m in messages
                ]
                image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
                inputs = self.vlm_processor(
                    text=texts,
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                # input_ids torch.Size([1, 80])
                # attention_mask torch.Size([1, 80])
                # pixel_values torch.Size([256, 1536])
                # image_grid_thw torch.Size([1, 3])
                batch_input_ids.append(inputs['input_ids'].squeeze(0))
                batch_attention_mask.append(inputs['attention_mask'].squeeze(0))
                batch_pixel_values.append(inputs['pixel_values'])
                batch_image_grid_thw.append(inputs['image_grid_thw'].squeeze(0))

            batch_input_ids = torch.stack(batch_input_ids) # (B, 80)
            batch_attention_mask = torch.stack(batch_attention_mask) # (B, 80)
            batch_pixel_values = torch.stack(batch_pixel_values) # (B, 256, 1536)
            batch_image_grid_thw = torch.stack(batch_image_grid_thw) # (B, 3)

            with torch.autocast(str(self.device).split(":")[0], dtype=torch.bfloat16):
                # extract vision + language features
                output = self.vlm_model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    pixel_values=batch_pixel_values,
                    image_grid_thw=batch_image_grid_thw,
                    output_hidden_states=True,
                    return_dict=True
                )
                vlm_hidden_states_ = output.hidden_states # len(vlm_hidden_states_) == 29
                

                # use hidden states from the last layer
                vlm_hidden_states = vlm_hidden_states_[-1] # shape (B, seq_len, D_h)  shape(16, 80, 2048)
                vlm_hidden_states = vlm_hidden_states.unsqueeze(1) # shape (B, 1, seq_len, D_h) (16, 1, 80, 2048)

                # generate action from noise
                action_samples = torch.randn(
                    bsz, self.action_horizon, self.action_dim, device=self.device
                )
                # target_noise = action_samples.clone().detach()
                self.noise_scheduler.set_timesteps(num_inference_steps)

                # self.noise_scheduler.timesteps: tensor([1000.,  889.,  778.,  667.,  556.,  445.,  334.,  223.,  112.,    1.])
                # self.noise_scheduler.sigmas: tensor([1.0000, 0.8890, 0.7780, 0.6670, 0.5560, 0.4450, 0.3340, 0.2230, 0.1120, 0.0010, 0.0000])

                # self.noise_scheduler.timesteps: tensor([1000.,  889.,  778.,  667.,  556.,  445.,  334.,  223.,  112.,    1.])
                # self.noise_scheduler.sigmas: tensor([1.0000, 0.8890, 0.7780, 0.6670, 0.5560, 0.4450, 0.3340, 0.2230, 0.1120, 0.0010, 0.0000])

                for timestep in self.noise_scheduler.timesteps:
                    batched_timestep = timestep.expand(bsz).to(self.device).detach()

                    ### pseudo inverse (with gradient):
                    # Enable gradient only for this step
                    with torch.enable_grad():
                        # Detach and enable gradient only for sample_actions in this step
                        action_samples_grad = action_samples.detach().requires_grad_(True)

                        model_pred = self.action_header(
                            hidden_states=None,
                            timestep=batched_timestep,
                            joint_attention_kwargs=dict(
                                action_hidden_embeds=action_samples_grad, # (B,Tp,Da)
                                views=vlm_hidden_states,  # (B,V,N,D)
                                obs=states,  # (B,1,M)
                                traj2ds=traj2ds,  # (B, C, 3, H, W)
                            ),
                            return_dict=True,
                        ).action

                        # 1. 外推干净data X_0
                        tau = self.noise_scheduler.sigmas[self.noise_scheduler.index_for_timestep(timestep)].detach()

                        # DEBUG test inpainting
                        # noisy_prev_actions = (1 - tau) * prev_actions + tau * target_noise
                        # action_samples[:, :d, 14:28] = noisy_prev_actions[:, :d, 14:28]

                        # print("tau:", tau.item())
                        # print("timestep:", timestep)
                        # print("self.scheduler.sigmas:", self.scheduler.sigmas)
                        # print("self.scheduler.index_for_timestep(timestep):", self.scheduler.index_for_timestep(timestep))
                        pred_x0 = action_samples_grad - tau * model_pred
                        # pred_x0 = epsilon - model_pred
                        # print("pred_x0:", pred_x0)
                        

                        # 2. 计算(Y - pred_X_0) * weights
                        error = (prev_actions - pred_x0.detach()) * mask_expanded
                        # print("error:", error)
                        # print("error.abs().mean():", error.abs().mean())

                        # 3. vjp
                        pinv_correction = torch.autograd.grad(
                            outputs=pred_x0,  # [1,T,D]
                            inputs=action_samples_grad,        # [1,T,D]
                            grad_outputs=error,         # 匹配形状 [1,T,D]
                            retain_graph=False,
                            create_graph=False
                        )[0]  # 输出 [1,T,D]
                        # print("pinv_correction:", pinv_correction)

                    # 4. 计算tau相关参数，得到修正项，修正v
                    inv_r2 = (tau**2 + (1 - tau) ** 2) / (tau**2)
                    c = torch.nan_to_num(tau / (1 - tau), posinf=guidance_weight)
                    g = torch.clamp(c * inv_r2, max=guidance_weight)
                    # print("c * inv_r2:", c * inv_r2)
                    # print("model_pred_before_correction:", model_pred)

                    model_pred = model_pred - g * pinv_correction
                    # print("pinv_correction.abs().mean():", pinv_correction.abs().mean())
                    ###

                    action_samples = self.noise_scheduler.step(
                        model_output=model_pred, timestep=timestep, sample=action_samples # type: ignore
                    ).prev_sample

                    # DEBUG test inpainting
                    # if i == len(self.noise_scheduler.timesteps) - 1:
                    #     action_samples[:, :d, 14:28] = prev_actions[:, :d, 14:28]

            return action_samples


    def predict_action_with_rtc_flow_naive_inpaint(
        self,
        observations: List[List[Image.Image]],  # B * List of PIL Image as [view1, view2]
        states: torch.Tensor, # (B, Ts, Ds)
        instructions: List[str], # (B,)
        num_inference_steps: int,
        traj2ds, 
        prev_actions: torch.Tensor = None, # (1, H, D)
        inference_delay: int = 0,
        execution_horizon: int = 0,
        mask_schedule: str = "exponential",
        guidance_weight: float = 5.0,
        **kwargs: str
    ) -> np.ndarray:
        with torch.no_grad():

            ## RTC related ##
            H = self.action_horizon
            assert prev_actions is not None and inference_delay > 0 and execution_horizon > 0, "prev_actions, inference_delay and execution_horizon must be provided"
            assert prev_actions.shape[0] == 1
            prev_actions = prev_actions.to(device=self.device, dtype=torch.float32)

            # Create soft mask for inpainting
            d = inference_delay
            s = execution_horizon

            # Validate constraint from paper: d ≤ s ≤ H - d
            assert d <= s and s <= H - d, f"Constraint violated: d={d}, s={s}, H={H}. Need d ≤ s ≤ H-d"

            # Create soft mask [H] according to Equation 5
            mask = self._create_soft_mask(
                H, d, s, schedule=mask_schedule, device=self.device
            )
            mask_expanded = mask.view(1, -1, 1).expand(1, self.action_horizon, self.action_dim)
            ##              ##


            bsz = states.shape[0]
            batch_input_ids = []
            batch_attention_mask = []
            batch_pixel_values = []
            batch_image_grid_thw = []


            for observation, instruction in zip(observations, instructions):
                messages = []
                content = [{"type": "image", "image": img} for img in observation]
                content.append({"type": "text", "text": instruction})
                user_msg = {"role": "user", "content": content}
                messages.append([user_msg])
                texts = [
                    self.vlm_processor.apply_chat_template(
                        m, tokenize=False, add_generation_prompt=True
                    )
                    for m in messages
                ]
                image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
                inputs = self.vlm_processor(
                    text=texts,
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                # input_ids torch.Size([1, 80])
                # attention_mask torch.Size([1, 80])
                # pixel_values torch.Size([256, 1536])
                # image_grid_thw torch.Size([1, 3])
                batch_input_ids.append(inputs['input_ids'].squeeze(0))
                batch_attention_mask.append(inputs['attention_mask'].squeeze(0))
                batch_pixel_values.append(inputs['pixel_values'])
                batch_image_grid_thw.append(inputs['image_grid_thw'].squeeze(0))

            batch_input_ids = torch.stack(batch_input_ids) # (B, 80)
            batch_attention_mask = torch.stack(batch_attention_mask) # (B, 80)
            batch_pixel_values = torch.stack(batch_pixel_values) # (B, 256, 1536)
            batch_image_grid_thw = torch.stack(batch_image_grid_thw) # (B, 3)

            with torch.autocast(str(self.device).split(":")[0], dtype=torch.bfloat16):
                # extract vision + language features
                output = self.vlm_model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    pixel_values=batch_pixel_values,
                    image_grid_thw=batch_image_grid_thw,
                    output_hidden_states=True,
                    return_dict=True
                )
                vlm_hidden_states_ = output.hidden_states # len(vlm_hidden_states_) == 29
                

                # use hidden states from the last layer
                vlm_hidden_states = vlm_hidden_states_[-1] # shape (B, seq_len, D_h)  shape(16, 80, 2048)
                vlm_hidden_states = vlm_hidden_states.unsqueeze(1) # shape (B, 1, seq_len, D_h) (16, 1, 80, 2048)

                # generate action from noise
                action_samples = torch.randn(
                    bsz, self.action_horizon, self.action_dim, device=self.device
                )
                self.noise_scheduler.set_timesteps(num_inference_steps)

                # Fixed noise for consistency in inpainting
                # Use the same noise as action_samples to ensure the whole sequence starts from the same latent x_1
                target_noise = action_samples.clone()

                for i, timestep in enumerate(self.noise_scheduler.timesteps):
                    batched_timestep = timestep.expand(bsz).to(self.device).detach()

                    ### Naive inpainting (Repaint method):
                    # Step 1: Add noise to prev_actions matching current timestep's noise level
                    # This ensures consistency in noise levels across the entire action sequence
                    tau = self.noise_scheduler.sigmas[self.noise_scheduler.index_for_timestep(timestep)]
                    noisy_prev_actions = (1 - tau) * prev_actions + tau * target_noise
                    
                    # Step 2: Replace first d positions with noisy prev_actions (hard mask)
                    action_samples[:, :d, :] = noisy_prev_actions[:, :d, :]

                    # Step 3: Model prediction
                    model_pred = self.action_header(
                        hidden_states=None,
                        timestep=batched_timestep,
                        joint_attention_kwargs=dict(
                            action_hidden_embeds=action_samples, # (B,Tp,Da)
                            views=vlm_hidden_states,  # (B,V,N,D)
                            obs=states,  # (B,1,M)
                            traj2ds=traj2ds,  # (B, C, 3, H, W)
                        ),
                        return_dict=True,
                    ).action

                    # Step 4: Denoise one step (updates all positions including first d)
                    action_samples = self.noise_scheduler.step(
                        model_output=model_pred, timestep=timestep, sample=action_samples # type: ignore
                    ).prev_sample
                    
                    # Step 5: Re-inpaint with noise level matching the denoised state
                    # TODO: do this or not??? boqian
                    if i == len(self.noise_scheduler.timesteps) - 1:
                        # Last step: use clean prev_actions (tau = 0)
                        action_samples[:, :d, :] = prev_actions[:, :d, :]

            return action_samples



    def predict_action_with_rtc_flow_based_gradient(
        self,
        observations: List[List[Image.Image]],  # B * List of PIL Image as [view1, view2]
        states: torch.Tensor, # (B, Ts, Ds)
        instructions: List[str], # (B,)
        num_inference_steps: int,
        traj2ds, 
        prev_actions: torch.Tensor | None = None, # (1, H, D)
        inference_delay: int = 0,
        execution_horizon: int = 0,
        mask_schedule: str = "exponential",
        guidance_weight: float = 5.0,
        **kwargs: str
    ) -> np.ndarray:
        with torch.no_grad():

            ## RTC related ##
            H = self.action_horizon
            assert prev_actions is not None and inference_delay > 0 and execution_horizon > 0, "prev_actions, inference_delay and execution_horizon must be provided"
            assert prev_actions.shape[0] == 1
            prev_actions = prev_actions.to(device=self.device, dtype=torch.float32)

            # Create soft mask for inpainting
            d = inference_delay
            s = execution_horizon

            # Validate constraint from paper: d ≤ s ≤ H - d
            assert d <= s and s <= H - d, f"Constraint violated: d={d}, s={s}, H={H}. Need d ≤ s ≤ H-d"

            # Create soft mask [H] according to Equation 5
            mask = self._create_soft_mask(
                H, d, s, schedule=mask_schedule, device=self.device
            )
            mask_expanded = mask.view(1, -1, 1).expand(1, self.action_horizon, self.action_dim)
            ##              ##


            bsz = states.shape[0]
            batch_input_ids = []
            batch_attention_mask = []
            batch_pixel_values = []
            batch_image_grid_thw = []


            for observation, instruction in zip(observations, instructions):
                messages = []
                content = [{"type": "image", "image": img} for img in observation]
                content.append({"type": "text", "text": instruction})
                user_msg = {"role": "user", "content": content}
                messages.append([user_msg])
                texts = [
                    self.vlm_processor.apply_chat_template(
                        m, tokenize=False, add_generation_prompt=True
                    )
                    for m in messages
                ]
                image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
                inputs = self.vlm_processor(
                    text=texts,
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                # input_ids torch.Size([1, 80])
                # attention_mask torch.Size([1, 80])
                # pixel_values torch.Size([256, 1536])
                # image_grid_thw torch.Size([1, 3])
                batch_input_ids.append(inputs['input_ids'].squeeze(0))
                batch_attention_mask.append(inputs['attention_mask'].squeeze(0))
                batch_pixel_values.append(inputs['pixel_values'])
                batch_image_grid_thw.append(inputs['image_grid_thw'].squeeze(0))

            batch_input_ids = torch.stack(batch_input_ids) # (B, 80)
            batch_attention_mask = torch.stack(batch_attention_mask) # (B, 80)
            batch_pixel_values = torch.stack(batch_pixel_values) # (B, 256, 1536)
            batch_image_grid_thw = torch.stack(batch_image_grid_thw) # (B, 3)

            with torch.autocast(str(self.device).split(":")[0], dtype=torch.bfloat16):
                # extract vision + language features
                output = self.vlm_model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    pixel_values=batch_pixel_values,
                    image_grid_thw=batch_image_grid_thw,
                    output_hidden_states=True,
                    return_dict=True
                )
                vlm_hidden_states_ = output.hidden_states # len(vlm_hidden_states_) == 29
                

                # use hidden states from the last layer
                vlm_hidden_states = vlm_hidden_states_[-1] # shape (B, seq_len, D_h)  shape(16, 80, 2048)
                vlm_hidden_states = vlm_hidden_states.unsqueeze(1) # shape (B, 1, seq_len, D_h) (16, 1, 80, 2048)

                # 初始化 action_samples
                action_samples = torch.randn(
                    bsz, self.action_horizon, self.action_dim, device=self.device
                )
                
                # 使用同样的 noise 构建目标，保证一致性
                target_noise = action_samples.clone()
                
                self.noise_scheduler.set_timesteps(num_inference_steps)

                for i, timestep in enumerate(self.noise_scheduler.timesteps):
                    # 1. 计算当前步的目标 (Noisy Target)
                    # ---------------------------------------------------------
                    tau = self.noise_scheduler.sigmas[self.noise_scheduler.index_for_timestep(timestep)]
                    
                    # 这是一个移动的目标：随着 tau 变小，它越来越接近真实的 prev_actions
                    noisy_prev_actions = (1 - tau) * prev_actions + tau * target_noise
                    
                    # 2. 基于梯度的 Guidance (核心部分)
                    # ---------------------------------------------------------
                    # 我们希望 action_samples 在 mask 区域尽可能接近 noisy_prev_actions
                    # 通过梯度下降来修改 action_samples
                    
                    # 定义 Guidance 的强度衰减：
                    # 早期(噪声大)需要强引导来定型，晚期(噪声小)减弱引导以允许模型平滑过渡
                    current_scale = guidance_weight * (tau ** 0.5) # 这里的衰减策略可以调整，例如用 linear

                    # 临时开启梯度计算
                    with torch.enable_grad():
                        # Detach 并开启梯度，创建一个新的叶子节点
                        x_in = action_samples.detach().requires_grad_(True)
                        
                        # 计算 Loss: 只在 mask 有值的地方计算 MSE
                        # 这里的 mask_expanded 充当了权重的角色
                        # 重点：只计算前 s 步的差异 (inference_delay + execution_horizon)
                        diff = (x_in - noisy_prev_actions)
                        
                        # 使用 mask 加权平方误差
                        # 形状: (B, H, D) * (1, H, D)
                        loss = (diff ** 2 * mask_expanded).sum() 
                        
                        # 计算梯度
                        grad = torch.autograd.grad(loss, x_in)[0]

                    # 更新 action_samples (朝着 loss 减小的方向)
                    # 注意：这里我们是在 Latent 空间进行微调
                    print("grad[:, :, 0:3].abs().mean():", grad[:, :, 0:3].abs().mean())
                    print("grad[:, :, 15:17].abs().mean():", grad[:, :, 15:17].abs().mean())
                    

                    action_samples = action_samples - current_scale * grad

                    # 3. 常规的 Flow Matching / Diffusion Step
                    # ---------------------------------------------------------
                    batched_timestep = timestep.expand(bsz).to(self.device).detach()
                    
                    model_pred = self.action_header(
                        hidden_states=None,
                        timestep=batched_timestep,
                        joint_attention_kwargs=dict(
                            action_hidden_embeds=action_samples, 
                            views=vlm_hidden_states,
                            obs=states,
                            traj2ds=traj2ds,
                        ),
                        return_dict=True,
                    ).action

                    # 4. Denoise one step
                    action_samples = self.noise_scheduler.step(
                        model_output=model_pred, timestep=timestep, sample=action_samples
                    ).prev_sample

                    # 5. (可选) Soft Blending 作为最后的保险
                    # 如果单纯靠梯度还是压不住，可以在最后一步稍微融合一下
                    # 但如果 guidance_weight 足够，通常不需要这步
                    # if i == len(self.noise_scheduler.timesteps) - 1:
                    #         # 最后一步不再有噪声，直接做一次软混合确保开头完全吻合
                    #         action_samples = prev_actions * mask_expanded + action_samples * (1 - mask_expanded)

            return action_samples




    def _create_soft_mask(self, H, d, s, schedule="exponential", device="cpu"):
        """
        Create soft mask for RTC inpainting (Equation 5 in paper).
        
        Paper: "Real-Time Execution of Action Chunking Flow Policies"
        Figure 3 and Equation 5:
        
        W_i = { 1                           if i < d (frozen region)
              { c_i * (e^(c_i) - 1)/(e - 1) if d ≤ i < H - s (intermediate region)
              { 0                           if i ≥ H - s (free region)
        
        where c_i = (H - s - i)/(H - s - d + 1)
        
        Args:
            H: Prediction horizon (total sequence length)
            d: Inference delay (number of frozen steps, already executed)
            s: Execution horizon (non-overlapping steps at end). 
            schedule: 'exponential' (paper default), 'linear', 'hard', or 'simple'
        
        Returns:
            mask: [H] tensor with values in [0, 1]
                  Guidance weights for each timestep
        """
        mask = torch.zeros(H, device=device)
        
        if schedule == "hard":
            # Hard mask: 1 for frozen, 0 for rest
            mask[:d] = 1.0
        
        elif schedule == "linear":
            # Linear decay
            mask[:d] = 1.0
            # Three-region version (full paper)
            overlap_end = H - s
            if d < overlap_end:
                # Linear decay from 1 to 0 in intermediate region
                indices = torch.arange(d, overlap_end, device=device).float()
                mask[d:overlap_end] = 1.0 - (indices - d) / (overlap_end - d)
            # mask[overlap_end:] remains 0
        
        elif schedule == "exponential":
            # Exponential decay (paper default, Equation 5)
            mask[:d] = 1.0
            
            # Three-region version (full paper implementation)
            overlap_end = H - s
            if d < overlap_end:
                indices = torch.arange(d, overlap_end, device=device).float()
                # c_i = (H - s - i) / (H - s - d + 1)
                c_i = (overlap_end - indices) / (overlap_end - d + 1)
                # W_i = c_i * (e^(c_i) - 1) / (e - 1)
                e = torch.tensor(torch.e, device=device)
                mask[d:overlap_end] = c_i * (torch.exp(c_i) - 1) / (e - 1)
            # mask[overlap_end:] remains 0
        
        elif schedule == "simple":
            # Simplified exponential (for backward compatibility)
            mask[:d] = 1.0
            if d < H:
                indices = torch.arange(d, H, device=device).float()
                mask[d:] = torch.exp(-5.0 * (indices - d) / (H - d))
        
        else:
            raise ValueError(f"Unknown mask schedule: {schedule}")
        
        return mask
    
    @torch.inference_mode()
    def chat(self):
        ...
    

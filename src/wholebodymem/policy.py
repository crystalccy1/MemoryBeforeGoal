"""Load a WholeBodyMem checkpoint bundle: the saved training configuration, the Qwen3-VL backbone and the action expert.

A bundle is a directory with
    model.safetensors     the full policy (Qwen3-VL backbone + action expert with visual and object memory)
    run_config.json       the training configuration (normalisation statistics included)
    vlm/                  Qwen3-VL configuration, tokenizer and processor files of the Psi0 backbone
    BUNDLE.json           task name, training step and the sha256 of model.safetensors

``Psi0KeyframeNativeObjectEvalModel`` is the inference subclass of the training model ``Psi0KeyframeNativeObjectModel``.
It packs the visual memory (7 frames x 77 tokens) and the object memory (7 frames x 2 slots) once per request, exactly as the
training forward does, and runs the unchanged Psi0 flow-matching decoders (plain and real-time chunking) with that packed
memory.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import hashlib
import json
import os
from pathlib import Path
import threading

import torch

from psi.models.psi0 import Psi0Model
from psi.models.psi0_framesamp import MEMORY_FIELD_NAMES, Psi0FrameSampCondition
from psi.models.psi0_keyframe_native_object import (
    GATE_CONTRACT, OBJECT_INPUTS, VISUAL_BUDGET, Psi0KeyframeNativeObjectModel,
)

OBJECT_TRAIN_NAME = "finetune_keyframe_native_object"
PACK_PRECISIONS = ("fp32", "bf16_autocast")   # memory encoders without autocast (paper evaluation) or under bf16 autocast
DEFAULT_PACK_PRECISION = "fp32"
PACK_FIELDS = MEMORY_FIELD_NAMES + OBJECT_INPUTS
# dtypes of the training collators
PACK_DTYPES = {
    "memory_features": torch.bfloat16, "memory_mask": torch.bool,
    "memory_raw_frame_indices": torch.int64, "memory_spatial_indices": torch.int64,
    "memory_object_features": torch.bfloat16, "memory_object_geometry": torch.float32,
    "memory_object_mask": torch.bool, "memory_object_raw_frame_indices": torch.int64,
    "memory_object_slots": torch.int64,
}
# Named parameters of Psi0Model.predict_action_with_training_rtc_flow; its trailing **kwargs would silently drop anything
# else, so anything else is refused.
TRAINING_RTC_ARGUMENTS = frozenset({
    "observations", "states", "instructions", "num_inference_steps", "traj2ds",
    "prev_actions", "inference_delay", "max_delay",
})

_parse_lock = threading.Lock()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def training_schema_environment():
    """The saved training schema insists on W&B logging; satisfy that guard during validation only.

    No W&B client is initialised, and the environment is restored even when validation raises.
    """
    with _parse_lock:
        saved = {name: os.environ.get(name) for name in ("WANDB_MODE", "WANDB_DISABLED")}
        os.environ["WANDB_MODE"] = "online"
        os.environ["WANDB_DISABLED"] = "false"
        try:
            yield
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def load_run_config(bundle: str | Path):
    """Validate ``run_config.json`` with the training schema; normalisation statistics come from the file itself."""
    from psi.config.train.finetune_simple_psi0_keyframe_native_object_config import DynamicLaunchConfig

    text = (Path(bundle) / "run_config.json").read_text()
    raw = json.loads(text)
    if raw.get("train", {}).get("name") != OBJECT_TRAIN_NAME:
        raise ValueError(f"run_config.json is not a {OBJECT_TRAIN_NAME} run")
    with training_schema_environment():
        config = DynamicLaunchConfig.model_validate_json(text)
    model = config.model
    if (model.object_memory_enabled is not True or model.object_memory_slots != 2
            or model.native_keyframe_selector != "sam3"):
        raise ValueError("WholeBodyMem inference requires object memory, 2 slots and the SAM3 keyframe selector")
    # ActionStateTransform re-reads its statistics from ``stat_path`` when such a file is reachable (e.g. relative to the
    # working directory); the checkpoint must use the statistics saved with it.
    field, saved = config.data.transform.field, raw["data"]["transform"]["field"]
    for name in ("action_min", "action_max", "state_min", "state_max"):
        if getattr(field, name) != saved.get(name):
            raise RuntimeError(f"normalisation {name} differs from run_config.json (a {field.stat_path!r} file was "
                               "picked up); start the server from another working directory")
    return config


class Psi0KeyframeNativeObjectEvalModel(Psi0KeyframeNativeObjectModel):
    """Training model plus packed-memory inference."""

    def __init__(self, model_cfg, vlm_model):
        super().__init__(model_cfg, vlm_model)
        gate_init = getattr(model_cfg, "object_memory_gate_init", "zero")
        if gate_init not in GATE_CONTRACT:
            raise ValueError(f"object_memory_gate_init={gate_init!r} is not one of {sorted(GATE_CONTRACT)}")
        header = self.action_header
        if (int(model_cfg.object_memory_slots) != 2 or header.object_budget != 14
                or int(model_cfg.framesamp_memory_budget) != VISUAL_BUDGET):
            raise ValueError("WholeBodyMem inference requires 539 visual + 14 object memory tokens")
        self._pack_precision = DEFAULT_PACK_PRECISION

    @property
    def pack_precision(self) -> str:
        return self._pack_precision

    @pack_precision.setter
    def pack_precision(self, value: str) -> None:
        if value not in PACK_PRECISIONS:
            raise ValueError(f"pack_precision must be one of {PACK_PRECISIONS}, got {value!r}")
        self._pack_precision = value

    def pack_object_memory(self, **memory) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode visual (539) and object (14) memory and concatenate them, as the training forward does."""
        if set(memory) != set(PACK_FIELDS):
            raise KeyError(f"Object-memory packing needs exactly {PACK_FIELDS}; "
                           f"missing={sorted(set(PACK_FIELDS) - set(memory))} "
                           f"unexpected={sorted(set(memory) - set(PACK_FIELDS))}")
        for name, value in memory.items():
            if not isinstance(value, torch.Tensor) or value.dtype != PACK_DTYPES[name]:
                raise TypeError(f"{name} must be a {PACK_DTYPES[name]} tensor")
        x = {name: value.to(self.device) for name, value in memory.items()}
        visual = self._memory_inputs(**{name: x[name] for name in MEMORY_FIELD_NAMES})
        precision = (torch.autocast(str(self.device).split(":")[0], dtype=torch.bfloat16)
                     if self._pack_precision == "bf16_autocast" else nullcontext())
        with torch.no_grad(), precision:
            hidden = self.action_header.encode_memory(**visual)
            objects = self.action_header.encode_object_memory(**{name: x[name] for name in OBJECT_INPUTS})
            packed = torch.cat([hidden, objects.to(hidden.dtype)], 1)
        mask = torch.cat([x["memory_mask"], x["memory_object_mask"]], 1)
        if (packed.shape[1:] != (VISUAL_BUDGET + self.action_header.object_budget, self.action_header.hidden_size)
                or mask.dtype != torch.bool or mask.shape != packed.shape[:2]):
            raise ValueError("Packed visual/object memory has an invalid shape")
        if not torch.isfinite(packed).all():
            raise ValueError("Packed visual/object memory is not finite")
        return packed.detach().clone(), mask.detach().clone()

    def encode_condition(self, observations, states, instructions, traj2ds=None, **memory) -> Psi0FrameSampCondition:
        """Psi0 condition (views, state, language) with the 553-token visual + object memory."""
        if self.framesamp_model_cfg.framesamp_entry_probe_only:
            raise RuntimeError("entry-probe model cannot perform inference")
        base = Psi0Model.encode_condition(self, observations=observations, states=states,
                                          instructions=instructions, traj2ds=traj2ds)
        packed, mask = self.pack_object_memory(**memory)
        normal = lambda value: None if value is None else value.detach().clone()
        return Psi0FrameSampCondition(
            views=normal(base.views), states=normal(base.states), traj2ds=normal(base.traj2ds),
            attention_mask=normal(base.attention_mask), vision_segment_ids=normal(base.vision_segment_ids),
            memory_hidden_states=packed, memory_mask=mask)

    def _predict_velocity_from_condition(self, condition, action_samples, timestep):
        if not isinstance(condition, Psi0FrameSampCondition):
            raise TypeError("Object-memory decoding requires Psi0FrameSampCondition")
        with self.action_header.use_preprojected_memory(condition.memory_hidden_states, condition.memory_mask):
            return Psi0Model._predict_velocity_from_condition(self, condition, action_samples, timestep)

    def predict_action_with_training_rtc_flow(self, *args, **kwargs):
        """Pack the nine memory fields, then run Psi0's real-time-chunking decoder with that memory."""
        memory = {name: kwargs.pop(name) for name in PACK_FIELDS if name in kwargs}
        unexpected = sorted(set(kwargs) - TRAINING_RTC_ARGUMENTS)
        if unexpected:
            raise TypeError(f"Unexpected training-RTC arguments: {unexpected}")
        packed, mask = self.pack_object_memory(**memory)
        with self.action_header.use_preprojected_memory(packed, mask):
            return Psi0Model.predict_action_with_training_rtc_flow(self, *args, **kwargs)

    def _run_direct_rtc(self, method_name, args, kwargs):
        raise NotImplementedError("only the plain and the training-RTC decoders are supported")


def load_policy(bundle: str | Path, config, device: str | torch.device, *, check_sha256: bool = True,
                pack_precision: str = DEFAULT_PACK_PRECISION) -> Psi0KeyframeNativeObjectEvalModel:
    """Build the model from the bundle (the restore steps of ``Psi0FrameSampModel.from_pretrained``, reading the
    backbone configuration from ``vlm/`` instead of the training machine's paths) and load the weights strictly."""
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration

    bundle = Path(bundle)
    model_path = bundle / "model.safetensors"
    vlm_dir = bundle / "vlm"
    if not os.path.isfile(vlm_dir / "config.json"):
        raise FileNotFoundError(f"{vlm_dir} must hold the Qwen3-VL config and processor files")
    if check_sha256:
        meta = json.loads((bundle / "BUNDLE.json").read_text())
        if sha256_file(model_path) != meta["model_sha256"]:
            raise RuntimeError(f"{model_path} does not match the sha256 in BUNDLE.json")
    if config.model.framesamp_entry_probe_only:
        raise RuntimeError("Entry-probe checkpoints are not deployable")

    state_dict = load_file(str(model_path), device="cpu")
    vlm_state_dict, action_state_dict = {}, {}
    for key, value in state_dict.items():
        if key.startswith("vlm_model."):
            vlm_state_dict[key.removeprefix("vlm_model.")] = value
        elif key.startswith("action_header."):
            action_state_dict[key.removeprefix("action_header.")] = value
        else:
            raise RuntimeError(f"Unexpected top-level checkpoint key: {key}")
    vlm_config = AutoConfig.from_pretrained(vlm_dir)
    # The memory features were extracted with SDPA; keep the policy and the online frame encoder on that backend.
    vlm_config._attn_implementation = "sdpa"
    vlm_config.dtype = torch.bfloat16
    vlm_config.vision_config.dtype = torch.bfloat16
    vlm_config.text_config.dtype = torch.bfloat16
    vlm_model = Qwen3VLForConditionalGeneration(vlm_config).to(dtype=torch.bfloat16)
    embedding_key = "model.language_model.embed_tokens.weight"
    if embedding_key not in vlm_state_dict:
        raise RuntimeError("Checkpoint is missing the Qwen embedding table")
    if vlm_state_dict[embedding_key].shape[0] != vlm_model.lm_head.weight.shape[0]:
        vlm_model.resize_token_embeddings(vlm_state_dict[embedding_key].shape[0], pad_to_multiple_of=192,
                                          mean_resizing=True)
    if "lm_head.weight" in vlm_state_dict and not torch.equal(vlm_state_dict["lm_head.weight"],
                                                               vlm_state_dict[embedding_key]):
        raise RuntimeError("Checkpoint contains inconsistent tied Qwen embeddings")
    vlm_state_dict["lm_head.weight"] = vlm_state_dict[embedding_key]
    vlm_model.load_state_dict(vlm_state_dict, strict=True)
    vlm_model.tie_weights()

    model = Psi0KeyframeNativeObjectEvalModel(model_cfg=config.model, vlm_model=vlm_model)
    version = action_state_dict.get("framesamp_contract_version")
    if version is None or version.numel() != 1 or int(version.item()) != 2:
        raise RuntimeError("Checkpoint is not a FrameSamp v2 memory checkpoint")
    memory_keys = set(model.action_header.memory_parameter_names())
    if not memory_keys.issubset(action_state_dict):
        raise RuntimeError(f"Checkpoint lacks memory parameters: {sorted(memory_keys - set(action_state_dict))}")
    model.action_header.load_state_dict(action_state_dict, strict=True)
    model.vlm_processor = AutoProcessor.from_pretrained(vlm_dir)

    if config.model.noise_scheduler == "flow":
        from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=config.model.train_diffusion_steps)
    elif config.model.noise_scheduler == "ddpm":
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler
        scheduler = DDIMScheduler(num_train_timesteps=config.model.train_diffusion_steps, beta_start=0.0001,
                                  beta_end=0.02, beta_schedule="squaredcos_cap_v2", clip_sample=True,
                                  set_alpha_to_one=True, steps_offset=0, prediction_type="epsilon")
    else:
        raise ValueError(f"Unsupported noise scheduler: {config.model.noise_scheduler}")
    model.noise_scheduler = scheduler
    model.action_horizon = config.model.action_chunk_size
    model.action_dim = config.model.action_dim
    model.device = torch.device(device)
    model.to(model.device)
    model.eval()
    model.framesamp_model_cfg = config.model

    # The checkpoint's gate contract must name the gate the configuration was trained with.
    header = model.action_header
    contract_version = int(header.object_memory_contract_version.item())
    modes = [mode for mode, version in GATE_CONTRACT.items() if version == contract_version]
    if modes != [getattr(config.model, "object_memory_gate_init", "zero")]:
        raise RuntimeError("Checkpoint gate contract differs from the saved object_memory_gate_init")
    if any(getattr(block.memory_attention, "gate_init", "zero") != modes[0] for block in header.transformer_blocks):
        raise RuntimeError("Built object gate modules differ from the checkpoint gate contract")
    model.pack_precision = pack_precision
    return model

"""Native 7x77 keyframes selected by SAM3 events, plus gated SAM3 object memory."""
from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import model_validator

from psi.config.model_psi0_keyframe_native import Psi0KeyframeNativeModelConfig


class Psi0KeyframeNativeObjectModelConfig(Psi0KeyframeNativeModelConfig):
    native_keyframe_selector: Literal["sam3"] = "sam3"
    object_memory_enabled: Literal[True] = True
    object_memory_contract_version: Literal[1] = 1
    object_memory_root: str
    object_memory_manifest_sha256: str
    object_memory_slots: Literal[2] = 2
    object_memory_geometry_dim: Literal[8] = 8
    object_memory_gate_init: Literal["zero", "zero_out_proj"] = "zero_out_proj"
    # Decision head (off by default: then the model and state_dict are exactly the object-memory model).
    decision_head_enabled: bool = False
    decision_head_inputs: Literal["objects", "visual", "both"] = "both"
    decision_num_classes: int = 3
    decision_loss_weight: float = 0.5
    # Teacher forcing of the decision input (0 = off = the decision-head model above): during training, at frames with a
    # decision label, the action header gets the one-hot answer instead of softmax(logits) with this probability, so the
    # action expert learns to follow the decision input. Inference always uses the head's own softmax.
    decision_teacher_forcing: float = 0.0
    # Warm start (training only): before training, load the whole model from a checkpoint of this same architecture
    # (e.g. a B+ ckpt_40000); the frozen VLM lm_head (tied, not saved) keeps the base weights. Inference is unaffected.
    warm_start_checkpoint: Optional[str] = None

    @model_validator(mode="after")
    def validate_object_memory(self):
        if not re.fullmatch(r"[0-9a-f]{64}", self.object_memory_manifest_sha256):
            raise ValueError("object_memory_manifest_sha256 must be a lowercase SHA-256")
        if not self.object_memory_root:
            raise ValueError("object_memory_root is required")
        if not 2 <= self.decision_num_classes <= 8 or self.decision_loss_weight < 0:
            raise ValueError("decision head needs 2..8 classes and a nonnegative loss weight")
        if not 0.0 <= self.decision_teacher_forcing <= 1.0:
            raise ValueError("decision_teacher_forcing must be in [0, 1]")
        if self.decision_teacher_forcing > 0 and not self.decision_head_enabled:
            raise ValueError("decision_teacher_forcing needs the decision head")
        return self

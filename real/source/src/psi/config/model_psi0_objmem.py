"""Cart Psi0 model config plus the SAM3-v2 memory (native 7x77 keyframes + prompt-slotted SAM3 objects)."""
from __future__ import annotations

import re
from typing import Literal

from pydantic import model_validator

from psi.config.model_psi0 import Psi0ModelConfig


class Psi0ObjMemModelConfig(Psi0ModelConfig):
    objmem_enabled: Literal[True] = True
    objmem_bank_root: str
    objmem_bank_manifest_sha256: str
    memory_position_dim: int = 768
    memory_heads: int = 6
    memory_head_dim: int = 256
    memory_modulation_init: Literal["robomme_normal_std_0_002"] = "robomme_normal_std_0_002"
    object_slots: int = 4
    object_gate_init: Literal["zero_out_proj"] = "zero_out_proj"

    @model_validator(mode="after")
    def validate_objmem(self):
        if not re.fullmatch(r"[0-9a-f]{64}", self.objmem_bank_manifest_sha256):
            raise ValueError("objmem_bank_manifest_sha256 must be a lowercase SHA-256")
        if self.hidden_dim != self.memory_heads * self.memory_head_dim:
            raise ValueError("hidden_dim must equal memory_heads * memory_head_dim")
        if self.memory_position_dim % 6 or self.memory_head_dim % 2:
            raise ValueError("memory position dim must divide by 6 and head dim must be even")
        if self.tune_vlm or self.use_dit:
            raise ValueError("SAM3-v2 memory requires a frozen VLM and the ActionTransformerModel header")
        if not 1 <= self.object_slots <= 8:
            raise ValueError("object_slots must be in [1, 8]")
        return self

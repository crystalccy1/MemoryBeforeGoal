"""Psi0 flow matching with SAM3-selected native keyframes and gated SAM3 object memory.

With model.decision_head_enabled the loss adds decision_loss_weight x cross-entropy of the decision head's logits against
batch["decision_label"] (episode answer, from the recorded decision step on; -100 = not supervised)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from psi.data.lerobot.sam3_object_memory import OBJECT_FIELDS, PaddedCollatorForNativeKeyframeObject
from psi.models.psi0_keyframe_native_object import Psi0KeyframeNativeObjectModel
from psi.trainers.finetune import FinetuneTrainer, overwatch
from psi.trainers.finetune_keyframe_native import FinetuneKeyframeNativeTrainer

VISUAL_FIELDS = ("memory_features", "memory_mask", "memory_raw_frame_indices", "memory_spatial_indices")


def _memory_key(key: str) -> bool:
    return key.startswith(("memory_encoder.", "object_memory_encoder.", "framesamp_contract_version",
                           "keyframe_native_contract_version", "object_memory_contract_version",
                           "decision_head.", "decision_embed.", "decision_head_contract_version")) or ".memory_" in key


class FinetuneKeyframeNativeObjectTrainer(FinetuneKeyframeNativeTrainer):
    MODEL_CLASS = Psi0KeyframeNativeObjectModel

    def _validate_strict_resume_lineage(self):
        super()._validate_strict_resume_lineage()
        if self.train_cfg.resume_from_checkpoint is None:
            return
        checkpoint = Path(self.train_cfg.resume_from_checkpoint).resolve(strict=True)
        run = checkpoint.parents[1] if checkpoint.name.startswith("ckpt_") else checkpoint
        source = json.loads((run / "run_config.json").read_text())
        for section in ("model", "data"):
            for field in ("object_memory_root", "object_memory_manifest_sha256"):
                if source.get(section, {}).get(field) != getattr(getattr(self.cfg, section), field):
                    raise ValueError(f"Object-memory resume lineage changed: {section}.{field}")

    def _load_clean_action_header(self, state):
        if any(_memory_key(key) for key in state):
            raise ValueError("Object-memory warm start requires the common clean Psi0 checkpoint")
        if not {"action_proj_out.linear.weight", "action_proj_in.dec_pos"}.issubset(state):
            raise ValueError("Warm start is missing the clean Psi0 action shape metadata")
        target = self.model.action_header.state_dict()
        legacy = {key for key in target if not _memory_key(key)}
        if state["action_proj_out.linear.weight"].shape[0] != self.model_cfg.action_dim:
            raise ValueError("Formal warm start requires the original action dimension")
        mismatch = state["action_proj_in.dec_pos"].shape[0] != self.model_cfg.action_chunk_size
        selected = {k: v for k, v in state.items() if k.startswith("transformer_blocks.")} if mismatch else state
        expected = {k for k in legacy if k.startswith("transformer_blocks.")} if mismatch else legacy
        if set(selected) != expected:
            raise ValueError("Warm start did not match the exact common Psi0 parameter subset")
        self.model.action_header.load_state_dict({**target, **selected}, strict=True)
        overwatch.info(f"Loaded {len(selected)} common Psi0 tensors; initialized native + object memory")

    def init_models(self):
        super().init_models()
        path = getattr(self.model_cfg, "warm_start_checkpoint", None)
        if not path:
            return
        from safetensors.torch import load_file
        source = Path(path).resolve(strict=True)
        source = source / "model.safetensors" if source.is_dir() else source
        state = load_file(str(source), device="cpu")
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        # the saved model omits the frozen VLM's lm_head (tied, never trained): it keeps the common base weights
        if unexpected or set(missing) - {"vlm_model.lm_head.weight"}:
            raise ValueError(f"Warm start must come from the same architecture: missing={sorted(missing)[:8]} "
                             f"unexpected={sorted(unexpected)[:8]}")
        print("WARM_START_JSON=" + json.dumps({"source": str(source), "loaded": len(state), "missing": sorted(missing)},
                                              sort_keys=True), flush=True)

    @staticmethod
    def _memory_kwargs(batch):
        names = VISUAL_FIELDS + OBJECT_FIELDS
        missing = [name for name in names if name not in batch]
        if missing:
            raise KeyError(f"Object-memory training batch is missing {missing}")
        return {name: batch[name] for name in names}

    def create_dataloaders(self, train_dataset, val_dataset):
        loaders = FinetuneTrainer.create_dataloaders(self, train_dataset, val_dataset)
        collator = PaddedCollatorForNativeKeyframeObject(
            model_max_length=self.tokenizer.model_max_length, pad_token_id=self.tokenizer.pad_token_id,
            padding_side="right", memory_budget=539, feature_dim=2048)
        for loader in loaders:
            if loader is not None:
                if loader._iterator is not None:
                    raise RuntimeError("Object-memory collator must be installed before iteration")
                loader.collate_fn = collator
        return loaders

    def forward_and_loss(self, model, batch):
        header = self.unwrap_model().action_header
        teacher = float(getattr(self.model_cfg, "decision_teacher_forcing", 0.0) or 0.0)
        if teacher > 0 and "decision_label" not in batch:
            raise RuntimeError("Decision teacher forcing needs decision labels in the batch")
        header._decision_teacher_labels = batch["decision_label"] if teacher > 0 else None
        try:
            losses = super().forward_and_loss(model, batch)
        finally:
            header._decision_teacher_labels = None
        if not getattr(self.model_cfg, "decision_head_enabled", False):
            return losses
        logits = self.unwrap_model().decision_logits()
        if logits is None or "decision_label" not in batch:
            raise RuntimeError("Decision head enabled but no logits were produced or the batch has no decision labels")
        labels = batch["decision_label"].to(logits.device)
        valid = labels >= 0
        if bool(valid.any()):
            ce = F.cross_entropy(logits.float()[valid], labels[valid])
            accuracy = float((logits.argmax(-1)[valid] == labels[valid]).float().mean())
        else:
            ce, accuracy = logits.float().sum() * 0.0, float("nan")
        self._decision_stats = {"decision/ce": float(ce.detach()), "decision/acc": accuracy,
                                "decision/valid_fraction": float(valid.float().mean()),
                                "decision/action_loss": float(losses["loss"].detach())}
        forced = getattr(self.unwrap_model().action_header, "last_decision_forced_fraction", None)
        if forced is not None:
            self._decision_stats["decision/teacher_forced_fraction"] = forced
        losses["loss"] = losses["loss"] + float(self.model_cfg.decision_loss_weight) * ce
        if not getattr(self, "_decision_first_printed", False):
            self._decision_first_printed = True
            print("DECISION_HEAD_FIRST_LOSS_JSON=" + json.dumps({
                "rank": self.accelerator.process_index if hasattr(self, "accelerator") else None,
                "inputs": self.model_cfg.decision_head_inputs, "classes": self.model_cfg.decision_num_classes,
                "weight": self.model_cfg.decision_loss_weight, **self._decision_stats}, sort_keys=True), flush=True)
        return losses

    def training_step(self, batch):
        sync, metrics = super().training_step(batch)
        if getattr(self.model_cfg, "decision_head_enabled", False):
            stats = getattr(self, "_decision_stats", {})
            metrics.update({k: v for k, v in stats.items() if not (isinstance(v, float) and math.isnan(v))})
        model = self.unwrap_model()
        gates = model.object_gates()
        metrics["object_memory/valid_tokens"] = batch["memory_object_mask"].sum(-1).float().mean().item()
        metrics["object_memory/gate_tanh_mean"] = sum(gates) / len(gates)
        metrics["object_memory/gate_tanh_absmax"] = max(abs(g) for g in gates)
        norms = model.object_out_proj_norms()
        metrics["object_memory/out_proj_norm_mean"] = sum(norms) / len(norms)
        if sync and self._native_updates == 1:
            if not model._object_gate_grad_seen:
                raise RuntimeError("Object-memory branch received no finite gradient on the first update")
            print("OBJECT_MEMORY_FIRST_UPDATE_PASSED_JSON=" + json.dumps({
                "rank": self.accelerator.process_index, "gates": gates,
                "valid_object_tokens": batch["memory_object_mask"].sum(-1).tolist(),
                "object_manifest_sha256": self.data_cfg.object_memory_manifest_sha256}, sort_keys=True), flush=True)
        return sync, metrics

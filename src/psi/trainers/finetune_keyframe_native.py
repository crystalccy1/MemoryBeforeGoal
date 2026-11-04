"""Psi0 flow matching with seven native 7x11 visual keyframes."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from psi.config.train.finetune_simple_psi0_gist_config import require_wandb
from psi.data.lerobot.keyframe_native_training import PaddedCollatorForNativeKeyframe
from psi.models.psi0_keyframe_native import Psi0KeyframeNativeModel
from psi.trainers.finetune import FinetuneTrainer, overwatch
from psi.trainers.finetune_framesamp import FinetuneFramesampTrainer


class FinetuneKeyframeNativeTrainer(FinetuneFramesampTrainer):
    MODEL_CLASS = Psi0KeyframeNativeModel

    def __init__(self, cfg, device):
        require_wandb(cfg.log.report_to)
        # The legacy constructor assumes a 16-token memory feature bank.
        FinetuneTrainer.__init__(self, cfg, device)
        self._validate_strict_resume_lineage()
        bank = self.data_cfg.feature_bank()
        if Path(self.model_cfg.model_name_or_path).resolve(strict=True) != Path(
                bank.contract["encoder"]["checkpoint_dir"]).resolve(strict=True):
            raise ValueError("Native memory and current observation require the same Qwen checkpoint")
        self._native_updates = 0

    def _validate_strict_resume_lineage(self):
        if self.train_cfg.resume_from_checkpoint is None:
            return
        checkpoint = Path(self.train_cfg.resume_from_checkpoint).resolve(strict=True)
        run = checkpoint.parents[1] if checkpoint.name.startswith("ckpt_") else checkpoint
        source = json.loads((run / "run_config.json").read_text())
        for section, fields in {
            "model": ("native_keyframe_enabled", "native_keyframe_grid_hw", "native_keyframe_selector",
                      "native_keyframe_contract_version", "native_keyframe_spatial_coordinates",
                      "native_keyframe_manifest_sha256", "framesamp_memory_budget", "framesamp_tokens_per_frame",
                      "framesamp_dataset_revision"),
            "data": ("native_keyframe_manifest_sha256", "native_keyframe_selector",
                     "framesamp_feature_manifest_sha256", "framesamp_dataset_revision"),
            "train": ("name", "max_training_steps", "train_batch_size", "gradient_accumulation_steps"),
        }.items():
            target = getattr(self.cfg, section).model_dump(mode="json")
            for field in fields:
                if source.get(section, {}).get(field) != target[field]:
                    raise ValueError(f"Native keyframe resume lineage changed: {section}.{field}")

    def _load_clean_action_header(self, state):
        def memory_key(key):
            return key.startswith(("memory_encoder.", "framesamp_contract_version",
                                   "keyframe_native_contract_version")) or ".memory_" in key
        if any(memory_key(key) for key in state):
            raise ValueError("Native warm start requires the common clean Psi0 checkpoint")
        if not {"action_proj_out.linear.weight", "action_proj_in.dec_pos"}.issubset(state):
            raise ValueError("Native warm start is missing the clean Psi0 action shape metadata")
        target = self.model.action_header.state_dict()
        legacy = {key for key in target if not memory_key(key)}
        if state["action_proj_out.linear.weight"].shape[0] != self.model_cfg.action_dim:
            raise ValueError("Native formal warm start requires the original action dimension")
        mismatch = state["action_proj_in.dec_pos"].shape[0] != self.model_cfg.action_chunk_size
        selected = {k: v for k, v in state.items() if k.startswith("transformer_blocks.")} if mismatch else state
        expected = {k for k in legacy if k.startswith("transformer_blocks.")} if mismatch else legacy
        if set(selected) != expected:
            raise ValueError("Native warm start did not match the exact common Psi0 parameter subset")
        self.model.action_header.load_state_dict({**target, **selected}, strict=True)
        overwatch.info(f"Loaded {len(selected)} common Psi0 tensors; initialized native memory parameters")

    def init_models(self):
        vlm = self.init_qwen3vl_models()
        self.model = self.MODEL_CLASS(model_cfg=self.model_cfg, vlm_model=vlm)
        from safetensors.torch import load_file
        path = Path(self.model_cfg.pretrained_action_header_path) / "action_header.safetensors"
        self._load_clean_action_header(load_file(str(path), device="cpu"))
        if self.train_cfg.data_parallel == "deepspeed":
            self.model.config = vlm.config
            self.model.config.hidden_size = vlm.config.text_config.hidden_size
        for parameter in vlm.parameters():
            parameter.requires_grad_(False)
        vlm.eval()
        print("NATIVE_KEYFRAME_TRAINER_ACTIVE_JSON=" + json.dumps({
            "model_class": type(self.model).__name__, "trainer_class": type(self).__name__,
            "memory_frames": 7, "tokens_per_frame": 77, "memory_budget": 539,
            "selector": self.model_cfg.native_keyframe_selector,
            "manifest_sha256": self.data_cfg.native_keyframe_manifest_sha256,
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        }, sort_keys=True), flush=True)

    def create_dataloaders(self, train_dataset, val_dataset):
        loaders = FinetuneTrainer.create_dataloaders(self, train_dataset, val_dataset)
        collator = PaddedCollatorForNativeKeyframe(
            model_max_length=self.tokenizer.model_max_length, pad_token_id=self.tokenizer.pad_token_id,
            padding_side="right", memory_budget=539, feature_dim=2048)
        for loader in loaders:
            if loader is not None:
                if loader._iterator is not None:
                    raise RuntimeError("Native collator must be installed before iteration")
                loader.collate_fn = collator
        return loaders

    def training_step(self, batch):
        sync, metrics = super().training_step(batch)
        model = self.unwrap_model()
        metrics["native_keyframe/valid_tokens"] = batch["memory_mask"].sum(-1).float().mean().item()
        metrics["native_keyframe/memory_gradient_seen"] = float(model._framesamp_memory_grad_seen)
        if sync and not getattr(self.accelerator, "optimizer_step_was_skipped", False):
            self._native_updates += 1
            if not math.isfinite(float(metrics["loss"])):
                raise FloatingPointError("Nonfinite native keyframe training loss")
            if not model._framesamp_memory_grad_seen:
                raise RuntimeError("Native keyframe memory received no finite gradient")
            if self._native_updates == 1:
                print("NATIVE_KEYFRAME_FIRST_UPDATE_PASSED_JSON=" + json.dumps({
                    "rank": self.accelerator.process_index, "loss": float(metrics["loss"]),
                    "memory_gradient_seen": True, "memory_budget": 539, "tokens_per_frame": 77,
                    "valid_tokens": batch["memory_mask"].sum(-1).tolist(),
                    "manifest_sha256": self.data_cfg.native_keyframe_manifest_sha256,
                }, sort_keys=True), flush=True)
        return sync, metrics

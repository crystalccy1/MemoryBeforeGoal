from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from psi.models.psi0_framesamp import Psi0FrameSampModel
from psi.data.lerobot.framesamp_features import sha256_file
from psi.trainers.finetune import FinetuneTrainer, overwatch
from psi.trainers.qwen3vl_framesamp_mixin import PaddedCollatorForFrameSamp


class FinetuneFramesampTrainer(FinetuneTrainer):
    """Isolated trainer that cannot silently instantiate the clean Psi0 class."""

    MODEL_CLASS = Psi0FrameSampModel

    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self._validate_strict_resume_lineage()
        if not self.model_cfg.framesamp_entry_probe_only:
            model_manifest = Path(
                self.model_cfg.framesamp_feature_manifest
            ).resolve(strict=True)
            data_manifest = (
                Path(self.data_cfg.framesamp_feature_root) / "manifest.json"
            ).resolve(strict=True)
            expected_sha = self.data_cfg.framesamp_feature_manifest_sha256
            if (
                model_manifest != data_manifest
                or self.model_cfg.framesamp_feature_manifest_sha256 != expected_sha
                or self.model_cfg.framesamp_dataset_revision
                != self.data_cfg.framesamp_dataset_revision
                or self.model_cfg.framesamp_feature_scope
                != self.data_cfg.framesamp_feature_scope
                or sha256_file(model_manifest) != expected_sha
            ):
                raise RuntimeError("Model/data configs do not bind the same feature manifest")
            contract = json.loads((data_manifest.parent / "contract.json").read_text())
            if Path(self.model_cfg.model_name_or_path).resolve(strict=True) != Path(
                contract["encoder"]["checkpoint_dir"]
            ).resolve(strict=True):
                raise RuntimeError(
                    "Training Qwen checkpoint differs from the feature producer checkpoint"
                )

    def _validate_strict_resume_lineage(self) -> None:
        """Reject v1/stride-24 checkpoints before any model or data is loaded."""

        requested = self.train_cfg.resume_from_checkpoint
        if requested is None:
            return
        path = Path(str(requested)).expanduser().resolve(strict=True)
        run_root = path.parents[1] if path.name.startswith("ckpt_") else path
        config_path = run_root / "run_config.json"
        if config_path.is_symlink() or not config_path.is_file():
            raise RuntimeError(
                "Strict FrameSamp resume requires the source run_config.json"
            )
        source = json.loads(config_path.read_text())
        source_model = source.get("model") or {}
        expected = {
            "framesamp_memory_clock": "execution_frame",
            "framesamp_query_stride": 1,
            "framesamp_temporal_position_source": "raw_frame_index",
            "framesamp_training_origin": "common_psi0_warm_start_step0",
            "framesamp_feature_manifest_sha256": (
                self.model_cfg.framesamp_feature_manifest_sha256
            ),
        }
        mismatched = {
            key: {"expected": value, "observed": source_model.get(key)}
            for key, value in expected.items()
            if source_model.get(key) != value
        }
        if mismatched:
            raise RuntimeError(
                "Refusing non-v2 FrameSamp continuation; regenerate the complete "
                "raw-frame feature bank and start at step 0 from the common Psi-0 "
                f"warm start. Mismatches: {mismatched}"
            )

    def init_qwen3vl_models(self):
        """Use the same Qwen backend as offline and online memory production."""

        vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_cfg.model_name_or_path,
            attn_implementation="sdpa",
            dtype=torch.bfloat16,
        )
        self.vlm_processor = AutoProcessor.from_pretrained(
            self.model_cfg.model_name_or_path
        )
        self.tokenizer = self.vlm_processor.tokenizer
        overwatch.info(
            "Loaded Qwen3-VL with SDPA for strict offline/train/online parity"
        )
        if self.cfg.train.lora:
            raise ValueError("Strict FrameSamp does not support LoRA")
        return vlm_model

    def init_models(self):
        vlm_model = self.init_qwen3vl_models()
        self.model = self.MODEL_CLASS(
            model_cfg=self.model_cfg,
            vlm_model=vlm_model,
        )

        if self.model_cfg.pretrained_action_header_path is not None:
            from safetensors.torch import load_file

            ckpt_path = self.model_cfg.pretrained_action_header_path
            state_dict = load_file(f"{ckpt_path}/action_header.safetensors")
            horizon_mismatch = (
                state_dict["action_proj_in.dec_pos"].shape[0]
                != self.model_cfg.action_chunk_size
            )
            action_dim_mismatch = (
                state_dict["action_proj_out.linear.weight"].shape[0]
                != self.model_cfg.action_dim
            )
            if action_dim_mismatch and not self.model_cfg.framesamp_entry_probe_only:
                raise RuntimeError("Formal FrameSamp warm start requires action_dim=36")
            if horizon_mismatch or action_dim_mismatch:
                state_to_load = {
                    key: value
                    for key, value in state_dict.items()
                    if key.startswith("transformer_blocks")
                }
                incompatible = self.model.action_header.load_state_dict(
                    state_to_load, strict=False
                )
                overwatch.warning(
                    "action header horizon mismatch; loaded transformer blocks only "
                    "to match the released Psi0 SIMPLE recipe"
                )
            else:
                state_to_load = state_dict
                incompatible = self.model.action_header.load_state_dict(
                    state_to_load, strict=False
                )
            if incompatible.unexpected_keys:
                raise RuntimeError(
                    "Unexpected pretrained action-header keys: "
                    f"{incompatible.unexpected_keys}"
                )
            if not self.model_cfg.framesamp_entry_probe_only:
                wrapper_keys = set(self.model.action_header.state_dict())
                expected_missing = wrapper_keys - set(state_to_load)
                actual_missing = set(incompatible.missing_keys)
                if actual_missing != expected_missing:
                    raise RuntimeError(
                        "Warm start did not load the exact declared key subset: "
                        f"missing={sorted(actual_missing ^ expected_missing)}"
                    )
            overwatch.info(f"Loaded pretrained action header from {ckpt_path}")
        else:
            overwatch.info("No pretrained action header specified; training from scratch")

        if self.train_cfg.data_parallel == "deepspeed":
            self.model.config = vlm_model.config
            self.model.config.hidden_size = vlm_model.config.text_config.hidden_size

        for parameter in self.model.vlm_model.parameters():
            parameter.requires_grad = bool(self.model_cfg.tune_vlm)
        if not self.model_cfg.tune_vlm:
            overwatch.info("VLM parameters are frozen (tune_vlm=False)")
        elif (
            hasattr(self.model.vlm_model.model, "language_model")
            and hasattr(self.model.vlm_model.model.language_model, "norm")
        ):
            for parameter in self.model.vlm_model.model.language_model.norm.parameters():
                parameter.requires_grad = False

        trainable = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        payload = {
            "schema_version": "psi0-framesamp-trainer-entry/v1",
            "trainer_class": type(self).__name__,
            "model_class": type(self.model).__name__,
            "framesamp_enabled": bool(self.model_cfg.framesamp_enabled),
            "entry_probe_only": bool(self.model_cfg.framesamp_entry_probe_only),
            "trainable_parameters": int(trainable),
        }
        print(
            "PSI0_FRAMESAMP_TRAINER_ACTIVE_JSON="
            + json.dumps(payload, sort_keys=True, separators=(",", ":")),
            flush=True,
        )
        if type(self.model) is not self.MODEL_CLASS:
            raise RuntimeError("FrameSamp trainer instantiated the wrong model class")

    def create_dataloaders(self, train_dataset, val_dataset):
        """Preserve the base exact-resume loader and replace only collation."""
        train_dataloader, val_dataloader = super().create_dataloaders(
            train_dataset, val_dataset
        )
        if self.model_cfg.framesamp_entry_probe_only:
            return train_dataloader, val_dataloader
        if train_dataloader._iterator is not None or val_dataloader._iterator is not None:
            raise RuntimeError("FrameSamp collator must be installed before iteration")
        collator = PaddedCollatorForFrameSamp(
            model_max_length=self.tokenizer.model_max_length,
            pad_token_id=self.tokenizer.pad_token_id,
            padding_side="right",
            memory_budget=self.model_cfg.framesamp_memory_budget,
            feature_dim=self.model_cfg.framesamp_feature_dim,
        )
        train_dataloader.collate_fn = collator
        val_dataloader.collate_fn = collator
        return train_dataloader, val_dataloader

    @staticmethod
    def _memory_kwargs(batch):
        names = (
            "memory_features",
            "memory_mask",
            "memory_raw_frame_indices",
            "memory_spatial_indices",
        )
        missing = [name for name in names if name not in batch]
        if missing:
            raise KeyError(f"FrameSamp training batch is missing {missing}")
        return {name: batch[name] for name in names}

    def _bind_memory(self, model, batch):
        memory_kwargs = self._memory_kwargs(batch)

        def model_with_memory(**kwargs):
            return model(**kwargs, **memory_kwargs)

        return model_with_memory

    def forward_and_loss(self, model, batch):
        if self.model_cfg.framesamp_entry_probe_only:
            return super().forward_and_loss(model, batch)
        return super().forward_and_loss(self._bind_memory(model, batch), batch)

    @torch.no_grad()
    def inference(self, eval_model, repacked_batch):
        if self.model_cfg.framesamp_entry_probe_only:
            return super().inference(eval_model, repacked_batch)
        return super().inference(
            self._bind_memory(eval_model, repacked_batch), repacked_batch
        )

    def training_step(self, batch):
        result = super().training_step(batch)
        if os.environ.get("PSI_FRAMESAMP_FORMAL_EXIT_AFTER_UPDATE") == "1":
            model = self.unwrap_model()
            if self.model_cfg.framesamp_entry_probe_only:
                raise RuntimeError("Formal FrameSamp smoke used the entry-probe model")
            if type(model) is not self.MODEL_CLASS:
                raise RuntimeError("Formal smoke instantiated the wrong model class")
            if not model._framesamp_memory_grad_seen:
                raise RuntimeError("FrameSamp modulation received no finite gradient")
            sync_gradients, losses = result
            if not sync_gradients:
                raise RuntimeError("Formal smoke did not complete an optimizer update")
            payload = {
                "schema_version": "psi0-framesamp-formal-update/v1",
                "trainer_class": type(self).__name__,
                "model_class": type(model).__name__,
                "optimizer_update_completed": True,
                "memory_gradient_seen": True,
                "loss": float(losses["loss"]),
            }
            print(
                "PSI0_FRAMESAMP_FORMAL_UPDATE_JSON="
                + json.dumps(payload, sort_keys=True, separators=(",", ":")),
                flush=True,
            )
            self.accelerator.log(
                {
                    "framesamp_formal_smoke/loss": float(losses["loss"]),
                    "framesamp_formal_smoke/optimizer_update_completed": 1,
                    "framesamp_formal_smoke/memory_gradient_seen": 1,
                },
                step=1,
            )
            self.accelerator.end_training()
            raise SystemExit(0)
        if os.environ.get("PSI_FRAMESAMP_ENTRY_EXIT_AFTER_UPDATE") == "1":
            model = self.unwrap_model()
            if type(model) is not self.MODEL_CLASS:
                raise RuntimeError("Entry probe optimizer step used the wrong model class")
            if not model._framesamp_entry_probe_grad_seen:
                raise RuntimeError("FrameSamp entry probe did not participate in autograd")
            sync_gradients, losses = result
            if not sync_gradients:
                raise RuntimeError("Entry probe did not complete an optimizer update")
            payload = {
                "schema_version": "psi0-framesamp-entry-update/v1",
                "trainer_class": type(self).__name__,
                "model_class": type(model).__name__,
                "optimizer_update_completed": True,
                "probe_gradient_seen": True,
                "loss": float(losses["loss"]),
            }
            print(
                "PSI0_FRAMESAMP_OPTIMIZER_UPDATE_JSON="
                + json.dumps(payload, sort_keys=True, separators=(",", ":")),
                flush=True,
            )
            self.accelerator.log(
                {
                    "entry_probe/loss": float(losses["loss"]),
                    "entry_probe/optimizer_update_completed": 1,
                    "entry_probe/probe_gradient_seen": 1,
                },
                step=1,
            )
            self.accelerator.end_training()
            raise SystemExit(0)
        return result

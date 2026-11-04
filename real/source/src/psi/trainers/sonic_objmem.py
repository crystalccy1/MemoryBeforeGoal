"""SonicTrainer (the moving_dining_cart recipe) with the SAM3-v2 memory model.

Everything that defines the cart recipe is inherited unchanged: full 30-step action chunks only, RealRepack /
bounds normalisation, training-time RTC (max_delay prefix without noise or loss), flow loss, optimiser,
schedule, validation and checkpointing. The only additions are the memory tensors attached to each sample
(decision frame = the chunk start / observation frame) and passed to the model through a context.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import torch
from safetensors.torch import load_file

from psi.data.objmem_bank import ObjMemBank
from psi.models.psi0_objmem import MEMORY_FIELDS, Psi0ObjMemModel
from psi.trainers.qwen3vl_mixin import PaddedCollatorForTogether
from psi.trainers.sonic import SonicTrainer, _FullActionChunkDataset
from psi.utils import initialize_overwatch

overwatch = initialize_overwatch(__name__)


class _FullActionChunkMemoryDataset(_FullActionChunkDataset):
    """Cart full-chunk sampling plus the SAM3-v2 memory of the observation frame."""

    def __init__(self, base, horizon: int, bank: ObjMemBank, source_episodes: list[int]):
        super().__init__(base, horizon)
        self.bank = bank
        self.source_episodes = list(source_episodes)

    def __getitem__(self, idx):
        source_idx = int(self.valid_starts[idx])
        raw = self.base.raw_dataset[source_idx]
        if "action_is_pad" not in raw:
            raise RuntimeError("loader must expose action_is_pad for the action time window")
        padding = torch.as_tensor(raw["action_is_pad"]).reshape(-1)
        if padding.shape != (self.horizon,) or bool(padding.any()):
            raise RuntimeError(f"invalid full-chunk start {source_idx}: padding={padding.tolist()}")
        episode, frame = int(raw["episode_index"]), int(raw["frame_index"])
        sample = self.base.transform(raw, **self.base.transform_kwargs)
        for name, value in self.bank.memory_for_frame(self.source_episodes[episode], frame).items():
            sample[name] = value
        return sample


class PaddedCollatorForObjMem(PaddedCollatorForTogether):
    def __call__(self, instances):
        memory = {name: torch.stack([instance[name] for instance in instances]) for name in MEMORY_FIELDS}
        output = super().__call__(instances)
        output.update(memory)
        return output


def _unwrap(model):
    while hasattr(model, "module"):
        model = model.module
    return model


class SonicObjmemTrainer(SonicTrainer):

    def init_models(self):
        vlm_model = self.init_qwen3vl_models()
        self.model = Psi0ObjMemModel(model_cfg=self.model_cfg, vlm_model=vlm_model)
        header = self.model.action_header
        path = self.model_cfg.pretrained_action_header_path
        if path is None:
            raise ValueError("The cart recipe warm-starts the action header from m1")
        state = load_file(f"{path}/action_header.safetensors")
        # Same rule as SonicTrainer.init_models: transformer blocks only when chunk/action dims differ.
        if (state["action_proj_in.dec_pos"].shape[0] != self.model_cfg.action_chunk_size
                or state["action_proj_out.linear.weight"].shape[0] != self.model_cfg.action_dim):
            selected = {k: v for k, v in state.items() if k.startswith("transformer_blocks")}
            overwatch.warning("action header size mismatch, only loaded transformer blocks.")
        else:
            selected = state
        result = header.load_state_dict(selected, strict=False)
        loaded = sorted(set(selected) - set(result.unexpected_keys))
        receipt = {"pretrained_action_header": str(path), "selected": len(selected), "loaded": len(loaded),
                   "unexpected": list(result.unexpected_keys),
                   "memory_parameters": len(header.memory_parameter_names())}
        overwatch.info("OBJMEM_WARM_START_JSON=" + json.dumps(receipt))
        target = header.state_dict()
        for key in loaded:
            if not torch.equal(target[key].cpu(), selected[key].to(target[key].dtype)):
                raise RuntimeError(f"warm-start tensor was not copied: {key}")
        for p in self.model.vlm_model.parameters():
            p.requires_grad = False
        overwatch.info("VLM parameters are frozen (tune_vlm=False)")
        self._objmem_grad_seen = {"object_out_proj": False, "memory_film": False}

        def hook(name):
            def record(gradient):
                if not self._objmem_grad_seen[name] and torch.isfinite(gradient).all() and gradient.abs().sum() > 0:
                    self._objmem_grad_seen[name] = True
                return gradient
            return record

        for block in header.transformer_blocks:
            block.memory_attention.object_attention.out_proj.weight.register_hook(hook("object_out_proj"))
            block.memory_to_scale_shift.weight.register_hook(hook("memory_film"))
        self._objmem_updates = 0
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        overwatch.info(f"Model has {trainable:,} trainable parameters")

    def _bank(self) -> ObjMemBank:
        if getattr(self, "_objmem_bank", None) is None:
            self._objmem_bank = ObjMemBank(self.model_cfg.objmem_bank_root,
                                           expected_manifest_sha256=self.model_cfg.objmem_bank_manifest_sha256)
            if self._objmem_bank.slots != self.model_cfg.object_slots:
                raise ValueError("object_slots differs between the model config and the bank")
            # Before the dataloader workers fork: they share these tensors instead of random NFS reads.
            start = time.monotonic()
            loaded = self._objmem_bank.preload()
            overwatch.info(f"OBJMEM_BANK_PRELOADED bytes={loaded} seconds={time.monotonic() - start:.1f}")
        return self._objmem_bank

    def _source_episodes(self, repo: str) -> list[int]:
        mapping = json.loads((Path(self.data_cfg.root_dir) / repo / "meta/source_episode_mapping.json").read_text())
        if [m["episode_index"] for m in mapping] != list(range(len(mapping))):
            raise ValueError(f"{repo} episode mapping is not contiguous")
        bank = self._bank()
        for m in mapping:
            if int(bank.records[m["source_episode"]]["raw_frames"]) != int(m["frames"]):
                raise ValueError(f"{repo} episode {m['episode_index']} frame count differs from the bank")
        return [int(m["source_episode"]) for m in mapping]

    def create_datasets(self):
        if len(self.data_cfg.train_repo_ids) != 1 or len(self.data_cfg.val_repo_ids) != 1:
            raise ValueError("objmem trainer expects one train and one val repo")
        transform_kwargs = dict(vlm_processor=self.vlm_processor)
        val_data_cfg = self.data_cfg.model_copy(deep=True)
        self.train_dataset = _FullActionChunkMemoryDataset(
            self.data_cfg(split="train", transform_kwargs=transform_kwargs), self.ac_chunk, self._bank(),
            self._source_episodes(self.data_cfg.train_repo_ids[0]))
        self.val_dataset = _FullActionChunkMemoryDataset(
            val_data_cfg(split="val", transform_kwargs={**transform_kwargs, "no_aug": True}), self.ac_chunk,
            self._bank(), self._source_episodes(self.data_cfg.val_repo_ids[0]))
        return self.train_dataset, self.val_dataset

    def create_dataloaders(self, train_dataset, val_dataset):
        loaders = super().create_dataloaders(train_dataset, val_dataset)
        collator = PaddedCollatorForObjMem(model_max_length=self.tokenizer.model_max_length,
                                           pad_token_id=self.tokenizer.pad_token_id, padding_side="right")
        for loader in loaders:
            if loader is not None:
                if loader._iterator is not None:
                    raise RuntimeError("memory collator must be installed before iteration")
                loader.collate_fn = collator
        return loaders

    @staticmethod
    def _memory(batch):
        return {name: batch[name] for name in MEMORY_FIELDS}

    def forward_and_loss(self, model, batch):
        with _unwrap(model).memory_context(self._memory(batch)):
            return super().forward_and_loss(model, batch)

    def inference(self, eval_model, repacked_batch):
        with _unwrap(eval_model).memory_context(self._memory(repacked_batch)):
            return super().inference(eval_model, repacked_batch)

    def training_step(self, batch):
        sync, metrics = super().training_step(batch)
        model = _unwrap(self.model)
        gates, norms = model.object_gates(), model.object_out_proj_norms()
        metrics["objmem/gate_mean"] = sum(gates) / len(gates)
        metrics["objmem/object_out_proj_norm_mean"] = sum(norms) / len(norms)
        metrics["objmem/object_tokens_valid"] = batch["memory_object_mask"].sum(-1).float().mean().item()
        metrics["objmem/memory_frames"] = batch["memory_mask"].sum(-1).float().mean().item() / 77
        if sync:
            self._objmem_updates += 1
            if self._objmem_updates == 1:
                if not all(self._objmem_grad_seen.values()):
                    raise RuntimeError(f"memory branch got no finite gradient on the first update: {self._objmem_grad_seen}")
                print("OBJMEM_FIRST_UPDATE_PASSED_JSON=" + json.dumps({"gates": gates, "out_proj_norms": norms}), flush=True)
        return sync, metrics

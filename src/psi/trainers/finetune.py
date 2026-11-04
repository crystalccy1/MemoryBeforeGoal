# Derived from Psi0 (https://github.com/physical-superintelligence-lab/Psi0, Apache License 2.0) and modified; see NOTICE.
from __future__ import annotations
import os
import json
import torch
import wandb
import pickle
import numpy as np
from tqdm import tqdm
from copy import copy, deepcopy
import contextlib
import torch.nn.functional as F
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, List, Union, Any, TYPE_CHECKING
from torch.utils.data import DataLoader, RandomSampler
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, Qwen3VLProcessor
from transformers.utils import is_flash_attn_2_available
from psi.trainers.qwen3vl_mixin import PaddedCollatorForTogether

if TYPE_CHECKING:    
    from psi.config.config import TrainConfig
    from psi.config.model_psi0 import Psi0ModelConfig
    from psi.config.transform import Psi0ModelTransform

from accelerate import Accelerator

# from utils.overwatch import initialize_overwatch
from psi.utils import (
    initialize_overwatch,
    shorten
)
from psi.utils.utils import batch_str_to_tensor

overwatch = initialize_overwatch(__name__)

# from .base import Trainer
from .trainer import Trainer, worker_init_fn

from psi.utils import flatten, shorten, move_to_device, rmse, seed_everything
from psi.models.psi0 import Psi0Model


@dataclass(frozen=True)
class TrainSamplerResumePlan:
    """Exact index-stream position for a checkpoint-boundary resume.

    This plan deliberately covers sampler indices only.  A newly launched
    segment creates new persistent DataLoader workers, so the workers' evolved
    augmentation/prefetch RNG state from the previous allocation is not
    recoverable from an Accelerate checkpoint.
    """

    initial_global_step: int
    completed_epochs: int
    steps_per_epoch: int
    skip_batches: int
    prepared_batches_per_epoch: int
    gradient_accumulation_steps: int


def _draw_dataloader_base_seed(generator: torch.Generator) -> int:
    """Mirror PyTorch 2.7 ``_BaseDataLoaderIter``'s base-seed draw."""

    return int(
        torch.empty((), dtype=torch.int64).random_(generator=generator).item()
    )


def _legacy_compatible_train_generators(
    seed: int,
) -> tuple[torch.Generator, torch.Generator, torch.Tensor, int]:
    """Split the legacy shared generator without changing its fresh trace.

    The released finetune loader passed one generator both to DataLoader (for
    the initial worker base seed) and to ``RandomSampler``.  On the first
    iterator PyTorch draws the worker base seed before it asks the sampler for
    its first index.  A separate sampler generator prevents a new segment's
    worker-seed draw from perturbing the shuffle stream; consuming the same
    int64 draw here keeps the fresh permutation byte-for-byte identical.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"train sampler seed must be an integer, got {seed!r}")

    worker_generator = torch.Generator()
    worker_generator.manual_seed(seed)

    sampler_generator = torch.Generator()
    sampler_generator.manual_seed(seed)
    legacy_base_seed = _draw_dataloader_base_seed(sampler_generator)
    epoch_zero_state = sampler_generator.get_state().clone()
    return (
        sampler_generator,
        worker_generator,
        epoch_zero_state,
        legacy_base_seed,
    )


def _plan_and_fast_forward_train_sampler(
    *,
    sampler: RandomSampler,
    epoch_zero_state: torch.Tensor,
    initial_global_step: int,
    prepared_batches_per_epoch: int,
    gradient_accumulation_steps: int,
) -> TrainSamplerResumePlan:
    """Reset and advance ``sampler`` to an uninterrupted checkpoint position.

    Exhausting the actual PyTorch ``RandomSampler`` is intentional.  In
    PyTorch 2.7 its no-replacement implementation consumes a second
    ``randperm`` even when ``num_samples == len(dataset)`` and the second slice
    is empty.  Replaying the sampler itself therefore avoids encoding a
    fragile assumption about RNG consumption.
    """

    for name, value in (
        ("initial_global_step", initial_global_step),
        ("prepared_batches_per_epoch", prepared_batches_per_epoch),
        ("gradient_accumulation_steps", gradient_accumulation_steps),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer, got {value!r}")

    if initial_global_step < 0:
        raise ValueError(
            f"initial_global_step must be non-negative, got {initial_global_step}"
        )
    if prepared_batches_per_epoch <= 0:
        raise ValueError(
            "prepared training loader must contain at least one batch per epoch; "
            f"got {prepared_batches_per_epoch}"
        )
    if gradient_accumulation_steps <= 0:
        raise ValueError(
            "gradient_accumulation_steps must be positive; "
            f"got {gradient_accumulation_steps}"
        )
    if not isinstance(sampler, RandomSampler):
        raise TypeError(
            "exact segmented finetune resume requires torch RandomSampler; "
            f"got {type(sampler).__name__}"
        )
    if sampler.replacement:
        raise ValueError("exact segmented finetune resume requires replacement=False")
    if sampler.generator is None:
        raise ValueError("exact segmented finetune resume requires an explicit generator")
    if sampler.num_samples != len(sampler.data_source):
        raise ValueError(
            "exact segmented finetune resume requires one full dataset permutation "
            f"per epoch; num_samples={sampler.num_samples}, "
            f"dataset_size={len(sampler.data_source)}"
        )
    if not isinstance(epoch_zero_state, torch.Tensor):
        raise TypeError("epoch_zero_state must be a torch.Tensor")

    # Accelerate synchronizes the final partial accumulation at end-of-loader,
    # so the optimizer-step cadence is ceil(num_batches / accumulation).
    steps_per_epoch = (
        prepared_batches_per_epoch + gradient_accumulation_steps - 1
    ) // gradient_accumulation_steps
    completed_epochs, steps_into_epoch = divmod(
        initial_global_step, steps_per_epoch
    )
    skip_batches = steps_into_epoch * gradient_accumulation_steps
    if skip_batches >= prepared_batches_per_epoch:
        raise RuntimeError(
            "computed resume offset lies outside the prepared training loader: "
            f"skip={skip_batches}, batches={prepared_batches_per_epoch}"
        )

    sampler.generator.set_state(epoch_zero_state.clone())
    for _ in range(completed_epochs):
        # Fully exhaust the sampler, including RandomSampler's empty second
        # randperm slice, while retaining no indices in memory.
        deque(iter(sampler), maxlen=0)

    return TrainSamplerResumePlan(
        initial_global_step=initial_global_step,
        completed_epochs=completed_epochs,
        steps_per_epoch=steps_per_epoch,
        skip_batches=skip_batches,
        prepared_batches_per_epoch=prepared_batches_per_epoch,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

class FinetuneTrainer(Trainer):

    def __init__(self, cfg, device: torch.device):
        super().__init__(cfg, device)

        self.noise_scheduler_name = self.model_cfg.noise_scheduler
        self.train_diffusion_steps = self.model_cfg.train_diffusion_steps
        self.eval_diffusion_steps = self.model_cfg.eval_diffusion_steps
        self.ac_chunk = self.model_cfg.action_chunk_size
        self.ac_dim = self.model_cfg.action_dim
        self.maxmin = self.data_cfg.transform.field

        if self.model_cfg.action_dim == 7:
            w_xyz, w_rpy, w_gripper = self.model_cfg.loss_w
            self.loss_w = (
                torch.from_numpy(np.array([w_xyz] * 3 + [w_rpy] * 3 + [w_gripper]))
                .float()
                .to(self.device)
            )  # (7,)
        else:
            self.loss_w = torch.tensor([1.0 / self.model_cfg.action_dim] * self.model_cfg.action_dim, dtype=torch.float32).to(self.device)
        assert self.loss_w.sum() == 1.0, "Weights better sum to 1.0 to keep loss range consistent"

        if self.noise_scheduler_name == "ddpm":
            from diffusers.schedulers.scheduling_ddim import DDIMScheduler
            self.noise_scheduler = DDIMScheduler(  # FIXME configure args
                num_train_timesteps=self.train_diffusion_steps,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule="squaredcos_cap_v2",
                clip_sample=True,
                set_alpha_to_one=True,
                steps_offset=0,
                prediction_type="epsilon",
            )
        elif self.noise_scheduler_name == "flow":
            from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
                FlowMatchEulerDiscreteScheduler,
            )
            self.noise_scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=self.train_diffusion_steps,  # MUST be 1000 as per pretrained SD3
            )
        # assert self.task_cfg.mixed_precision == "no", "other options not tested"
        # assert self.model_cfg.n_conditions == len(self.data_cfg.transform.repack.conditions), "inconsistent confs" # type: ignore

    @property
    def task_cfg(self) -> TrainConfig:
        return self.cfg.train

    @property
    def model_cfg(self) -> Psi0ModelConfig:
        return self.cfg.model # type: ignore
    
    def init_qwen3vl_models(self):
        attention_backend = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
        overwatch.info(f"Using {attention_backend} attention backend")
        vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_cfg.model_name_or_path,
            attn_implementation=attention_backend,
            dtype=torch.bfloat16
        )
        overwatch.info(f"Load pretrained VLM model from {self.model_cfg.model_name_or_path}")

        self.vlm_processor = AutoProcessor.from_pretrained(self.model_cfg.model_name_or_path)
        self.tokenizer = self.vlm_processor.tokenizer

        assert self.cfg.train.lora == False

        return vlm_model

    def init_models(self):
        # init pretrained vlm
        vlm_model = self.init_qwen3vl_models()

        self.model = Psi0Model(
            model_cfg=self.model_cfg,
            vlm_model=vlm_model,
        )

        # TODO refac
        if self.model_cfg.pretrained_action_header_path is not None:
            # load pretrained action header ckpt
            from safetensors.torch import load_file
            ckpt_path = self.model_cfg.pretrained_action_header_path
            state_dict = load_file(f"{ckpt_path}/action_header.safetensors")
            if (state_dict["action_proj_in.dec_pos"].shape[0] != self.model_cfg.action_chunk_size or
                state_dict["action_proj_out.linear.weight"].shape[0] != self.model_cfg.action_dim):
                # only load transformer blocks
                reduced_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("transformer_blocks"):
                        reduced_state_dict[k] = v
                overwatch.info(f"Loading pretrained action header from {ckpt_path}")
                self.model.action_header.load_state_dict(reduced_state_dict, strict=False)
                overwatch.warning("action header size mismatch, only loaded transformer blocks.")
            else:
                self.model.action_header.load_state_dict(state_dict, strict=False)
            overwatch.info("loaded pretrained action header successfully.")
        else:
            overwatch.info("No pretrained action header specified, training from scratch.")
        
        if self.train_cfg.data_parallel == "deepspeed":
            # HACK to set correct config for deepspeed
            self.model.config = vlm_model.config
            self.model.config.hidden_size = vlm_model.config.text_config.hidden_size

        # Set VLM parameters trainability based on tune_vlm flag
        if not self.model_cfg.tune_vlm:
            for p in self.model.vlm_model.parameters():
                p.requires_grad = False
            overwatch.info("VLM parameters are frozen (tune_vlm=False)")
        else:
            for p in self.model.vlm_model.parameters():
                p.requires_grad = True
            # Freeze the final norm layer since we don't use it (we extract hidden_states[-1])
            if hasattr(self.model.vlm_model.model, 'language_model') and hasattr(self.model.vlm_model.model.language_model, 'norm'):
                for p in self.model.vlm_model.model.language_model.norm.parameters():
                    p.requires_grad = False
                overwatch.info("VLM final norm layer frozen (not used in forward pass)")
            overwatch.info("VLM parameters are trainable (tune_vlm=True)")

        num_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        overwatch.info(f"Model has {num_parameters:,} trainable parameters")


    def create_optimizers(self):
        # Separate parameters into groups
        action_header_params = []
        vlm_model_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if name.startswith("action_header."):
                action_header_params.append(param)
            elif name.startswith("vlm_model."):
                if self.model_cfg.tune_vlm:
                    vlm_model_params.append(param)
            else:
                other_params.append(param)

        param_groups = []
        if action_header_params:
            param_groups.append({
                'params': action_header_params,
                'lr': self.cfg.train.learning_rate,
                'group_name': 'action_header'
            })
        if vlm_model_params:
            param_groups.append({
                'params': vlm_model_params,
                'lr': self.model_cfg.lang_backbone_lr,
                'group_name': 'vlm_model'
            })
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': self.cfg.train.learning_rate,  # Default group
                'group_name': 'other'
            })

        optimizer_kwargs = dict(self.cfg.train.lr_scheduler_kwargs)
        if self.cfg.train.optimizer_foreach is not None:
            optimizer_kwargs["foreach"] = self.cfg.train.optimizer_foreach
        self.optimizer = torch.optim.AdamW(
            param_groups,
            **optimizer_kwargs, # type: ignore
        )

    def create_lr_schedulers(
        self,
        num_training_steps: int | None = None,
        optimizer: torch.optim.Optimizer | None = None,
    ):
        from transformers.optimization import get_scheduler
        assert num_training_steps is not None
        self.lr_scheduler = get_scheduler(
            self.cfg.train.lr_scheduler_type,
            num_training_steps=num_training_steps * self.world_size,
            optimizer=optimizer if optimizer is not None else self.optimizer,
            num_warmup_steps=self.num_warmup_steps * self.world_size,
            scheduler_specific_kwargs=self.cfg.train.scheduler_specific_kwargs,
        )

    def create_optimizer_and_scheduler(self, num_training_steps: int | None = None):
        optimizer = self.create_optimizers()
        self.create_lr_schedulers(
            num_training_steps=num_training_steps, optimizer=optimizer
        )

    def create_datasets(self):  # TODO use parent impl.
        transform_kwargs=dict(
            vlm_processor=self.vlm_processor,
        )
        self.train_dataset = self.data_cfg(split="train", transform_kwargs=transform_kwargs)
        self.val_dataset = self.data_cfg(split="val", transform_kwargs=transform_kwargs)
        return self.train_dataset, self.val_dataset

    def create_dataloaders(self, train_dataset, val_dataset):
        train_seed = self.cfg.seed or 42
        num_workers = self.cfg.train.num_workers
        if num_workers > 0:
            (
                sampler_generator,
                worker_generator,
                sampler_epoch_zero_state,
                legacy_worker_base_seed,
            ) = _legacy_compatible_train_generators(train_seed)
            exact_segmented_resume_supported = True
        else:
            # With no persistent workers, legacy DataLoader draws from its
            # shared generator on every new iterator.  Retain that behavior
            # for fresh debug runs, but reject segmented resume below because
            # separating it without changing later fresh epochs is impossible.
            sampler_generator = torch.Generator()
            sampler_generator.manual_seed(train_seed)
            worker_generator = sampler_generator
            sampler_epoch_zero_state = sampler_generator.get_state().clone()
            legacy_worker_base_seed = None
            exact_segmented_resume_supported = False
        train_sampler = RandomSampler(
            train_dataset,
            replacement=False,
            generator=sampler_generator,
        )

        # Keep direct references after Accelerate wraps the DataLoader.  They
        # are used to reconstruct the uninterrupted sampler state only after
        # the optimizer checkpoint has supplied the completed global step.
        self._train_random_sampler = train_sampler
        self._train_sampler_epoch_zero_state = sampler_epoch_zero_state
        self._train_sampler_seed = train_seed
        self._train_legacy_worker_base_seed = legacy_worker_base_seed
        self._train_worker_generator = worker_generator
        self._exact_segmented_sampler_resume_supported = (
            exact_segmented_resume_supported
        )

        train_dataloader_kwargs = {
            "num_workers": num_workers,
            "drop_last": True,
            "sampler": train_sampler,
            "generator": worker_generator,
            "worker_init_fn": worker_init_fn,
            "persistent_workers": num_workers > 0,
        }

        val_dataloader_kwargs = {
            "num_workers": num_workers,
            "drop_last": False,
            # "pin_memory": True,
            "persistent_workers": num_workers > 0,
            "shuffle": True,
        }  # 1 small enough to fit im Mem. 2. no need distributed sampler

        collator = PaddedCollatorForTogether(
            model_max_length=self.tokenizer.model_max_length,
            pad_token_id=self.tokenizer.pad_token_id,
            padding_side="right",
        )

        # create training and validation dataloaders
        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.task_cfg.train_batch_size,
            collate_fn=collator,
            **train_dataloader_kwargs,
        )
        self.val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.task_cfg.val_batch_size,
            collate_fn=collator,
            **val_dataloader_kwargs,
        )
        return self.train_dataloader, self.val_dataloader

    @staticmethod
    def _loader_contains_sampler(loader: Any, target: RandomSampler) -> bool:
        """Check that Accelerate's wrapper still reaches the sampler we own."""

        pending = [loader]
        visited: set[int] = set()
        while pending:
            current = pending.pop()
            if current is target:
                return True
            current_id = id(current)
            if current_id in visited:
                continue
            visited.add(current_id)
            for attribute in ("base_dataloader", "batch_sampler", "sampler"):
                child = getattr(current, attribute, None)
                if child is not None and child is not current:
                    pending.append(child)
        return False

    def configure_train_sampler_resume(
        self, initial_global_step: int
    ) -> TrainSamplerResumePlan:
        """Position the prepared loader at the uninterrupted index stream.

        This is called after ``accelerator.load_state`` because only the
        checkpoint name establishes how many optimizer updates completed.
        It is automatic for this trainer even when the legacy config field
        ``skip_resumed_steps`` is false; that field predates exact segmented
        sampler reconstruction.
        """

        if self.cfg.train.overfit_single_batch:
            if initial_global_step:
                raise ValueError(
                    "segmented sampler resume is undefined with overfit_single_batch"
                )
            return TrainSamplerResumePlan(
                initial_global_step=0,
                completed_epochs=0,
                steps_per_epoch=1,
                skip_batches=0,
                prepared_batches_per_epoch=1,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
            )
        if not self._exact_segmented_sampler_resume_supported:
            if initial_global_step:
                raise ValueError(
                    "exact segmented finetune resume requires num_workers > 0 "
                    "and persistent workers; non-persistent legacy DataLoaders "
                    "consume an additional shared-generator base seed on every epoch"
                )
            prepared_batches = len(self.train_dataloader)
            steps_per_epoch = (
                prepared_batches + self.gradient_accumulation_steps - 1
            ) // self.gradient_accumulation_steps
            plan = TrainSamplerResumePlan(
                initial_global_step=0,
                completed_epochs=0,
                steps_per_epoch=steps_per_epoch,
                skip_batches=0,
                prepared_batches_per_epoch=prepared_batches,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
            )
            self.train_sampler_resume_plan = plan
            return plan
        if not self._loader_contains_sampler(
            self.train_dataloader, self._train_random_sampler
        ):
            raise RuntimeError(
                "Accelerate prepared loader no longer contains the registered "
                "finetune RandomSampler"
            )
        if initial_global_step >= self.max_training_steps:
            raise ValueError(
                "resume step has reached or exceeds max_training_steps: "
                f"resume={initial_global_step}, max={self.max_training_steps}"
            )

        prepared_batches = len(self.train_dataloader)
        plan = _plan_and_fast_forward_train_sampler(
            sampler=self._train_random_sampler,
            epoch_zero_state=self._train_sampler_epoch_zero_state,
            initial_global_step=initial_global_step,
            prepared_batches_per_epoch=prepared_batches,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
        )
        self.train_sampler_resume_plan = plan
        return plan

    @property
    def task_run_name(self):
        dataset_name = shorten(self.data_cfg.transform.repack.dataset_name) # type:ignore
        return (
            f".{dataset_name}"
            f".{self.noise_scheduler_name}{self.train_diffusion_steps}"
            f".{shorten(self.task_cfg.lr_scheduler_type)}"
            f".lr{self.task_cfg.learning_rate:.1e}"
        )

    def prepare(self, accelerator: Accelerator) -> DataLoader:
        # NOTE: if use deepspeed, model, optimizer, lr_scheduler must be prepared together !!! 
        self.model, self.optimizer, self.lr_scheduler, self.train_dataloader = accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler, self.train_dataloader
        )

        if self.train_cfg.data_parallel == "deepspeed":
            # Gradient Checkpoint Setup
            if self.model_cfg.gradient_checkpointing and self.model_cfg.tune_vlm:
                assert "DeepSpeedEngine" in self.model.__class__.__name__, "deepspeed is not properly initialized!"
                if hasattr(self.model, "enable_input_require_grads"): 
                    self.model.vlm_model.enable_input_require_grads()
                else:
                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)
                    self.model.vlm_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        if self.cfg.train.overfit_single_batch:
            overwatch.warning("Overfitting a single batch: reusing first batch every step. set cfg.data.image_aug = False for true memorization.")
            first_batch = next(iter(self.train_dataloader))
            class SingleBatchLoader:
                def __iter__(self): 
                    while True:
                        yield first_batch
                def __len__(self):
                    return 1
            self.train_dataloader = SingleBatchLoader()

        val_dataloader = getattr(self, "val_dataloader", None)
        if val_dataloader is not None: # not using if self.val_dataloader to avoid DataLoader.__len__() being called on iterable dataset
            self.val_dataloader = accelerator.prepare(self.val_dataloader)
            
        self.accelerator = accelerator
        return self.train_dataloader # type: ignore

    def training_step(
        self,
        batch: dict[str, Union[torch.Tensor, Any]],
    ) -> tuple[bool, dict[str, Any]]:

        if self.train_cfg.data_parallel == "deepspeed":
            gac = contextlib.nullcontext()
        else:
            # otherwise gradient accumulation is managed by deepspeed
            gac = self.accelerator.accumulate(self.model)
        with gac:
            with self.accelerator.autocast():
                losses = self.forward_and_loss(self.model, batch)

            self.accelerator.backward(losses["loss"])
            if self.accelerator.sync_gradients:
                if self.train_cfg.max_grad_norm is not None:
                    self._grad_norm_act = self.accelerator.clip_grad_norm_(
                        self.model.parameters(), self.cfg.train.max_grad_norm
                    )
                else:
                    self._grad_norm_act = self.get_total_grad_norm(self.model.parameters())

                if self.train_cfg.data_parallel == "ddp":
                    # NOTE: It is normal that param.grad is None for deepspeed and fsdp
                    for name, param in self.model.named_parameters():
                        if param.requires_grad and param.grad is None:
                            overwatch.critical(f"[Unused] {name} did not receive a gradient.")
            else:
                self._grad_norm_act = 0.0

            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

        # Log batch["raw_image"] to wandb every 100 steps
        if (
            hasattr(self, "global_step")
            and self.global_step % self.cfg.log.log_freq == 0
            and "raw_images" in batch
            and self.accelerator.is_main_process
            and self.cfg.log.report_to == "wandb"
            and os.getenv("PSI_LOG_RAW_IMAGES", "0") == "1"
        ):
            raw_imgs = batch["raw_images"]
            # Support both torch.Tensor and numpy
            if isinstance(raw_imgs, torch.Tensor):
                raw_imgs = raw_imgs.detach().cpu().numpy()
            # Log up to 4 images for visualization: concat them horizontally into one image
            img_arrays = []
            for i in range(min(10, len(raw_imgs))):
                img = raw_imgs[i][0]  # t=0
                img_arrays.append(img)

            # Images are assumed to have same shape and 3 channels; concat directly
            concat_img = np.concatenate(img_arrays, axis=1)
            wandb.log({"raw_images": [wandb.Image(concat_img, caption=f"raw images {self.global_step}")]}, step=self.global_step)

        # step_loss = self.accelerator.gather(losses["loss"].detach()).mean().item() # type:ignore
        # self.train_loss_tracker = step_loss
        self.train_loss_tracker = step_loss = losses["loss"].detach().item()

        return (self.accelerator.sync_gradients, {
            "lr_act": self.lr, 
            "grad_norm_act": self._grad_norm_act, # type: ignore
            "loss": step_loss
        })

    @torch.no_grad()
    def inference(self, eval_model, repacked_batch):

        traj2ds = repacked_batch["traj2ds"] if "traj2ds" in repacked_batch else None  # (B, 3, H, W)
        obs = repacked_batch["states"]  # (B,1,M)

        # imgs_in = {"cam0": view_features}  # views # TODO support multi-view
        bsz = obs.shape[0]

        action_samples = torch.randn(
            bsz, self.ac_chunk, self.ac_dim, device=self.device
        )
        self.noise_scheduler.set_timesteps(self.eval_diffusion_steps)

        for timestep in self.noise_scheduler.timesteps:
            batched_timestep = timestep.expand(bsz).to(self.device)

            model_pred = eval_model(
                input_ids=repacked_batch["input_ids"],#####
                attention_mask=repacked_batch["attention_mask"], # vlm related
                pixel_values=repacked_batch["pixel_values"],
                image_grid_thw=repacked_batch["image_grid_thw"], ####
                action_samples=action_samples, # (B,Tp,Da)
                states=obs,
                timestep=batched_timestep,
                traj2ds=traj2ds,
                return_dict=True,
            ).action

            action_samples = self.noise_scheduler.step(
                model_output=model_pred, timestep=timestep, sample=action_samples # type: ignore
            ).prev_sample

        return action_samples
    
    def save_checkpoint(self, global_step: int) -> str | None:
        # Accelerator.save_state() in the base implementation already saves
        # the model. Calling save_model() first wrote the same multi-GB model
        # twice to the shared filesystem for every checkpoint.
        return super().save_checkpoint(global_step)

    def evaluate(self) -> dict[str, float] | None:
        accelerator = self.accelerator
        global_step = self.global_step
        eval_model = self.unwrap_model()

        total_val_batches = (
            len(self.val_dataloader)
            if self.task_cfg.val_num_batches == -1
            else min(self.task_cfg.val_num_batches, len(self.val_dataloader))
        )
        val_progress_bar = tqdm(
            self.val_dataloader,
            total=total_val_batches,
            disable=not accelerator.is_local_main_process,
            position=1,
            leave=False,
        )
        val_progress_bar.set_description(f"Eval at global step {global_step}")

        val_loss_list = []
        action_l1_err_list = []

        for val_step, val_batch in enumerate(val_progress_bar):
            val_batch = batch_str_to_tensor(val_batch)
            # mask = val_batch["mask"]
            mask = torch.ones_like(val_batch["actions"]) # FIXME
            gt_actions = val_batch["actions"]  # (B, Tp, Da)
            # gt_actions = self.data_cfg.data_transforms.normalize_action(repacked_batch[1])

            # Tp -> predicted action horizon, Da -> action dim
            B, Tp, Da = gt_actions.shape

            # validation loss
            with accelerator.autocast():
                val_loss = self.forward_and_loss(eval_model, val_batch)
                val_loss_list.append(accelerator.gather(val_loss["loss"]))

                # action prediction loss
                pred_actions = self.inference(eval_model, val_batch)
                err_action_l1 = pred_actions - gt_actions  # (B, Tp, Da)

                err_action_l1_all = accelerator.gather(
                    err_action_l1[:, :Tp].contiguous()
                )
                err_action_masks_all = accelerator.gather(mask[:, :Tp].contiguous())
                err_action_l1 = err_action_l1_all[
                    err_action_masks_all.to(torch.bool) # type: ignore
                ].abs()
                action_l1_err_list.append(err_action_l1.reshape(-1, Da).float().cpu().numpy())  # (B*world_size*Ta, 7)

            if val_step + 1 >= total_val_batches:
                if accelerator.is_local_main_process:
                    val_progress_bar.close()
                self.val_dataloader.end() # type: ignore
                break

        avg_val_loss = torch.cat(val_loss_list).mean().item()
        action_l1_err_list = np.concatenate(action_l1_err_list, axis=0)  # (len_val_dataset*Ta, Da)
        action_l1_err_list_denormed = (
            self.maxmin.denormalize_L1_action_err( # type: ignore FIXME
                action_l1_err_list
            )
        )

        # action L1 errors
        avg_action_errors_denormed = action_l1_err_list_denormed.mean(0)  # (Da,) NOTE only if the error is L1 (linear)
        # Define dimension splits: hand_joints(14) + arm_joints(14) + rpy(3) + height(1) = 32
        hand_joints_start, hand_joints_end = 0, 14
        arm_joints_start, arm_joints_end = 14, 28
        rpy_start, rpy_end = 28, 31
        height_start, height_end = 31, 32
        torso_vx_start, torso_vx_end = 32, 33
        torso_vy_start, torso_vy_end = 33, 34
        torso_vyaw_start, torso_vyaw_end = 34, 35
        torso_dyaw_start, torso_dyaw_end = 35, 36
    
        labels_denormed = [
            "err_l1_hand_joints",
            "err_l1_arm_joints",
            "err_l1_torso_rpy",
            "err_l1_height",
            "err_l1_vx",
            "err_l1_vy",
            "err_l1_vyaw",
            "err_l1_target_yaw",
        ]
    
        avg_lr_action_err_denormed = np.split(
            avg_action_errors_denormed, [hand_joints_end, arm_joints_end, rpy_end, height_end, torso_vx_end, torso_vy_end, torso_vyaw_end], axis=-1
        )

        # log metrics
        return {
            "loss": avg_val_loss,
            **dict(zip(labels_denormed, map(np.linalg.norm, avg_lr_action_err_denormed)))
        }

    def forward_and_loss(self, model, batch) -> dict[str, torch.Tensor]:
        bsz, Tp, Da = batch["actions"].shape

        if self.noise_scheduler_name == "ddpm":
            timesteps = torch.randint(
                low=0, high=self.train_diffusion_steps, size=(bsz,), device=self.device
            ).long()

        elif self.noise_scheduler_name == "flow":
            # TODO check how pi_0 does this ?
            sigmas = torch.rand((bsz,), device=self.device)
            timesteps = sigmas * self.train_diffusion_steps

            if self.model_cfg.rtc:
                delay = torch.randint(
                    low=0,
                    high=self.model_cfg.max_delay,
                    size=(bsz,),
                    device=self.device,
                    dtype=torch.long
                )
                prefix_mask = torch.arange(Tp, device=self.device)[None, :] < delay[:, None]
                sigmas = torch.where(
                    prefix_mask,
                    torch.tensor(0.0, device=self.device, dtype=sigmas.dtype),
                    sigmas[:, None]
                )
                timesteps = sigmas * self.train_diffusion_steps

        else:
            raise ValueError(f"Unknown noise scheduler: {self.noise_scheduler_name}")

        noise_action = torch.randn_like(batch["actions"])
        if self.noise_scheduler_name == "ddpm":
            noisy_actions = self.noise_scheduler.add_noise(
                batch["actions"], noise_action, timesteps # type: ignore
            )
            target_action = noise_action  # ddpm epsilon-prediction
        elif self.noise_scheduler_name == "flow":
            sigmas_action = sigmas  # type: ignore
            while len(sigmas_action.shape) < len(batch["actions"].shape):
                sigmas_action = sigmas_action.unsqueeze(-1)  # Expand to B, Tp, action_dim 
            noisy_actions = (1 - sigmas_action) * batch["actions"] + sigmas_action * noise_action
            target_action = noise_action - batch["actions"] # flow-matching velocity prediction
        else:
            raise ValueError

        model_output = model(
            input_ids=batch["input_ids"],#####
            attention_mask=batch["attention_mask"], # vlm related
            pixel_values=batch["pixel_values"],
            image_grid_thw=batch["image_grid_thw"], ####
            action_samples=noisy_actions,  # (B,Tp,Da)
            states=batch["states"],  # (B,1,M)
            timestep=timesteps,
            traj2ds=batch["traj2ds"] if "traj2ds" in batch else None,  # (B, C, 3, H, W)
            return_dict=True,
        )
        action_pred = model_output.action
        loss_action = F.mse_loss(
            action_pred.float(), target_action.float(), reduction="none"
        )  # (B, Tp, Da)
        """  
            sum(1) -> to keep gradient consistent when training with different action chunks
            mean(0) -> to average over the batch
            sum() -> to sum over the weighted loss of each action dimension
        """
        mask = batch["actions_mask"].float() if "actions_mask" in batch else torch.ones_like(batch["actions"])
        if self.model_cfg.rtc:
            postfix_mask = (~prefix_mask)[:, :, None].float() # type: ignore  (B, Tp, 1)  
            mask = mask * postfix_mask
        
        loss_action = (loss_action * mask).sum(1)  # (B, Da)
        loss_action = (loss_action.mean(0) * self.loss_w).sum()
        return {"loss": loss_action}

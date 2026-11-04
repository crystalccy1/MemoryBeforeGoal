# Derived from Psi0 (https://github.com/physical-superintelligence-lab/Psi0, Apache License 2.0) and modified; see NOTICE.
from __future__ import annotations
from typing import Union, Optional, List, Any, TYPE_CHECKING
from pathlib import Path
import torch
import math
from copy import copy
from abc import ABC, abstractmethod
if TYPE_CHECKING:
    from psi.config.config import LaunchConfig, TrainConfig, LoggingConfig
    from psi.config.config import  DataConfig
from accelerate import Accelerator
from transformers.trainer_utils import PredictionOutput
from transformers.optimization import get_scheduler
from torch.optim import Optimizer
from psi.utils import snake_to_pascal
import os
import re
import importlib
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import numpy as np
import datetime
import accelerate
import random
import shutil
import json
import hashlib

from psi.utils import initialize_overwatch
overwatch = initialize_overwatch(__name__)


def _checkpoint_file_identity(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    """Measure one immutable regular-file payload for a checkpoint manifest."""
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Checkpoint artifact is missing or unsafe: {path}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        while chunk := stream.read(8 * 1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
        after = os.fstat(stream.fileno())
    stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in stable):
        raise RuntimeError(f"Checkpoint artifact changed while hashing: {path}")
    if size != before.st_size or size <= 0:
        raise RuntimeError(f"Checkpoint artifact is empty or unstable: {path}")
    return {
        "path": (
            path.relative_to(relative_to).as_posix()
            if relative_to is not None
            else str(path.resolve())
        ),
        "bytes": size,
        "sha256": digest.hexdigest(),
    }


def _canonical_json_sha256(value: Any) -> str:
    """Hash one JSON value using the preflight contract's canonical form."""
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def worker_init_fn(worker_id):
    # print(f"worker_init_fn called by worker {worker_id}")
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

class Trainer(ABC):
    cfg: LaunchConfig
    model: Any
    optimizer: torch.optim.Optimizer 
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler

    train_dataset: Dataset
    val_dataset: Optional[Dataset]

    def __init__(self, cfg: LaunchConfig, device: Union[torch.device, int]):
        self.device = torch.device(device)
        self.cfg = cfg

        # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
        if cfg.train.mixed_precision == "fp16":
            self.dtype = torch.float16
        elif cfg.train.mixed_precision == "bf16":
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32
        
        # avoid duplicate run names in a real training run
        # should read the timestamp from command line args instead
        # self.timestamp = datetime.datetime.now().strftime("%y%m%d%H%M")
        self.timestamp = self.cfg.timestamp

    @classmethod
    def instantiate(
        cls, cfg: LaunchConfig, device: Union[torch.device, int]
    ) -> "Trainer":
        trainer_name = cfg.train.name
        try:
            parts = re.split(r"[-_]", trainer_name)
            trainer_name = "_".join([p.lower() for p in parts])
            module = importlib.import_module(f"psi.trainers.{trainer_name}")
            trainer_clazz = getattr(module, f"{snake_to_pascal(trainer_name)}Trainer")
        except Exception as e:
            raise ValueError(
                f"fail to import {trainer_name} from psi.trainers"
            ) from e

        return trainer_clazz(cfg, device)

    @property
    def train_cfg(self) -> TrainConfig:
        return self.cfg.train

    @property
    def log_cfg(self) -> LoggingConfig:
        return self.cfg.log

    @property
    def data_cfg(self) -> DataConfig:
        return self.cfg.data

    @property
    def hf_token(self):
        return self.cfg.train.hf_token.read_text().strip() \
            if isinstance(self.cfg.train.hf_token, Path) \
            else os.environ[self.cfg.train.hf_token]

    def get_fsdp_plugin(self) -> accelerate.utils.FullyShardedDataParallelPlugin | None: ...

    def create_datasets(self) -> tuple[Dataset, Dataset|None]: 
        ...

    def create_dataloaders(
        self, train_dataset, val_dataset
    ) -> tuple[DataLoader, DataLoader|None]: ...

    def create_optimizer(self):
        optimizer_kwargs = dict(self.cfg.train.lr_scheduler_kwargs)
        if self.cfg.train.optimizer_foreach is not None:
            optimizer_kwargs["foreach"] = self.cfg.train.optimizer_foreach
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.train.learning_rate,
            **optimizer_kwargs, # type: ignore
        )

    def create_scheduler(
        self, num_training_steps: int | None = None, optimizer: Optimizer | None = None
    ):
        if num_training_steps is None:
            num_training_steps = self.max_training_steps

        # Creates Dummy Scheduler if `scheduler` was specified in the config file else creates `args.lr_scheduler_type` Scheduler
        if (
            self.accelerator.state.deepspeed_plugin is None
            or "scheduler" not in self.accelerator.state.deepspeed_plugin.deepspeed_config
        ):
            self.lr_scheduler = get_scheduler(
                name=self.cfg.train.lr_scheduler_type,
                optimizer=optimizer if optimizer is not None else self.optimizer,
                num_warmup_steps=self.num_warmup_steps * self.world_size,
                num_training_steps=num_training_steps * self.world_size,
                scheduler_specific_kwargs=self.cfg.train.scheduler_specific_kwargs,
            )
        else:
            self.lr_scheduler = accelerate.utils.DummyScheduler(
                optimizer, total_num_steps=self.max_training_steps, warmup_num_steps=self.num_warmup_steps
            ) # type: ignore

        return self.lr_scheduler

    def log(
        self, metrics: dict[str, Any], start_time: Optional[float] = None
    ) -> None: 
        wandb_dict = copy(metrics)
        for k,v in list(wandb_dict.items()):
            if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray):
                wandb_dict[k] = v.item()
        # wandb_dict.update({"train/grad_norm": grad_norm})
        # wandb_dict.update(self.get_log_kv())
        self.accelerator.log(wandb_dict, step=self.global_step) # WANDB logging
        if (
            os.getenv("PSI_LOCAL_METRICS_JSONL", "0") == "1"
            and self.accelerator.is_main_process
        ):
            metrics_path = os.path.join(self.project_dir, "metrics.jsonl")
            os.makedirs(self.project_dir, exist_ok=True)
            if not hasattr(self, "_local_metrics_next_event_index"):
                next_event_index = 0
                if os.path.exists(metrics_path):
                    with open(metrics_path, encoding="utf-8") as existing:
                        for line_number, line in enumerate(existing, start=1):
                            try:
                                prior = json.loads(line)
                            except json.JSONDecodeError as exc:
                                raise RuntimeError(
                                    "Existing local metrics are not valid JSONL: "
                                    f"{metrics_path}:{line_number}"
                                ) from exc
                            if prior.get("event_index") != next_event_index:
                                raise RuntimeError(
                                    "Existing local metric event indices are not "
                                    f"contiguous at {metrics_path}:{line_number}"
                                )
                            next_event_index += 1
                self._local_metrics_next_event_index = next_event_index
            namespaces = sorted(
                {
                    key.split("/", 1)[0] if "/" in key else "global"
                    for key in wandb_dict
                }
            )
            record = {
                "event_index": self._local_metrics_next_event_index,
                "step": self.global_step,
                "completed_updates": self.global_step + 1,
                "stream": namespaces[0] if len(namespaces) == 1 else "mixed",
                "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(
                    timespec="seconds"
                ),
                **wandb_dict,
            }
            with open(metrics_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._local_metrics_next_event_index += 1

    def write_parameter_audit(self) -> Path:
        """Publish the immutable frozen/trainable roster before optimization.

        All ranks independently construct and hash the roster.  Rank zero only
        publishes it after the distributed digests agree, and a resumed
        attempt must reproduce the existing artifact byte-for-byte.
        """

        trainable: list[dict[str, Any]] = []
        frozen: list[dict[str, Any]] = []
        for name, parameter in sorted(
            self.model.named_parameters(), key=lambda item: item[0]
        ):
            if not name or "\x00" in name:
                raise RuntimeError(f"Unsafe model parameter name: {name!r}")
            numel = int(parameter.numel())
            if numel <= 0:
                raise RuntimeError(f"Empty model parameter cannot be audited: {name}")
            record = {
                "name": name,
                "numel": numel,
                "dtype": str(parameter.dtype),
                "requires_grad": bool(parameter.requires_grad),
            }
            (trainable if parameter.requires_grad else frozen).append(record)

        if not trainable or not frozen:
            raise RuntimeError(
                "Parameter audit requires non-empty trainable and frozen rosters"
            )
        trainable_names = {record["name"] for record in trainable}
        frozen_names = {record["name"] for record in frozen}
        if trainable_names & frozen_names:
            raise RuntimeError("Parameter audit rosters overlap")

        tune_vlm = bool(getattr(self.cfg.model, "tune_vlm"))
        if not tune_vlm:
            if any(name.startswith("vlm_model.") for name in trainable_names):
                raise RuntimeError("tune_vlm=false but a VLM parameter is trainable")
            if not any(name.startswith("vlm_model.") for name in frozen_names):
                raise RuntimeError("Parameter audit found no frozen VLM parameter")
        if not any(name.startswith("action_header.") for name in trainable_names):
            raise RuntimeError("Parameter audit found no trainable action_header parameter")

        rosters = {"trainable": trainable, "frozen": frozen}
        summary = {
            kind: {
                "parameter_count": len(records),
                "numel": sum(record["numel"] for record in records),
                "roster_sha256": _canonical_json_sha256(records),
            }
            for kind, records in rosters.items()
        }
        payload = {
            "schema_version": 1,
            "artifact_type": "psi0_xmovepick_parameter_audit",
            "formal_eligible": os.environ.get("PSI_FORMAL_ELIGIBLE") == "1",
            "train_seed": self.cfg.seed,
            "audit_point": (
                "after_model_construction_before_first_optimizer_step"
            ),
            "tune_vlm": tune_vlm,
            "parameters": rosters,
            "summary": summary,
        }
        document = {
            **payload,
            "parameter_audit_payload_sha256": _canonical_json_sha256(payload),
        }
        serialized = (
            json.dumps(
                document,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")

        local_digest = hashlib.sha256(serialized).hexdigest()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank_digests: list[str | None] = [
                None for _ in range(torch.distributed.get_world_size())
            ]
            torch.distributed.all_gather_object(rank_digests, local_digest)
            if any(digest != local_digest for digest in rank_digests):
                raise RuntimeError(
                    "Parameter audit differs across distributed ranks: "
                    f"{rank_digests}"
                )

        audit_path = Path(self.project_dir) / "parameter_audit.json"
        publication_error: list[str | None] = [None]
        if self.accelerator.is_main_process:
            try:
                audit_path.parent.mkdir(parents=True, exist_ok=True)
                if audit_path.exists() or audit_path.is_symlink():
                    if audit_path.is_symlink() or not audit_path.is_file():
                        raise RuntimeError(
                            f"Existing parameter audit is unsafe: {audit_path}"
                        )
                    if audit_path.read_bytes() != serialized:
                        raise RuntimeError(
                            "Parameter audit drift detected across attempts: "
                            f"{audit_path}"
                        )
                else:
                    stage = audit_path.parent / (
                        f".{audit_path.name}.{os.getpid()}.incomplete"
                    )
                    if os.path.lexists(stage):
                        raise RuntimeError(
                            f"Stale parameter-audit stage exists: {stage}"
                        )
                    try:
                        with stage.open("xb") as stream:
                            stream.write(serialized)
                            stream.flush()
                            os.fsync(stream.fileno())
                        os.replace(stage, audit_path)
                        directory_fd = os.open(audit_path.parent, os.O_RDONLY)
                        try:
                            os.fsync(directory_fd)
                        finally:
                            os.close(directory_fd)
                    finally:
                        if stage.exists():
                            stage.unlink()
            except Exception as exc:
                publication_error[0] = f"{type(exc).__name__}: {exc}"

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast_object_list(publication_error, src=0)
        if publication_error[0] is not None:
            raise RuntimeError(
                f"Parameter audit publication failed: {publication_error[0]}"
            )
        self.accelerator.wait_for_everyone()
        return audit_path

    def create_optimizer_and_scheduler(self, num_training_steps: int | None = None):
        optimizer = self.create_optimizer()
        self.create_scheduler(
            num_training_steps=num_training_steps, optimizer=optimizer
        )
    def compute_loss(
        self, batch#, return_outputs=False#, num_items_in_batch=None
    ) -> dict: ...

    def training_step(
        self,
        # model: nn.Module,
        batch: dict[str, Union[torch.Tensor, Any]],
    ) -> tuple[bool, dict[str, Any]]: ...

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> tuple[
        Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]
    ]: ...

    def evaluate(
        self,
        # eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        # ignore_keys: Optional[list[str]] = None,
        # metric_key_prefix: str = "eval",
    ) -> dict[str, float] | None: 
        ...

    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "test",
    ) -> PredictionOutput: ...

    @property
    def world_size(self) -> int:
        """Returns the number of processes in the current distributed group."""
        return overwatch.world_size()

    @property
    def device_train_batch_size(self) -> int:
        """Returns the training batch size per process."""
        return self.cfg.train.train_batch_size

    @property
    def gradient_accumulation_steps(self) -> int:
        """Returns the number of gradient accumulation steps."""
        return self.cfg.train.gradient_accumulation_steps

    @property
    def global_train_batch_size(self) -> int:
        """Returns the global training batch size."""
        return (
            self.device_train_batch_size
            * self.world_size
            * self.gradient_accumulation_steps
        )

    @property
    def len_train_dataset(self) -> int:
        """Returns the length of the training dataset."""
        if hasattr(self, "_len_train_dataset") and self._len_train_dataset: # type: ignore
            return self._len_train_dataset # type: ignore
        return len(self.train_dataset) # type: ignore

    @len_train_dataset.setter
    def len_train_dataset(self, length: int):
        self._len_train_dataset = length

    @property
    def len_val_dataset(self) -> int:
        """Returns the length of the training dataset."""
        if hasattr(self, "_len_train_dataset") and self._len_val_dataset: # type: ignore
            return self._len_val_dataset # type: ignore
        return len(self.val_dataset) # type: ignore
    
    @len_val_dataset.setter
    def len_val_dataset(self, length: int):
        self._len_val_dataset = length

    @property
    def len_train_dataloader(self) -> int:
        total = self.len_train_dataset / (
            self.device_train_batch_size * self.world_size
        )
        if hasattr(self, "train_dataloader_drop_last") and self.train_dataloader_drop_last: # type: ignore
            return math.floor(total)  # drop the last incomplete batch
        else:
            return math.ceil(total)  # include the last incomplete batch
        

    @property
    def len_val_dataloader(self) -> int:
        if not hasattr(self, "val_dataloader") or self.val_dataloader is None:
            return 0
        total = self.len_val_dataset / (
            self.device_train_batch_size * self.world_size
        )
        if hasattr(self, "val_dataloader_drop_last") and self.val_dataloader_drop_last: # type: ignore
            return math.floor(total)  # drop the last incomplete batch
        else:
            return math.ceil(total)  # include the last incomplete batch

    @property
    def num_steps_per_epoch(self) -> int:
        """number of global steps (sync gradients) per epoch"""
        # assert self.len_train_dataloader > self.gradient_accumulation_steps
        return max(self.len_train_dataloader // self.gradient_accumulation_steps, 1)

    @property
    def max_training_steps(self) -> int:
        """Returns the maximum number of training steps."""
        if self.cfg.train.max_training_steps is not None:
            return self.cfg.train.max_training_steps
        else:
            assert self.cfg.train.num_train_epochs is not None
            return self.num_steps_per_epoch * self.world_size * self.cfg.train.num_train_epochs

    @property
    def max_training_epochs(self) -> int:
        return math.ceil(self.max_training_steps / self.num_steps_per_epoch)

    @property
    def num_warmup_steps(self) -> int:
        """
        Get number of steps used for a linear warmup.
        """
        warmup_steps = (
            self.cfg.train.warmup_steps
            if self.cfg.train.warmup_steps is not None
            else math.ceil(self.max_training_steps * self.cfg.train.warmup_ratio)  # type: ignore
        )
        return warmup_steps # because accelerate divides the steps by world_size when preparing the scheduler

    @property
    @abstractmethod
    def task_run_name(self) -> str:
        """Returns the task-specific run name."""

    @property
    def run_name(self) -> str:
        if re.fullmatch(r"r-[0-9a-f]{12}", self.cfg.exp):
            return self.cfg.exp
        run_name = (
            f"{self.cfg.exp}{self.task_run_name}"
            f".b{self.global_train_batch_size}.gpus{overwatch.world_size()}"
        )

        run_name = f"{run_name}.{self.timestamp}"
        if self.cfg.debug:
            run_name = f"debug-{run_name}"
        return run_name

    @property
    def project_dir(self) -> str:
        """Returns the project directory for saving checkpoints and logs."""
        return os.path.join(
            self.cfg.train.output_dir, self.cfg.train.name, self.run_name
        )

    def next_epoch(self, epoch):
        """called between epochs, e.g. for resetting the distributed samplers"""
        if hasattr(self, "train_sampler") and self.train_sampler is not None: # type: ignore
            self.train_sampler.set_epoch(epoch) # type: ignore

    @abstractmethod
    def init_models(self):
        """Initialize the models for training."""

    # @abstractmethod
    def set_train(self):
        self.model.train()

    # @abstractmethod
    def set_eval(self):
        self.model.eval()

    # @abstractmethod
    # def create_optimizers(self):
    #     ...

    # @abstractmethod
    # def create_lr_schedulers(self):
    #     ...

    # @abstractmethod
    # def train_one_step(self, batch_input, global_step, local_step, accelerator=None):
    #     ...

    def step(self, batch_input, global_step, local_step) -> tuple[bool, dict[str, Any]]:
        """ Perform a single training step. """

        self.local_step = local_step
        self.global_step = global_step

        sync_gradients, losses = self.training_step(batch_input)
        return sync_gradients, losses

    @abstractmethod
    def prepare(self, accelerator: Accelerator) -> DataLoader:
        self.optimizer, self.lr_scheduler = accelerator.prepare(
            self.optimizer, self.lr_scheduler
        )

        self.train_dataloader = accelerator.prepare(self.train_dataloader)

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


        # FIXME if self.train_dataloader.dataset is IterableDataset:
        # assert (
        #     abs(len(self.train_dataloader) - self.len_train_dataloader)
        #     <= 1  # because of drop_last option is lost in prepare.
        # ), f"check calculations again, {len(self.train_dataloader)} != {self.len_train_dataloader}"

        val_dataloader = getattr(self, "val_dataloader", None)
        if val_dataloader is not None: # not using if self.val_dataloader to avoid DataLoader.__len__() being called on iterable dataset
            self.val_dataloader = accelerator.prepare(self.val_dataloader)
        """ NOTE:
            We have to manually record how many steps we have accumulated gradients
            because the accelerator does not do it for us.
        """
        self.accelerator = accelerator
        return self.train_dataloader # type: ignore

    def save_checkpoint(self, global_step: int) -> str | None:
        save_dir = os.path.join(self.project_dir, "checkpoints")
        ckpt_name = f"ckpt_{global_step}"
        ckpt_dir = os.path.join(save_dir, ckpt_name)
        attempt_id = os.environ.get("PSI_ATTEMPT_ID") or (
            f"timestamp-{self.cfg.timestamp}"
        )
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", attempt_id):
            raise ValueError(f"Invalid checkpoint attempt identity: {attempt_id!r}")
        try:
            additional_output_binding_paths = json.loads(
                os.environ.get(
                    "PSI_ADDITIONAL_OUTPUT_CHECKPOINT_BINDINGS_JSON", "{}"
                )
            )
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "PSI_ADDITIONAL_OUTPUT_CHECKPOINT_BINDINGS_JSON must be a JSON object"
            ) from exc
        if not isinstance(additional_output_binding_paths, dict) or any(
            not isinstance(role, str)
            or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", role) is None
            or not isinstance(path, str)
            or not path
            for role, path in additional_output_binding_paths.items()
        ):
            raise RuntimeError(
                "PSI_ADDITIONAL_OUTPUT_CHECKPOINT_BINDINGS_JSON contains an "
                "invalid role/path"
            )
        additional_output_required_roles = {
            role.strip()
            for role in os.environ.get(
                "PSI_ADDITIONAL_OUTPUT_CHECKPOINT_BINDING_ROLES", ""
            ).split(",")
            if role.strip()
        }
        if set(additional_output_binding_paths) != additional_output_required_roles:
            raise RuntimeError(
                "Additional output checkpoint binding roles must exactly match "
                "PSI_ADDITIONAL_OUTPUT_CHECKPOINT_BINDINGS_JSON"
            )
        if (
            os.environ.get("PSI_FORMAL_ELIGIBLE") == "1"
            and additional_output_binding_paths
        ):
            raise RuntimeError(
                "Formal training forbids additional output-only checkpoint bindings"
            )
        os.makedirs(save_dir, exist_ok=True)
        stage_dir = os.path.join(
            save_dir, f".{ckpt_name}.{attempt_id}.incomplete"
        )

        # A full checkpoint is published only by an atomic directory rename.
        # Interrupted stages remain immutable evidence and never become a
        # resume candidate merely because Accelerate wrote a subset of files.
        self.accelerator.wait_for_everyone()
        if os.path.lexists(ckpt_dir):
            raise FileExistsError(f"Refusing to replace checkpoint: {ckpt_dir}")
        if os.path.lexists(stage_dir):
            raise FileExistsError(
                f"Refusing stale checkpoint stage: {stage_dir}"
            )
        if self.accelerator.is_main_process:
            os.makedirs(stage_dir, exist_ok=False)
        self.accelerator.wait_for_everyone()

        self.accelerator.save_state(stage_dir)
        self.accelerator.wait_for_everyone()

        rank = self.accelerator.process_index
        world_size = self.accelerator.num_processes
        prepared_batches = len(self.train_dataloader)
        accumulation = self.gradient_accumulation_steps
        steps_per_epoch = (prepared_batches + accumulation - 1) // accumulation
        completed_epochs, steps_into_epoch = divmod(global_step, steps_per_epoch)
        skip_batches = steps_into_epoch * accumulation
        epoch_zero_state = getattr(self, "_train_sampler_epoch_zero_state", None)
        epoch_zero_sha256 = None
        if isinstance(epoch_zero_state, torch.Tensor):
            epoch_zero_sha256 = hashlib.sha256(
                epoch_zero_state.detach().cpu().contiguous().numpy().tobytes()
            ).hexdigest()
        sampler_seed = getattr(self, "_train_sampler_seed", None)
        worker_base_seed = getattr(self, "_train_legacy_worker_base_seed", None)
        sampler_record = {
            "schema_version": 1,
            "protocol": "legacy-shared-generator-split-v1",
            "checkpoint_step": global_step,
            "rank": rank,
            "world_size": world_size,
            "train_seed": int(self.cfg.seed),
            "sampler_seed": sampler_seed,
            "legacy_initial_worker_base_seed": worker_base_seed,
            "sampler_epoch_zero_state_sha256": epoch_zero_sha256,
            "prepared_batches_per_epoch": prepared_batches,
            "steps_per_epoch": steps_per_epoch,
            "gradient_accumulation_steps": accumulation,
            "resume_completed_epochs": completed_epochs,
            "resume_batch_offset": skip_batches,
            "sample_index_continuity": "exact",
            "persistent_worker_augmentation_rng_continuity": "not_restored",
        }
        if os.environ.get("PSI_FORMAL_ELIGIBLE") == "1" and (
            not isinstance(sampler_seed, int)
            or not isinstance(worker_base_seed, int)
            or epoch_zero_sha256 is None
        ):
            raise RuntimeError(
                "Formal checkpoint lacks the exact finetune sampler evidence"
            )
        rank_path = Path(stage_dir) / f"sampler_resume_rank_{rank}.json"
        rank_tmp = rank_path.with_suffix(".json.tmp")
        rank_tmp.write_text(
            json.dumps(sampler_record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(rank_tmp, rank_path)
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            rank_states = []
            for expected_rank in range(world_size):
                state_path = (
                    Path(stage_dir)
                    / f"sampler_resume_rank_{expected_rank}.json"
                )
                if not state_path.is_file():
                    raise RuntimeError(
                        f"Missing sampler resume evidence for rank {expected_rank}"
                    )
                state_document = json.loads(state_path.read_text(encoding="utf-8"))
                if state_document.get("rank") != expected_rank:
                    raise RuntimeError(
                        f"Sampler evidence rank mismatch in {state_path}"
                    )
                state_digest = hashlib.sha256(state_path.read_bytes()).hexdigest()
                rank_states.append(
                    {
                        "rank": expected_rank,
                        "sampler_seed": state_document.get("sampler_seed"),
                        "state": {
                            "path": state_path.name,
                            "bytes": state_path.stat().st_size,
                            "sha256": state_digest,
                        },
                    }
                )
            sampler_manifest = {
                "schema_version": 1,
                "protocol": "legacy-shared-generator-split-v1",
                "checkpoint_step": global_step,
                "world_size": world_size,
                "rank_states": rank_states,
            }
            sampler_manifest_path = Path(stage_dir) / "sampler_resume_manifest.json"
            sampler_manifest_tmp = sampler_manifest_path.with_suffix(".json.tmp")
            sampler_manifest_tmp.write_text(
                json.dumps(sampler_manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(sampler_manifest_tmp, sampler_manifest_path)

            stage_path = Path(stage_dir)
            source_run_path = Path(self.project_dir).resolve()
            run_config_path = source_run_path / "run_config.json"
            live_metrics_path = source_run_path / "metrics.jsonl"
            statistics_path = source_run_path / "dataset_statistics.json"
            metrics_prefix_path = stage_path / "metrics_prefix.jsonl"
            if live_metrics_path.is_file() and not live_metrics_path.is_symlink():
                with live_metrics_path.open("rb") as source, metrics_prefix_path.open(
                    "xb"
                ) as destination:
                    while chunk := source.read(8 * 1024 * 1024):
                        destination.write(chunk)
                    destination.flush()
                    os.fsync(destination.fileno())

            required_payload_names = {
                "model.safetensors",
                "optimizer.bin",
                "scheduler.bin",
                "sampler_resume_manifest.json",
                *(f"random_states_{rank}.pkl" for rank in range(world_size)),
                *(f"sampler_resume_rank_{rank}.json" for rank in range(world_size)),
            }
            actual_payload_names = {
                path.name
                for path in stage_path.iterdir()
                if path.is_file() and not path.name.endswith(".tmp")
            }
            missing_payload_names = required_payload_names - actual_payload_names
            if missing_payload_names:
                raise RuntimeError(
                    "Full checkpoint is missing required payloads: "
                    f"{sorted(missing_payload_names)}"
                )
            payload_inventory = {
                name: _checkpoint_file_identity(
                    stage_path / name, relative_to=stage_path
                )
                for name in sorted(actual_payload_names)
            }
            model_identity = payload_inventory["model.safetensors"]
            model_path = stage_path / "model.safetensors"
            model_sha256 = model_identity["sha256"]

            require_bindings = os.environ.get("PSI_REQUIRE_CHECKPOINT_BINDINGS") == "1"
            if require_bindings:
                for label, path in (
                    ("run_config", run_config_path),
                    ("checkpoint-local metrics prefix", metrics_prefix_path),
                    ("dataset statistics", statistics_path),
                ):
                    if not path.is_file() or path.is_symlink():
                        raise RuntimeError(
                            f"Formal checkpoint lacks {label} binding: {path}"
                        )
            local_bindings = {}
            for label, path in (
                ("run_config", run_config_path),
                ("dataset_statistics", statistics_path),
            ):
                if path.is_file() and not path.is_symlink():
                    local_bindings[label] = _checkpoint_file_identity(path)
            if metrics_prefix_path.is_file() and not metrics_prefix_path.is_symlink():
                local_bindings["metrics_prefix"] = _checkpoint_file_identity(
                    metrics_prefix_path, relative_to=stage_path
                )

            raw_external_bindings = os.environ.get(
                "PSI_CHECKPOINT_BINDINGS_JSON", "{}"
            )
            try:
                external_binding_paths = json.loads(raw_external_bindings)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "PSI_CHECKPOINT_BINDINGS_JSON must be a JSON object"
                ) from exc
            if not isinstance(external_binding_paths, dict) or any(
                not isinstance(role, str)
                or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", role) is None
                or not isinstance(path, str)
                or not path
                for role, path in external_binding_paths.items()
            ):
                raise RuntimeError(
                    "PSI_CHECKPOINT_BINDINGS_JSON contains an invalid role/path"
                )
            required_binding_roles = {
                role.strip()
                for role in os.environ.get(
                    "PSI_REQUIRED_CHECKPOINT_BINDING_ROLES", ""
                ).split(",")
                if role.strip()
            }
            if require_bindings and not required_binding_roles:
                raise RuntimeError(
                    "Formal checkpoint requires explicit binding roles"
                )
            if not required_binding_roles.issubset(external_binding_paths):
                raise RuntimeError(
                    "Formal checkpoint external bindings are missing roles: "
                    f"{sorted(required_binding_roles - set(external_binding_paths))}"
                )
            overlapping_output_roles = set(external_binding_paths) & set(
                additional_output_binding_paths
            )
            if overlapping_output_roles:
                raise RuntimeError(
                    "Additional output checkpoint bindings overlap base roles: "
                    f"{sorted(overlapping_output_roles)}"
                )
            external_binding_paths = {
                **external_binding_paths,
                **additional_output_binding_paths,
            }
            required_binding_roles |= additional_output_required_roles
            external_bindings = {
                role: _checkpoint_file_identity(Path(path).expanduser().resolve())
                for role, path in sorted(external_binding_paths.items())
            }
            final_dir_abs = os.path.abspath(ckpt_dir)
            source_run_abs = os.path.abspath(self.project_dir)
            checkpoint_identity = {
                "schema_version": 1,
                "protocol": "full-checkpoint-atomic-resume-v1",
                "checkpoint_id": (
                    f"{Path(self.project_dir).name}/{ckpt_name}"
                    f"@sha256:{model_sha256}"
                ),
                "step": global_step,
                "model_sha256": model_sha256,
                "model_bytes": model_identity["bytes"],
                "source_run": source_run_abs,
                "train_seed": int(self.cfg.seed),
                "formal_eligible": (
                    os.environ.get("PSI_FORMAL_ELIGIBLE") == "1"
                ),
                "world_size": world_size,
                "attempt_id": attempt_id,
                "COMPLETE": global_step,
                "complete_marker_path": os.path.join(final_dir_abs, "COMPLETE"),
                "model_path": os.path.join(final_dir_abs, "model.safetensors"),
                "sampler_manifest_path": os.path.join(
                    final_dir_abs, "sampler_resume_manifest.json"
                ),
                "payload_inventory": payload_inventory,
                "local_bindings": local_bindings,
                "external_bindings": external_bindings,
                "required_external_binding_roles": sorted(required_binding_roles),
                "created_at": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(timespec="seconds"),
            }
            identity_path = Path(stage_dir) / "checkpoint_manifest.json"
            identity_tmp = identity_path.with_suffix(".json.tmp")
            identity_tmp.write_text(
                json.dumps(checkpoint_identity, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(identity_tmp, identity_path)

            # A marker prevents an interrupted checkpoint directory from being
            # mistaken for a valid latest checkpoint on the next launch.
            with open(os.path.join(stage_dir, "COMPLETE"), "x") as f:
                f.write(f"{global_step}\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(stage_dir, ckpt_dir)
            print(
                "FULL_CHECKPOINT_IDENTITY_JSON="
                + json.dumps(
                    checkpoint_identity, sort_keys=True, separators=(",", ":")
                ),
                flush=True,
            )

            # Keep only the latest max_checkpoints_to_keep checkpoints
            max_to_keep = self.train_cfg.max_checkpoints_to_keep or 100
            if max_to_keep is not None and max_to_keep > 0:
                # List all checkpoint directories matching ckpt_*
                ckpt_dirs = [d for d in os.listdir(save_dir) if d.startswith("ckpt_") and os.path.isdir(os.path.join(save_dir, d))]
                # Extract step numbers and sort by step (assume ckpt_{step})
                def extract_step(d):
                    try:
                        return int(d.split("ckpt_")[-1])
                    except Exception:
                        return -1
                ckpt_dirs_sorted = sorted(ckpt_dirs, key=extract_step, reverse=True)
                # Remove older checkpoints if exceeding max_to_keep
                for old_ckpt in ckpt_dirs_sorted[max_to_keep:]:
                    old_ckpt_path = os.path.join(save_dir, old_ckpt)
                    if "0000" in old_ckpt:
                        # force keeping ckpt saved every 10k
                        continue
                    try:
                        shutil.rmtree(old_ckpt_path)
                        overwatch.info(f"Removed old checkpoint: {old_ckpt_path}")
                    except Exception as e:
                        overwatch.warning(f"Failed to remove old checkpoint {old_ckpt_path}: {e}")

        self.accelerator.wait_for_everyone()
        return ckpt_dir

    @staticmethod
    def _rng_state_sha256() -> str:
        """Hash all process-local RNG streams without advancing any of them."""
        digest = hashlib.sha256()
        digest.update(repr(random.getstate()).encode("utf-8"))
        numpy_state = np.random.get_state()
        digest.update(str(numpy_state[0]).encode("utf-8"))
        digest.update(numpy_state[1].tobytes())
        digest.update(repr(numpy_state[2:]).encode("utf-8"))
        digest.update(torch.random.get_rng_state().cpu().numpy().tobytes())
        if torch.cuda.is_available():
            for state in torch.cuda.get_rng_state_all():
                digest.update(state.cpu().numpy().tobytes())
        return digest.hexdigest()

    def save_model_only_snapshot(self, global_step: int) -> str:
        """Atomically save an evaluation-only model without touching RNG state.

        Unlike ``save_checkpoint``, this intentionally excludes optimizer,
        scheduler and RNG payloads.  It is therefore not a resume point.  The
        before/after digest makes RNG neutrality a fail-closed invariant on
        every distributed rank.
        """
        snapshot_root = os.path.join(self.project_dir, "model_snapshots")
        final_dir = os.path.join(snapshot_root, f"ckpt_{global_step}")
        stage_name = f".ckpt_{global_step}.incomplete"
        stage_dir = os.path.join(snapshot_root, stage_name)

        if self.accelerator.is_main_process:
            if os.path.lexists(final_dir):
                raise FileExistsError(
                    f"Refusing to replace model-only snapshot: {final_dir}"
                )
            os.makedirs(snapshot_root, exist_ok=True)
            if os.path.lexists(stage_dir):
                raise FileExistsError(
                    f"Refusing stale model-only snapshot stage: {stage_dir}"
                )
            os.makedirs(stage_dir, exist_ok=False)
        self.accelerator.wait_for_everyone()

        rng_before = self._rng_state_sha256()
        self.accelerator.save_model(
            self.model,
            stage_dir,
            safe_serialization=True,
            max_shard_size="10GB",
        )
        self.accelerator.wait_for_everyone()
        rng_after = self._rng_state_sha256()
        if rng_before != rng_after:
            raise RuntimeError(
                "Model-only snapshot advanced a process-local RNG stream: "
                f"rank={self.accelerator.process_index} step={global_step}"
            )

        rng_record = {
            "schema_version": 1,
            "protocol": "model-only-atomic-rng-neutral-v1",
            "rank": self.accelerator.process_index,
            "world_size": self.accelerator.num_processes,
            "step": global_step,
            "rng_before_sha256": rng_before,
            "rng_after_sha256": rng_after,
            "rng_neutral": True,
        }
        rng_rank_path = (
            Path(stage_dir)
            / f"rng_audit_rank_{self.accelerator.process_index}.json"
        )
        rng_rank_tmp = rng_rank_path.with_suffix(".json.tmp")
        rng_rank_tmp.write_text(
            json.dumps(rng_record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(rng_rank_tmp, rng_rank_path)
        self.accelerator.wait_for_everyone()

        print(
            "MODEL_SNAPSHOT_RNG_JSON="
            + json.dumps(rng_record, sort_keys=True, separators=(",", ":")),
            flush=True,
        )

        if self.accelerator.is_main_process:
            rng_rank_states = []
            for expected_rank in range(self.accelerator.num_processes):
                path = Path(stage_dir) / f"rng_audit_rank_{expected_rank}.json"
                if not path.is_file() or path.is_symlink():
                    raise RuntimeError(
                        f"Missing model-only RNG evidence for rank {expected_rank}"
                    )
                document = json.loads(path.read_text(encoding="utf-8"))
                expected = {
                    "protocol": "model-only-atomic-rng-neutral-v1",
                    "rank": expected_rank,
                    "world_size": self.accelerator.num_processes,
                    "step": global_step,
                    "rng_neutral": True,
                }
                if any(document.get(key) != value for key, value in expected.items()):
                    raise RuntimeError(
                        f"Model-only RNG evidence mismatch for rank {expected_rank}"
                    )
                if document.get("rng_before_sha256") != document.get(
                    "rng_after_sha256"
                ):
                    raise RuntimeError(
                        f"Model-only RNG evidence is non-neutral for rank {expected_rank}"
                    )
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                rng_rank_states.append(
                    {
                        "rank": expected_rank,
                        "artifact": {
                            "path": path.name,
                            "bytes": path.stat().st_size,
                            "sha256": digest,
                        },
                    }
                )
            rng_manifest = {
                "schema_version": 1,
                "protocol": "model-only-atomic-rng-neutral-v1",
                "step": global_step,
                "world_size": self.accelerator.num_processes,
                "all_ranks_rng_neutral": True,
                "rank_states": rng_rank_states,
            }
            rng_manifest_path = Path(stage_dir) / "rng_audit_manifest.json"
            rng_manifest_tmp = rng_manifest_path.with_suffix(".json.tmp")
            rng_manifest_tmp.write_text(
                json.dumps(rng_manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(rng_manifest_tmp, rng_manifest_path)
            rng_manifest_identity = {
                "path": rng_manifest_path.name,
                "bytes": rng_manifest_path.stat().st_size,
                "sha256": hashlib.sha256(rng_manifest_path.read_bytes()).hexdigest(),
            }
            stage_path = Path(stage_dir)
            source_run_path = Path(self.project_dir).resolve()
            run_config_path = source_run_path / "run_config.json"
            live_metrics_path = source_run_path / "metrics.jsonl"
            statistics_path = source_run_path / "dataset_statistics.json"
            metrics_prefix_path = stage_path / "metrics_prefix.jsonl"
            if live_metrics_path.is_file() and not live_metrics_path.is_symlink():
                with live_metrics_path.open("rb") as source, metrics_prefix_path.open(
                    "xb"
                ) as destination:
                    while chunk := source.read(8 * 1024 * 1024):
                        destination.write(chunk)
                    destination.flush()
                    os.fsync(destination.fileno())

            require_bindings = os.environ.get("PSI_REQUIRE_CHECKPOINT_BINDINGS") == "1"
            if require_bindings:
                for label, path in (
                    ("run_config", run_config_path),
                    ("checkpoint-local metrics prefix", metrics_prefix_path),
                    ("dataset statistics", statistics_path),
                ):
                    if not path.is_file() or path.is_symlink():
                        raise RuntimeError(
                            f"Formal model-only snapshot lacks {label}: {path}"
                        )
            local_bindings = {}
            for label, path in (
                ("run_config", run_config_path),
                ("dataset_statistics", statistics_path),
            ):
                if path.is_file() and not path.is_symlink():
                    local_bindings[label] = _checkpoint_file_identity(path)
            if metrics_prefix_path.is_file() and not metrics_prefix_path.is_symlink():
                local_bindings["metrics_prefix"] = _checkpoint_file_identity(
                    metrics_prefix_path, relative_to=stage_path
                )

            try:
                external_binding_paths = json.loads(
                    os.environ.get("PSI_CHECKPOINT_BINDINGS_JSON", "{}")
                )
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "PSI_CHECKPOINT_BINDINGS_JSON must be JSON during snapshot"
                ) from exc
            if not isinstance(external_binding_paths, dict) or any(
                not isinstance(role, str)
                or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", role) is None
                or not isinstance(path, str)
                or not path
                for role, path in external_binding_paths.items()
            ):
                raise RuntimeError(
                    "PSI_CHECKPOINT_BINDINGS_JSON has an invalid role/path during snapshot"
                )
            required_binding_roles = {
                role.strip()
                for role in os.environ.get(
                    "PSI_REQUIRED_CHECKPOINT_BINDING_ROLES", ""
                ).split(",")
                if role.strip()
            }
            if require_bindings and (
                not required_binding_roles
                or not required_binding_roles.issubset(external_binding_paths)
            ):
                raise RuntimeError(
                    "Formal model-only snapshot external binding roster is incomplete"
                )
            external_bindings = {
                role: _checkpoint_file_identity(Path(path).expanduser().resolve())
                for role, path in sorted(external_binding_paths.items())
            }
            model_files = sorted(Path(stage_dir).glob("*.safetensors"))
            if len(model_files) != 1 or model_files[0].name != "model.safetensors":
                raise RuntimeError(
                    "Expected one unsharded model.safetensors in model-only "
                    f"snapshot, found {[path.name for path in model_files]}"
                )
            model_path = model_files[0]
            file_digest = hashlib.sha256()
            with model_path.open("rb") as stream:
                while chunk := stream.read(8 * 1024 * 1024):
                    file_digest.update(chunk)
            model_sha256 = file_digest.hexdigest()
            final_dir_abs = os.path.abspath(final_dir)
            source_run_abs = os.path.abspath(self.project_dir)
            complete_marker_abs = os.path.join(final_dir_abs, "COMPLETE")
            model_path_abs = os.path.join(final_dir_abs, "model.safetensors")
            manifest = {
                "schema_version": 1,
                "protocol": "model-only-atomic-rng-neutral-v1",
                # Canonical candidate-checkpoint identity fields.  Keep the
                # historical aliases below so existing consumers remain
                # readable while formal evaluators can validate one strict
                # schema across official and newly trained candidates.
                "checkpoint_id": (
                    f"{Path(self.project_dir).name}/ckpt_{global_step}"
                    f"@sha256:{model_sha256}"
                ),
                "step": global_step,
                "source_run": source_run_abs,
                "train_seed": int(self.cfg.seed),
                "formal_eligible": (
                    os.environ.get("PSI_FORMAL_ELIGIBLE") == "1"
                ),
                "COMPLETE": global_step,
                "complete_marker_path": complete_marker_abs,
                "model_path": model_path_abs,
                "checkpoint_step": global_step,
                "source_run_dir": source_run_abs,
                "model_file": "model.safetensors",
                "model_bytes": model_path.stat().st_size,
                "model_sha256": model_sha256,
                "rng_neutrality": "verified_on_every_rank_with_hash_bound_artifacts",
                "rng_audit_manifest": rng_manifest_identity,
                "world_size": self.accelerator.num_processes,
                "local_bindings": local_bindings,
                "external_bindings": external_bindings,
                "required_external_binding_roles": sorted(required_binding_roles),
                "created_at": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(timespec="seconds"),
            }
            manifest_tmp = Path(stage_dir) / "snapshot_manifest.json.tmp"
            manifest_path = Path(stage_dir) / "snapshot_manifest.json"
            manifest_tmp.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(manifest_tmp, manifest_path)
            (Path(stage_dir) / "COMPLETE").write_text(
                f"{global_step}\n", encoding="utf-8"
            )
            os.replace(stage_dir, final_dir)

        self.accelerator.wait_for_everyone()
        return final_dir

    @staticmethod
    def _validate_metrics_prefix_for_resume(metrics_path: Path, step: int) -> None:
        if metrics_path.is_symlink() or not metrics_path.is_file():
            raise RuntimeError(f"Resume metrics prefix is missing or unsafe: {metrics_path}")
        records = []
        for line_number, line in enumerate(
            metrics_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                raise RuntimeError(
                    f"Resume metrics prefix has a blank line: {metrics_path}:{line_number}"
                )
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Resume metrics prefix is invalid JSONL: {metrics_path}:{line_number}"
                ) from exc
            if not isinstance(record, dict):
                raise RuntimeError(
                    f"Resume metrics record is not an object: {metrics_path}:{line_number}"
                )
            if record.get("event_index") != len(records):
                raise RuntimeError(
                    f"Resume metrics event_index gap: {metrics_path}:{line_number}"
                )
            stream = record.get("stream")
            record_step = record.get("step")
            if stream not in {"train", "eval"}:
                raise RuntimeError(
                    f"Resume metrics stream is invalid: {metrics_path}:{line_number}"
                )
            if (
                not isinstance(record_step, int)
                or isinstance(record_step, bool)
                or record_step < 0
                or record.get("completed_updates") != record_step + 1
            ):
                raise RuntimeError(
                    f"Resume metrics step accounting is invalid: {metrics_path}:{line_number}"
                )
            records.append(record)
        train_steps = [record["step"] for record in records if record["stream"] == "train"]
        if train_steps != list(range(step)):
            raise RuntimeError(
                "Resume metrics train prefix must cover exactly one event for every "
                f"completed update 0..{step - 1}; observed first/last/count="
                f"{train_steps[:1]}/{train_steps[-1:]}/{len(train_steps)}"
            )
        if any(record["step"] >= step for record in records):
            raise RuntimeError(
                "Resume metrics contain events beyond the selected checkpoint boundary"
            )

    def _validate_resume_checkpoint(self, checkpoint_dir: Path, step: int) -> None:
        """Reopen a resume boundary and fail closed before Accelerate loads it."""
        if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
            raise RuntimeError(
                f"Resume checkpoint directory is missing or unsafe: {checkpoint_dir}"
            )
        complete_path = checkpoint_dir / "COMPLETE"
        if complete_path.is_symlink() or not complete_path.is_file():
            raise RuntimeError(f"Resume checkpoint lacks COMPLETE: {checkpoint_dir}")
        if complete_path.read_text(encoding="utf-8").strip() != str(step):
            raise RuntimeError(
                f"Resume checkpoint COMPLETE does not equal ckpt step {step}: {checkpoint_dir}"
            )

        formal = os.environ.get("PSI_FORMAL_ELIGIBLE") == "1"
        manifest_path = checkpoint_dir / "checkpoint_manifest.json"
        if not manifest_path.is_file() or manifest_path.is_symlink():
            if formal:
                raise RuntimeError(
                    f"Formal resume checkpoint lacks checkpoint_manifest.json: {checkpoint_dir}"
                )
            return
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Resume checkpoint manifest is invalid JSON: {manifest_path}"
            ) from exc
        if not isinstance(manifest, dict):
            raise RuntimeError(f"Resume checkpoint manifest is not an object: {manifest_path}")
        expected_values = {
            "protocol": "full-checkpoint-atomic-resume-v1",
            "step": step,
            "COMPLETE": step,
            "train_seed": int(self.cfg.seed),
            "formal_eligible": formal,
        }
        differing = {
            key: {"expected": expected, "actual": manifest.get(key)}
            for key, expected in expected_values.items()
            if manifest.get(key) != expected
        }
        if differing:
            raise RuntimeError(f"Resume checkpoint manifest identity mismatch: {differing}")
        expected_source_run = str(Path(self.project_dir).resolve())
        if manifest.get("source_run") != expected_source_run:
            raise RuntimeError(
                "Resume checkpoint belongs to a different source run: "
                f"{manifest.get('source_run')!r} != {expected_source_run!r}"
            )
        if manifest.get("complete_marker_path") != str(complete_path.resolve()):
            raise RuntimeError("Resume checkpoint manifest COMPLETE path mismatch")
        model_path = checkpoint_dir / "model.safetensors"
        if manifest.get("model_path") != str(model_path.resolve()):
            raise RuntimeError("Resume checkpoint manifest model path mismatch")
        model_identity = _checkpoint_file_identity(model_path)
        if (
            manifest.get("model_sha256") != model_identity["sha256"]
            or manifest.get("model_bytes") != model_identity["bytes"]
        ):
            raise RuntimeError("Resume checkpoint model identity mismatch")

        payload_inventory = manifest.get("payload_inventory")
        if formal and not isinstance(payload_inventory, dict):
            raise RuntimeError("Formal resume checkpoint lacks exact payload inventory")
        if isinstance(payload_inventory, dict):
            actual_names = {
                path.name
                for path in checkpoint_dir.iterdir()
                if path.is_file() and path.name not in {"COMPLETE", "checkpoint_manifest.json"}
            }
            if set(payload_inventory) != actual_names:
                raise RuntimeError(
                    "Resume checkpoint payload inventory mismatch: "
                    f"manifest={sorted(payload_inventory)}, actual={sorted(actual_names)}"
                )
            for name in sorted(actual_names):
                claimed = payload_inventory.get(name)
                actual = _checkpoint_file_identity(
                    checkpoint_dir / name, relative_to=checkpoint_dir
                )
                if not isinstance(claimed, dict) or any(
                    claimed.get(key) != actual[key]
                    for key in ("path", "bytes", "sha256")
                ):
                    raise RuntimeError(
                        f"Resume checkpoint payload identity mismatch: {name}"
                    )

        expected_world_size = self.accelerator.num_processes
        if formal and manifest.get("world_size") != expected_world_size:
            raise RuntimeError(
                "Formal resume checkpoint world size mismatch: "
                f"{manifest.get('world_size')} != {expected_world_size}"
            )
        if formal:
            required_names = {
                "model.safetensors",
                "optimizer.bin",
                "scheduler.bin",
                "sampler_resume_manifest.json",
                *(f"random_states_{rank}.pkl" for rank in range(expected_world_size)),
                *(f"sampler_resume_rank_{rank}.json" for rank in range(expected_world_size)),
            }
            missing = required_names - actual_names
            if missing:
                raise RuntimeError(
                    f"Formal resume checkpoint is missing payloads: {sorted(missing)}"
                )

        local_bindings = manifest.get("local_bindings")
        if formal and not isinstance(local_bindings, dict):
            raise RuntimeError("Formal resume checkpoint lacks local bindings")
        if isinstance(local_bindings, dict):
            current_local_paths = {
                "run_config": Path(self.project_dir) / "run_config.json",
                "metrics_prefix": checkpoint_dir / "metrics_prefix.jsonl",
                "dataset_statistics": Path(self.project_dir) / "dataset_statistics.json",
            }
            if formal and set(local_bindings) != set(current_local_paths):
                raise RuntimeError("Formal resume checkpoint local binding roster mismatch")
            for role, claimed in local_bindings.items():
                path = current_local_paths.get(role)
                if path is None:
                    raise RuntimeError(f"Unknown resume local binding role: {role}")
                actual = _checkpoint_file_identity(
                    path,
                    relative_to=checkpoint_dir if role == "metrics_prefix" else None,
                )
                if not isinstance(claimed, dict) or any(
                    claimed.get(key) != actual[key]
                    for key in ("path", "bytes", "sha256")
                ):
                    raise RuntimeError(f"Resume local binding changed: {role}")
            if "metrics_prefix" in local_bindings:
                live_metrics_path = Path(self.project_dir) / "metrics.jsonl"
                if (
                    not live_metrics_path.is_file()
                    or live_metrics_path.is_symlink()
                    or live_metrics_path.read_bytes()
                    != current_local_paths["metrics_prefix"].read_bytes()
                ):
                    raise RuntimeError(
                        "Resume metrics contain data beyond or differing from the "
                        "immutable checkpoint prefix"
                    )
                self._validate_metrics_prefix_for_resume(
                    live_metrics_path, step
                )

        external_bindings = manifest.get("external_bindings")
        required_roles = manifest.get("required_external_binding_roles")
        if formal:
            if not isinstance(external_bindings, dict) or not isinstance(
                required_roles, list
            ):
                raise RuntimeError("Formal resume checkpoint lacks external bindings")
            try:
                current_paths = json.loads(
                    os.environ.get("PSI_CHECKPOINT_BINDINGS_JSON", "{}")
                )
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "PSI_CHECKPOINT_BINDINGS_JSON is invalid during resume"
                ) from exc
            if (
                not isinstance(current_paths, dict)
                or set(external_bindings) != set(current_paths)
                or set(required_roles) != set(external_bindings)
            ):
                raise RuntimeError("Formal resume external binding roster changed")
            for role, path_value in sorted(current_paths.items()):
                actual = _checkpoint_file_identity(Path(path_value).expanduser().resolve())
                claimed = external_bindings[role]
                if not isinstance(claimed, dict) or any(
                    claimed.get(key) != actual[key]
                    for key in ("path", "bytes", "sha256")
                ):
                    raise RuntimeError(f"Formal resume external binding changed: {role}")

    def resume_from_checkpoint(self) -> tuple[int, Optional[str]]:
        """ resume from a checkpoint if specified in the config. 
            the checkpoint path can be either:
            1) full path to a checkpoint folder, e.g.
               .runs/trainer_name/run_name/checkpoints/ckpt_xxxxxx
            3) or .runs/trainer_name/run_name (latest)
        """
        if self.cfg.train.resume_from_checkpoint is None:
            return 0, None

        requested = Path(str(self.cfg.train.resume_from_checkpoint)).expanduser()
        direct_match = re.fullmatch(r"ckpt_([0-9]+)", requested.name)
        if direct_match is not None:
            checkpoint_dir = requested.absolute()
            step = int(direct_match.group(1))
        else:
            run_root = requested.absolute()
            candidates_root = run_root / "checkpoints"
            if candidates_root.is_symlink() or not candidates_root.is_dir():
                raise RuntimeError(
                    f"Requested resume run has no safe checkpoints directory: {run_root}"
                )
            candidates: list[tuple[int, Path]] = []
            for candidate in candidates_root.iterdir():
                match = re.fullmatch(r"ckpt_([0-9]+)", candidate.name)
                if (
                    match is not None
                    and candidate.is_dir()
                    and not candidate.is_symlink()
                    and (candidate / "COMPLETE").is_file()
                ):
                    candidates.append((int(match.group(1)), candidate))
            if not candidates:
                raise RuntimeError(
                    f"Requested resume run has no COMPLETE checkpoint: {run_root}"
                )
            step, checkpoint_dir = max(candidates, key=lambda item: item[0])

        if os.environ.get("PSI_FORMAL_ELIGIBLE") == "1":
            expected_dir = (
                Path(self.project_dir).resolve() / "checkpoints" / f"ckpt_{step}"
            )
            if checkpoint_dir.resolve() != expected_dir:
                raise RuntimeError(
                    "Formal resume must use this run's canonical checkpoint boundary: "
                    f"{checkpoint_dir} != {expected_dir}"
                )

        validation_error: list[str | None] = [None]
        if self.accelerator.is_main_process:
            try:
                self._validate_resume_checkpoint(checkpoint_dir, step)
            except Exception as exc:
                validation_error[0] = f"{type(exc).__name__}: {exc}"
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast_object_list(validation_error, src=0)
        if validation_error[0] is not None:
            raise RuntimeError(
                f"Resume checkpoint authorization failed: {validation_error[0]}"
            )

        load_path = str(checkpoint_dir)
        overwatch.info(f"Resuming from checkpoint {load_path}")
        self.accelerator.load_state(load_path)
        # ckpt_N is saved after N completed optimization steps, so the next
        # global step is N (zero-based), not N + 1.
        return step, load_path

    @property
    def lr(self):
        return self.get_lr()
    
    def get_lr(self):
        return self.lr_scheduler.get_last_lr()[0]
    
    def get_total_grad_norm(self, params=None, norm_type=2):
        if "DeepSpeedEngine" in self.model.__class__.__name__:
            grad_norm = self.model.get_global_grad_norm()
            return grad_norm
        else:
            total_norm = 0.0
            if params is None:
                params = self.model.parameters()
            for p in params:
                if p.grad is not None:
                    param_norm = p.grad.data.norm(norm_type)
                    total_norm += param_norm.item() ** norm_type
                
            total_norm = total_norm ** (1. / norm_type)
            return total_norm

    def unwrap_model(self):
        # Function for unwrapping if model was compiled with `torch.compile`.
        model = self.accelerator.unwrap_model(self.model)
        from diffusers.utils.torch_utils import is_compiled_module

        model = model._orig_mod if is_compiled_module(model) else model
        # songlin: revert original forward which is changed by accelerate.prepare to always return fp32
        # if hasattr(model, "_original_forward"):
        #     model.forward = model._original_forward
        return model
    
    # def log(
    #     self, ):
    #     ...

    # def evaluate(self, global_step: int, accelerator: Accelerator):
    #     ...

    def finalize(self):
        """Release persistent DataLoader workers before interpreter teardown.

        ``accelerate.prepare`` wraps each PyTorch DataLoader, while
        ``DataLoaderStateMixin.end`` only updates Accelerate's gradient state.
        It does not stop the underlying persistent worker processes.  Leaving
        PyAV workers alive until interpreter teardown can make the workers
        abort after an otherwise successful training run, and torchrun may
        still report exit code zero.  Shut the workers down while the runtime
        and distributed process group are still intact so cleanup failures are
        surfaced as training failures.
        """
        for name in ("train_dataloader", "val_dataloader"):
            dataloader = getattr(self, name, None)
            if dataloader is None:
                continue

            # Accelerate's DataLoaderAdapter owns the actual PyTorch
            # DataLoader as ``base_dataloader``.  Keep the fallback for
            # unprepared/native loaders and for other trainer backends.
            base_dataloader = getattr(dataloader, "base_dataloader", dataloader)
            iterator = getattr(base_dataloader, "_iterator", None)
            if iterator is None:
                continue

            shutdown_workers = getattr(iterator, "_shutdown_workers", None)
            if shutdown_workers is not None:
                shutdown_workers()
            base_dataloader._iterator = None

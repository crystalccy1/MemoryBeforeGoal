# Derived from Psi0 (https://github.com/physical-superintelligence-lab/Psi0, Apache License 2.0) and modified; see NOTICE.
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()  # optional .env, e.g. PSI_HOME, WANDB_API_KEY
import os
import re
import torch
import gc
from tqdm import tqdm
import shutil
import json
import hashlib
import datetime
import secrets
import sys
import shlex
import tyro
import importlib
from pathlib import Path
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from psi.config.config import LaunchConfig
from psi.utils import batch_str_to_tensor, seed_everything, initialize_overwatch, nice, flatten
from psi.trainers import Trainer

overwatch = initialize_overwatch(__name__)

from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.data_loader import (
    DataLoaderStateMixin as AcceleratorDataLoaderStateMixin,
)
from accelerate.utils import ProjectConfiguration
from torch.distributed.elastic.multiprocessing.errors import record
import random, numpy as np

MAX_TRAINING_EPOCHS = 1_000_000


def _resume_invariant_config(config: dict) -> dict:
    """Return the fields that must remain stable across run attempts."""
    normalized = json.loads(json.dumps(config))
    normalized.get("train", {})["resume_from_checkpoint"] = None
    normalized.get("wandb", {})["resume"] = "never"
    return normalized


def _format_argv_lines(argv: list[str]) -> str:
    lines: list[str] = []
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg.startswith("--"):
            values = [arg]
            cursor = index + 1
            while cursor < len(argv) and not argv[cursor].startswith("--"):
                values.append(argv[cursor])
                cursor += 1
            lines.append(" ".join(values))
            index = cursor
        else:
            lines.append(arg)
            index += 1
    return "\n".join(lines) + "\n"


def _write_text_exclusive(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _resolve_dataset_statistics_source(data_config: object) -> Path | None:
    """Resolve the exact statistics file consumed by a data configuration."""

    field = getattr(getattr(data_config, "transform", None), "field", None)
    raw_stat_path = getattr(field, "stat_path", None)
    if not isinstance(raw_stat_path, str) or not raw_stat_path:
        return None

    from psi.utils import resolve_data_path, resolve_path

    stat_path = Path(raw_stat_path).expanduser()
    resolved = Path(resolve_path(stat_path))
    if resolved.exists():
        return resolved

    repo_ids = getattr(data_config, "train_repo_ids", None)
    root_dir = getattr(data_config, "root_dir", None)
    if (
        not stat_path.is_absolute()
        and isinstance(root_dir, str)
        and root_dir
        and isinstance(repo_ids, (list, tuple))
        and repo_ids
        and isinstance(repo_ids[0], str)
        and repo_ids[0]
    ):
        repo_relative = Path(root_dir).expanduser() / repo_ids[0] / stat_path
        resolved = Path(resolve_data_path(repo_relative))
        if resolved.exists():
            return resolved
    return None


def _record_attempt_provenance(trainer: Trainer) -> Path:
    """Preserve immutable initial and per-attempt configuration evidence."""
    project_dir = Path(trainer.project_dir)
    attempt_id = os.environ.get("PSI_ATTEMPT_ID") or (
        datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + f"-pid{os.getpid()}"
    )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", attempt_id):
        raise ValueError(f"Invalid PSI_ATTEMPT_ID: {attempt_id!r}")
    # JSON mode canonicalizes Path and enum values before we create any
    # attempt directory, so serialization failures cannot leave a misleading
    # empty attempt artifact behind.
    config = json.loads(trainer.cfg.model_dump_json())
    config_text = json.dumps(config, indent=2, sort_keys=True) + "\n"
    argv_text = _format_argv_lines(list(sys.argv))
    argv_json = json.dumps(list(sys.argv), indent=2) + "\n"
    env_info = (
        json.loads(os.environ["_ENV_INFO_JSON"])
        if "_ENV_INFO_JSON" in os.environ
        else {}
    )
    env_text = "".join(f"{key}={value}\n" for key, value in env_info.items())

    canonical_config = project_dir / "run_config.json"
    canonical_argv = project_dir / "argv.txt"
    canonical_env = project_dir / "envs.txt"
    if canonical_config.exists():
        if (
            canonical_config.is_symlink()
            or canonical_argv.is_symlink()
            or canonical_env.is_symlink()
        ):
            raise RuntimeError(
                f"Existing run uses unsafe provenance symlinks: {project_dir}"
            )
        initial = json.loads(canonical_config.read_text(encoding="utf-8"))
        if _resume_invariant_config(initial) != _resume_invariant_config(config):
            raise RuntimeError(
                "Resume configuration drift detected; refusing to overwrite "
                f"the immutable initial config at {canonical_config}"
            )
        if not canonical_argv.is_file() or not canonical_env.is_file():
            raise RuntimeError(
                "Existing run is missing immutable argv/env provenance: "
                f"{project_dir}"
            )

    statistics_source: Path | None = None
    statistics_destination: Path | None = None
    if hasattr(trainer.cfg.data.transform.field, "stat_path"):
        resolved = _resolve_dataset_statistics_source(trainer.cfg.data)
        if resolved is not None:
            if resolved.is_symlink() or not resolved.is_file():
                raise RuntimeError(f"Dataset statistics source is unsafe: {resolved}")
            statistics_source = resolved
            statistics_destination = project_dir / "dataset_statistics.json"
            if statistics_destination.exists():
                if (
                    statistics_destination.is_symlink()
                    or not statistics_destination.is_file()
                    or statistics_source.read_bytes()
                    != statistics_destination.read_bytes()
                ):
                    raise RuntimeError(
                        "Dataset statistics drift detected on resume: "
                        f"{statistics_source} != {statistics_destination}"
                    )
        elif os.environ.get("PSI_REQUIRE_CHECKPOINT_BINDINGS") == "1":
            raise RuntimeError(
                "Required dataset statistics source could not be resolved from "
                f"{trainer.cfg.data.transform.field.stat_path!r}"
            )

    attempt_dir = project_dir / "attempts" / attempt_id
    if os.path.lexists(attempt_dir):
        raise FileExistsError(f"Attempt provenance already exists: {attempt_dir}")

    # All resume/config/statistics validation is complete before any attempt
    # directory is created and, critically, before remote trackers are opened.
    if not canonical_config.exists():
        _write_text_exclusive(canonical_config, config_text)
        _write_text_exclusive(canonical_argv, argv_text)
        _write_text_exclusive(canonical_env, env_text)
    if statistics_source is not None and statistics_destination is not None:
        if not statistics_destination.exists():
            shutil.copy2(statistics_source, statistics_destination)

    attempt_dir.mkdir(parents=True, exist_ok=False)
    try:
        _write_text_exclusive(attempt_dir / "run_config.json", config_text)
        _write_text_exclusive(attempt_dir / "argv.txt", argv_text)
        _write_text_exclusive(attempt_dir / "argv.json", argv_json)
        _write_text_exclusive(attempt_dir / "envs.txt", env_text)
        attempt = {
            "schema_version": 1,
            "attempt_id": attempt_id,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(
                timespec="seconds"
            ),
            "run_dir": str(project_dir.resolve()),
            "resume_from_checkpoint": (
                str(trainer.cfg.train.resume_from_checkpoint)
                if trainer.cfg.train.resume_from_checkpoint is not None
                else None
            ),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "hostname": os.uname().nodename,
            "world_size": trainer.world_size,
            "formal_eligible": os.environ.get("PSI_FORMAL_ELIGIBLE") == "1",
            "command_shell": shlex.join(sys.argv),
        }
        _write_text_exclusive(
            attempt_dir / "attempt.json",
            json.dumps(attempt, indent=2, sort_keys=True) + "\n",
        )
    except Exception:
        shutil.rmtree(attempt_dir)
        raise
    return attempt_dir


def _bind_wandb_identity_before_tracking(trainer: Trainer) -> None:
    """Resolve the W&B identity locally before immutable resume validation."""

    if trainer.cfg.log.report_to != "wandb":
        return
    run_id = getattr(trainer.cfg.wandb, "id", None)
    if (
        not run_id
        and trainer.cfg.train.resume_from_checkpoint is not None
        and trainer.cfg.wandb.resume != "never"
    ):
        resume_path = Path(
            str(trainer.cfg.train.resume_from_checkpoint)
        ).expanduser()
        run_dir = (
            resume_path.parent.parent
            if resume_path.name.startswith("ckpt_")
            else resume_path
        )
        run_config_path = run_dir / "run_config.json"
        if run_config_path.is_symlink() or not run_config_path.is_file():
            raise RuntimeError(
                f"Resume run lacks safe immutable configuration: {run_config_path}"
            )
        run_id = (
            json.loads(run_config_path.read_text(encoding="utf-8")).get("wandb")
            or {}
        ).get("id")
        if not run_id:
            raise RuntimeError(
                f"Resume run has no immutable W&B id: {run_config_path}"
            )
        overwatch.info(f"resume wandb with run id: {run_id}")
    if not run_id:
        # W&B accepts user-supplied lowercase hexadecimal IDs. Generating it
        # locally avoids opening a remote run merely to learn its identity.
        run_id = secrets.token_hex(4)
    trainer.cfg.wandb.id = run_id
    trainer.cfg.wandb.name = trainer.run_name
    if not trainer.cfg.wandb.group:
        trainer.cfg.wandb.group = trainer.cfg.train.name


def _verify_expected_train_argv() -> None:
    """Bind each formal rank's real ``sys.argv`` to the audited launch vector."""

    path_raw = os.environ.get("PSI_EXPECTED_TRAIN_ARGV_JSON")
    digest_raw = os.environ.get("PSI_EXPECTED_TRAIN_ARGV_SHA256")
    formal = os.environ.get("PSI_FORMAL_ELIGIBLE") == "1"
    if not path_raw and not digest_raw:
        if formal:
            raise RuntimeError(
                "Formal training requires an audited effective argv artifact"
            )
        return
    if not path_raw or not digest_raw or re.fullmatch(r"[0-9a-f]{64}", digest_raw) is None:
        raise RuntimeError("Expected training argv path/hash binding is incomplete")
    path = Path(path_raw).expanduser()
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Expected training argv artifact is unsafe: {path}")
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable):
        raise RuntimeError("Expected training argv artifact changed while reading")
    if hashlib.sha256(payload).hexdigest() != digest_raw:
        raise RuntimeError("Expected training argv artifact hash mismatch")
    try:
        vector = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Expected training argv artifact is invalid JSON") from exc
    if (
        not isinstance(vector, list)
        or not vector
        or any(not isinstance(item, str) or not item for item in vector)
    ):
        raise RuntimeError("Expected training argv must be a non-empty string list")
    entrypoint = str(Path(__file__).resolve())
    if vector.count(entrypoint) != 1:
        raise RuntimeError("Expected training argv has an invalid train entrypoint")
    expected_python_argv = vector[vector.index(entrypoint) :]
    if list(sys.argv) != expected_python_argv:
        raise RuntimeError(
            "Actual scripts/train.py argv differs from the audited effective vector: "
            f"expected={expected_python_argv!r}, actual={list(sys.argv)!r}"
        )


def _model_only_snapshot_steps(max_training_steps: int) -> frozenset[int]:
    raw = os.environ.get("PSI_MODEL_ONLY_SNAPSHOT_STEPS", "").strip()
    if not raw:
        return frozenset()
    try:
        values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    except ValueError as exc:
        raise ValueError(
            "PSI_MODEL_ONLY_SNAPSHOT_STEPS must be a comma-separated list of integers"
        ) from exc
    if not values or len(values) != len(set(values)):
        raise ValueError(
            "PSI_MODEL_ONLY_SNAPSHOT_STEPS must contain unique non-empty steps"
        )
    invalid = [step for step in values if not 1 <= step <= max_training_steps]
    if invalid:
        raise ValueError(
            "PSI_MODEL_ONLY_SNAPSHOT_STEPS contains out-of-range steps: "
            f"{invalid}; max={max_training_steps}"
        )
    return frozenset(values)

def _auto_tag_run(run_name: str):
    """commit the code and tag the run with the run name"""
    import subprocess

    subprocess.run(["git", "add", "."], check=False)
    subprocess.run(["git", "commit", "-m", f"Auto-commit for {run_name}"], check=False)
    subprocess.run(["git", "tag", run_name], check=False)
    # subprocess.run(["git", "push", "--tags"], check=True)

def _initialize_accelerator(trainer: Trainer) -> Accelerator:
    os.makedirs(trainer.project_dir, exist_ok=True)
    logging_dir = os.path.join(trainer.project_dir, trainer.cfg.log.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=trainer.project_dir, logging_dir=logging_dir
    )

    # setup FSDP training strategy if needed
    fsdp_plugin = None
    if trainer.cfg.train.data_parallel == "fsdp":
        fsdp_plugin = trainer.get_fsdp_plugin()
    
    deepspeed_plugin = None
    if trainer.cfg.train.data_parallel == "deepspeed":
        # SONGLIN: use accelerate launch config instead
        ds_config_path = trainer.cfg.train.deepspeed_config 
        deepspeed_plugin = DeepSpeedPlugin(zero_stage=3, hf_ds_config=ds_config_path)
    

    accelerator = Accelerator(
        gradient_accumulation_steps=trainer.cfg.train.gradient_accumulation_steps,
        mixed_precision=trainer.cfg.train.mixed_precision,
        # downcast_bf16=True,
        log_with=trainer.cfg.log.report_to,
        project_config=accelerator_project_config,
        fsdp_plugin=fsdp_plugin,
        deepspeed_plugin=deepspeed_plugin
    )

    if trainer.cfg.train.data_parallel == "deepspeed":
        # When using DeepSpeed, accelerate.prepare() requires you 
        # to pass at least one of training or evaluation dataloaders 
        # with batch_size attribute returning an integer value or alternatively 
        # set an integer value in train_micro_batch_size_per_gpu 
        # in the deepspeed config file or assign integer value to
        accelerator.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = trainer.device_train_batch_size # type: ignore

    trainer.accelerator = accelerator
    return accelerator


def _initialize_trackers_and_attempt_provenance(
    trainer: Trainer, accelerator: Accelerator
) -> None:
    """Start remote tracking only after any requested resume passed validation."""

    if not overwatch.is_rank_zero() or trainer.cfg.eval:
        return
    _bind_wandb_identity_before_tracking(trainer)

    # This performs every immutable-config, attempt-ID, and dataset-statistics
    # check before any W&B API can create or resume a remote run.
    os.makedirs(trainer.project_dir, exist_ok=True)
    attempt_dir = _record_attempt_provenance(trainer)
    overwatch.info(f"Saved immutable run/attempt configuration to {attempt_dir}")

    if trainer.cfg.auto_tag_run:
        _auto_tag_run(trainer.run_name)

    if trainer.cfg.log.report_to == "wandb":
        tracker_config = trainer.cfg.model_dump()

        if "_ENV_INFO_JSON" in os.environ:
            env_info = json.loads(os.environ["_ENV_INFO_JSON"])
            tracker_config["environment_variables"] = env_info

        wandb_config = dict(trainer.cfg.wandb)
        wandb_config["name"] = trainer.run_name
        wandb_config["dir"] = os.path.abspath(
            os.path.expanduser(trainer.project_dir)
        )
        if not wandb_config.get("group"):
            wandb_config["group"] = trainer.cfg.train.name
        project = wandb_config.pop("project", "default")

        accelerator.init_trackers(project, tracker_config, {"wandb": wandb_config})
        for tracker in accelerator.trackers:
            if tracker.name == "wandb":
                trainer.cfg.wandb.id = tracker.run.id
                trainer.cfg.wandb.name = tracker.run.name
                trainer.cfg.wandb.entity = tracker.run.entity
                trainer.cfg.wandb.group = tracker.run.group

@record
def train(config: LaunchConfig):
    torch.cuda.set_device(device_id := overwatch.local_rank())
    torch.cuda.empty_cache()

    if config.seed:
        overwatch.info(f"Seed everything with {config.seed}")
        seed_everything(config.seed)
        
    trainer = Trainer.instantiate(config, device_id)
    overwatch.info("Initialize models ... ")
    trainer.init_models()
    accelerator = _initialize_accelerator(trainer)
    if os.environ.get("PSI_WRITE_PARAMETER_AUDIT", "0") == "1":
        parameter_audit_path = trainer.write_parameter_audit()
        if overwatch.is_rank_zero():
            overwatch.info(
                f"Saved immutable pre-optimizer parameter audit to {parameter_audit_path}"
            )
    overwatch.info(f"Training configurations:")
    overwatch.info(f"training task: '{config.train.name}'", ctx_level=1)
    overwatch.info(f"run name: {trainer.run_name}", ctx_level=1)
    overwatch.info(f"seed: {config.seed}", ctx_level=1)
    overwatch.info(f"mixed precision: {trainer.dtype}", ctx_level=1)
    overwatch.info(f"warmup steps: {trainer.num_warmup_steps}", ctx_level=1)
    overwatch.info(f"validation steps: {config.train.validation_steps}", ctx_level=1)
    overwatch.info(f"checkpoint steps: {config.train.checkpointing_steps}", ctx_level=1)
    overwatch.info(f"max gradient norm: {config.train.max_grad_norm}", ctx_level=1)

    train_dataset, val_dataset = trainer.create_datasets()
    overwatch.info(f"Num training samples:", ctx_level=1)
    overwatch.info(f"Training dataset size: {trainer.len_train_dataset:,}", ctx_level=2)  # type: ignore
    if val_dataset:
        overwatch.info(f"Val dataset size: {trainer.len_val_dataset:,}", ctx_level=2)  # type: ignore
    trainer.create_dataloaders(train_dataset, val_dataset)
    
    overwatch.info("Initialize optimizers and schedulers...")
    trainer.create_optimizer_and_scheduler(trainer.max_training_steps)

    # fmt: off
    overwatch.info("***** Running training *****")
    overwatch.info(f"Num training examples = {trainer.len_train_dataset}", ctx_level=1)
    overwatch.info(f"Max training Epochs = {trainer.max_training_epochs}", ctx_level=1)
    overwatch.info(f"Total optimization steps = {trainer.max_training_steps}", ctx_level=2)
    overwatch.info(f"Num steps Per Epoch = {trainer.num_steps_per_epoch}", ctx_level=2)
    overwatch.info(f"Effective training epochs = {trainer.max_training_steps/trainer.num_steps_per_epoch*trainer.world_size:.2f}", ctx_level=2)
    overwatch.info(f"Global train batch size (w. parallel, distributed & accumulation) = {trainer.global_train_batch_size}", ctx_level=1)
    overwatch.info(f"Device train batch size = {trainer.device_train_batch_size}", ctx_level=2)
    overwatch.info(f"Gradient Accumulation steps = {trainer.gradient_accumulation_steps}", ctx_level=2)
    overwatch.info(f"Num processes (GPUs) = {trainer.world_size}", ctx_level=2)
    # fmt: on

    trainer.prepare(accelerator)
    global_step = initial_global_step = trainer.resume_from_checkpoint()[0]
    model_only_snapshot_steps = _model_only_snapshot_steps(
        trainer.max_training_steps
    )
    if overwatch.is_rank_zero() and model_only_snapshot_steps:
        overwatch.info(
            "Model-only evaluation snapshots: "
            + ",".join(str(step) for step in sorted(model_only_snapshot_steps))
        )

    stop_after_steps = None
    stop_after_steps_raw = os.environ.get("PSI_STOP_AFTER_STEPS")
    if stop_after_steps_raw:
        try:
            stop_after_steps = int(stop_after_steps_raw)
        except ValueError as exc:
            raise ValueError(
                "PSI_STOP_AFTER_STEPS must be an integer number of completed updates"
            ) from exc
        if not (
            initial_global_step < stop_after_steps <= trainer.max_training_steps
        ):
            raise ValueError(
                "PSI_STOP_AFTER_STEPS must be greater than the resumed step and "
                f"at most {trainer.max_training_steps}; got "
                f"initial={initial_global_step}, stop={stop_after_steps}"
            )
        if (
            stop_after_steps != trainer.max_training_steps
            and stop_after_steps % config.train.checkpointing_steps != 0
        ):
            raise ValueError(
                "PSI_STOP_AFTER_STEPS must align with checkpointing_steps; got "
                f"stop={stop_after_steps}, interval={config.train.checkpointing_steps}"
            )

    if initial_global_step >= trainer.max_training_steps:
        raise RuntimeError(
            "The selected checkpoint already reached max_training_steps; refusing "
            "to execute an extra optimizer update."
        )

    # FinetuneTrainer reconstructs its RandomSampler from the original seed,
    # advances every fully completed *prepared-loader* epoch, and returns the
    # remaining device batches to skip.  This is automatic for segmented
    # finetuning: the legacy skip_resumed_steps flag did not restore sampler
    # state and its configured num_steps_per_epoch can differ from Accelerate's
    # actual drop-last shard length (task1 reports 666 but yields 665).
    configure_sampler_resume = getattr(
        trainer, "configure_train_sampler_resume", None
    )
    if configure_sampler_resume is not None:
        sampler_resume_plan = configure_sampler_resume(initial_global_step)
        epoch_start = sampler_resume_plan.completed_epochs
        resume_batches_to_skip = sampler_resume_plan.skip_batches
        # Machine-readable, unstyled evidence for per-attempt segment audits.
        # Plain print avoids Rich terminal-width wrapping.  Every rank emits a
        # record, so an audit can require exact cross-rank agreement.
        sampler_resume_evidence = {
            "protocol": "legacy-shared-generator-split-v1",
            "rank": accelerator.process_index,
            "world_size": accelerator.num_processes,
            "initial_global_step": sampler_resume_plan.initial_global_step,
            "configured_steps_per_epoch": trainer.num_steps_per_epoch,
            "prepared_steps_per_epoch": sampler_resume_plan.steps_per_epoch,
            "prepared_batches_per_epoch": (
                sampler_resume_plan.prepared_batches_per_epoch
            ),
            "resume_completed_epochs": sampler_resume_plan.completed_epochs,
            "resume_batch_offset": sampler_resume_plan.skip_batches,
            "sampler_epochs_fast_forwarded": (
                sampler_resume_plan.completed_epochs
            ),
            "gradient_accumulation_steps": (
                sampler_resume_plan.gradient_accumulation_steps
            ),
            "sampler_seed": getattr(trainer, "_train_sampler_seed", None),
            "legacy_initial_worker_base_seed": getattr(
                trainer, "_train_legacy_worker_base_seed", None
            ),
            "sample_index_continuity": "exact",
            "persistent_worker_augmentation_rng_continuity": "not_restored",
        }
        print(
            "TRAIN_SAMPLER_RESUME_JSON="
            + json.dumps(
                sampler_resume_evidence, sort_keys=True, separators=(",", ":")
            ),
            flush=True,
        )
        if sampler_resume_plan.steps_per_epoch != trainer.num_steps_per_epoch:
            overwatch.warning(
                "Configured num_steps_per_epoch differs from the prepared "
                "DataLoader cadence; exact resume uses the prepared cadence "
                f"({trainer.num_steps_per_epoch} configured vs "
                f"{sampler_resume_plan.steps_per_epoch} prepared)."
            )
    else:
        epoch_start = global_step // trainer.num_steps_per_epoch
        resume_batches_to_skip = (
            initial_global_step % trainer.num_steps_per_epoch
            if config.train.skip_resumed_steps
            else 0
        )

    # Every local resume, segment-control, sampler, immutable-config, and
    # dataset-statistics check has now passed. Only now may rank zero create or
    # resume a remote tracker. The provenance helper itself creates the attempt
    # directory transactionally before calling ``init_trackers``.
    _initialize_trackers_and_attempt_provenance(trainer, accelerator)

    overwatch.info(f"Accelerator runs in: {trainer.project_dir}")
    trainer.set_train()
    
    progress_bar = tqdm(
        range(0, trainer.max_training_steps),
        initial=initial_global_step,
        desc="Training steps",
        disable=not overwatch.is_rank_zero(),
        position=0,
    )

    is_max_train_steps_reached = False
    segment_paused = False

    for epoch in range(epoch_start, MAX_TRAINING_EPOCHS):
        trainer.next_epoch(epoch)
        accelerator.wait_for_everyone()
        for local_step, batch in enumerate(trainer.train_dataloader):
            if epoch == epoch_start and local_step < resume_batches_to_skip:
                # Reconstruct the exact per-rank sample-index position inside
                # the first resumed epoch.  Worker augmentation/prefetch RNG
                # remains segment-local and is not claimed bitwise-restored.
                continue
            sync_gradients, losses = trainer.step(batch_str_to_tensor(batch), global_step, local_step)
            
            if sync_gradients:
                # log metrics
                trainer.log(flatten({**losses, "epoch": epoch}, parent_key="train")) 

                # validation
                if config.train.validation_steps > 0 and (
                        global_step == 0 or
                        (global_step + 1) % config.train.validation_steps == 0 or
                        global_step == trainer.max_training_steps - 1
                    ) :
                    gc.collect()
                    torch.cuda.empty_cache()
                    trainer.set_eval()
                    with torch.no_grad(), torch.autocast(
                        device_type="cuda", dtype=trainer.dtype
                    ):
                        eval_losses = trainer.evaluate()
                        if eval_losses is not None: # FIXME
                            trainer.log(flatten(eval_losses, parent_key="eval")) 
                    trainer.set_train()
                    """ NOTE:
                        This is a workaround for iterating over val_dataloader in the middle of training_dataloader.
                        When val dataloader dost not end explicitly, eg., the val dataloader is wrapped in a tqdm progress bar,
                        and the bar is closed before the end of the dataloader, it will cause an inconsistent accelerate gradient state
                    """
                    if hasattr(trainer, "val_dataloader") and isinstance(
                        trainer.val_dataloader, AcceleratorDataLoaderStateMixin
                    ):
                        if (
                            accelerator.gradient_state.active_dataloader
                            != trainer.train_dataloader
                        ):
                            try:
                                trainer.val_dataloader.end()
                                overwatch.warning_once(
                                    "Warning only Once: "
                                    "Val dataloader does not end after evaluation. I killed it for you."
                                )
                            except ValueError:
                                overwatch.warning_once("Even I can not save you.")

                # Save after validation so a resumed segment restores the main
                # process RNG state after validation. Model/optimizer/scheduler
                # values are unchanged by validation. The finetune sample-index
                # position is reconstructed separately from its seed, but
                # persistent-worker augmentation/prefetch RNG is not
                # checkpointed, so allocation-boundary resume is not bitwise
                # equivalent to an uninterrupted data stream.
                if (global_step + 1) in model_only_snapshot_steps:
                    accelerator.wait_for_everyone()
                    snapshot_path = trainer.save_model_only_snapshot(global_step + 1)
                    accelerator.wait_for_everyone()
                    if overwatch.is_rank_zero():
                        tqdm.write(f"Saved model-only snapshot to {snapshot_path}")

                if (global_step + 1) % config.train.checkpointing_steps == 0 or global_step == trainer.max_training_steps - 1:
                    accelerator.wait_for_everyone() # ensures fsdp checkpointing without deadlock
                    save_path = trainer.save_checkpoint(global_step+1)
                    accelerator.wait_for_everyone()

                    if overwatch.is_rank_zero():
                        tqdm.write(f"Saved state to {save_path}")

                progress_bar.update()
                progress_bar.set_postfix(dict(loss=losses["loss"], lr=nice(trainer.lr)))
                global_step += 1

                if stop_after_steps is not None and global_step == stop_after_steps:
                    if overwatch.is_rank_zero():
                        tqdm.write(
                            f"Training segment reached checkpoint boundary {stop_after_steps}."
                        )
                    segment_paused = global_step < trainer.max_training_steps

            if segment_paused or global_step >= trainer.max_training_steps:
                if overwatch.is_rank_zero():
                    if segment_paused:
                        tqdm.write("Training segment is pausing cleanly.")
                    else:
                        tqdm.write("Training has reached maximum steps.")
                is_max_train_steps_reached = True  # set to break outer loop
                break

        if is_max_train_steps_reached:
            break

    accelerator.wait_for_everyone()
    trainer.finalize()
    if segment_paused:
        overwatch.info(
            f"Segment paused cleanly at completed update {global_step}."
        )
    else:
        overwatch.info("Happy Ending!")
    accelerator.end_training()
    
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

def _is_slurm_job_process() -> bool:
    return "SLURM_JOB_ID" in os.environ and not os.isatty(sys.stdout.fileno())


def _load_json_config(path, overrides):
    """configs/train/*.json: ``launch_config`` names the module in psi.config.train, every other key is a field of its
    DynamicLaunchConfig. Overrides are dotted key=value pairs (values parsed as JSON), e.g. data.root_dir=/data/humego."""
    cfg = json.loads(Path(path).read_text())
    module_name = cfg.pop("launch_config")
    cfg.pop("description", None)
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override must be key=value: {item!r}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        *parents, leaf = key.split(".")
        node = cfg
        for name in parents:
            node = node.setdefault(name, {})
        node[leaf] = value
    module = importlib.import_module(f"psi.config.train.{module_name}")
    return module.DynamicLaunchConfig.model_validate(cfg)

if __name__ == "__main__":
    _verify_expected_train_argv()
    if _is_slurm_job_process():
        overwatch.info("SLURM job detected. Setting up distributed environment variables...")
        # ``srun --overlap`` steps do not consistently export
        # SLURM_JOB_NUM_NODES, while torchrun already exports WORLD_SIZE and
        # LOCAL_WORLD_SIZE.  Preserve torchrun's authoritative value and use
        # the standard SLURM_NNODES/single-node fallback only when needed.
        if "LOCAL_WORLD_SIZE" not in os.environ:
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            node_count = int(
                os.environ.get(
                    "SLURM_JOB_NUM_NODES", os.environ.get("SLURM_NNODES", "1")
                )
            )
            os.environ["LOCAL_WORLD_SIZE"] = str(world_size // node_count)

    # Print and log important environment variables
    env_vars_to_track = [
        "OMP_NUM_THREADS", "HF_HOME", "TORCH_HOME", "HF_TOKEN", "HF_LEROBOT_HOME",
        "DATA_HOME", "UV_CACHE_DIR", "WANDB_API_KEY",
        "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "CUDA_VISIBLE_DEVICES",
        "WORLD_SIZE", "LOCAL_WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT",
        "LEROBOT_VIDEO_BACKEND", "PSI_STOP_AFTER_STEPS",
        "PSI_FORMAL_ELIGIBLE", "PSI_FORMAL_RECIPE_PATH",
        "PSI_FORMAL_RECIPE_SHA256"
    ]

    env_info = {}
    overwatch.info("=== Environment Variables ===")
    for var in env_vars_to_track:
        value = os.environ.get(var, "Not Set")
        # Mask sensitive tokens for logging
        if "TOKEN" in var or "KEY" in var:
            display_value = f"{value[:3]}...{value[-4:]}" if value != "Not Set" and len(value) > 7 else value
        else:
            display_value = value
        
        env_info[var] = display_value
        overwatch.info(f"{var}: {display_value}")
    
    # Store env info for wandb logging later
    os.environ["_ENV_INFO_JSON"] = json.dumps(env_info)

    start = datetime.datetime.now()
    overwatch.info("Parsing configuration...")
    try:
        if sys.argv[1].endswith(".json"):
            config = _load_json_config(sys.argv[1], sys.argv[2:])
        else:
            # By convention, the first argument after the script name is the config module name,
            # eg., {trainer}_{data}_{model}_config, which corresponds to psi.config.train.pretrain_egodex_qwen3vl_config
            module = importlib.import_module(f"psi.config.train.{sys.argv[1]}")
            DynamicLaunchConfigClass = getattr(module, "DynamicLaunchConfig")
            config = tyro.cli(DynamicLaunchConfigClass, config=(tyro.conf.ConsolidateSubcommandArgs,), args=sys.argv[2:])
    except Exception as e:
        overwatch.error(f"Failed to load the configuration {sys.argv[1]!r}")
        raise e

    end = datetime.datetime.now()
    overwatch.info(f"Config parsing took {(end - start).total_seconds():.2f}s") 
    train(config) # type: ignore

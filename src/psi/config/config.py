# Derived from Psi0 (https://github.com/physical-superintelligence-lab/Psi0, Apache License 2.0) and modified; see NOTICE.
from __future__ import annotations
import os
from typing_extensions import Self, Annotated
from pydantic import BaseModel, Field, model_validator
import os
import tyro
from typing import Union, Any, TYPE_CHECKING, Literal
from tyro.conf import subcommand as cmd
from pathlib import Path
import datetime
from psi.config.transform import DataTransform

class LoggingConfig(BaseModel):
    logging_dir: str = "logs"
    report_to: str | None = None
    log_freq: int = 100


class WandbConfig(BaseModel):
    project: str = Field(default_factory=lambda: os.getenv("WANDB_PROJECT", "psi"))
    entity: str | None = None
    group: str | None = None
    id: str | None = None
    name: str | None = None
    tags: list[str] = Field(default_factory=list)
    resume: str = "allow"  # allow, must, never

    def model_post_init(self, __context: Any) -> None:
        if self.entity is None:
            self.entity = os.getenv("WANDB_ENTITY", None)


class TrainConfig(BaseModel):
    num_workers: int = 8
    overfit_single_batch: bool = False
    name: str = "human3d"  # "vqvae"
    resume_from_checkpoint: str | None = None
    skip_resumed_steps: bool = False

    # HF Hub Credentials (for any gated models)
    hf_token: str | Path = Path(".hf_token")  # Environment variable or Path to HF Token

    lora: bool = False
    output_dir: str = ".runs"
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "bf16"  # "no" for fp32, "fp16", "bf16"
    max_grad_norm: float | None = None  # 1.0
    optimizer_foreach: bool | None = None

    train_batch_size: int = 16
    val_batch_size: int = 16

    val_num_batches: int = 20

    checkpointing_steps: int = 5000
    max_checkpoints_to_keep: int | None = None
    validation_steps: int = 50

    learning_rate: float = 1e-5
    # linear, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup
    lr_scheduler_type: str = "cosine_with_min_lr"
    lr_scheduler_kwargs: dict[str, float | tuple[float, ...]] = Field(
        default_factory=lambda: {
            # "num_warmup_steps": 0,
            "betas": (0.9, 0.95),
            "weight_decay": 1e-8,
            "eps": 1e-8,
        }
    )
    scheduler_specific_kwargs: dict[str, float | tuple[float, ...]] = Field(
        default_factory=lambda: {"min_lr": 5.0e-07}
    )

    ## FSDP or DDP
    data_parallel: str = "ddp"  # "deepspeed", "ddp" or "fsdp"
    sharding_strategy: str = "full-shard"
    deepspeed_config: str = "src/InternVLA/config/deepseeds/zero3.json"
    enable_gradient_checkpointing: bool = True
    enable_mixed_precision_training: bool = True
    reduce_in_full_precision: bool = True

    max_training_steps: int | None = 100_000
    num_train_epochs: int | None = None
    warmup_steps: int | None = 0  # if > 0, overrides warmup_ratio
    warmup_ratio: float | None = 0.05  # used if warmup_steps is not set or 0

    @model_validator(mode="after")
    def check_warmup(self) -> Self:
        steps, ratio = self.warmup_steps, self.warmup_ratio
        if (steps is not None and steps > 0) and (ratio is not None and ratio > 0):
            raise ValueError("Only one of warmup_steps or warmup_ratio can be set")

        steps, ratio = self.max_training_steps, self.num_train_epochs
        if (steps is None or steps == 0) and (ratio is None or ratio == 0):
            raise ValueError(
                "At least one of max_training_steps or num_train_epochs must be set"
            )
        if steps is not None and ratio is not None and steps > 0 and ratio > 0:
            raise ValueError(
                "Only one of max_training_steps or num_train_epochs can be set"
            )
        return self

    def model_post_init(self, __context: Any) -> None:
        if self.lr_scheduler_type != "cosine_with_min_lr":
            self.scheduler_specific_kwargs = {}

        if not os.path.isabs(self.deepspeed_config):
            from psi.utils import resolve_path
            self.deepspeed_config = os.path.abspath(
                resolve_path(self.deepspeed_config)
            )

class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 21074
    device: str = "cuda:0"
    policy: str | None = None
    action_exec_horizon: int | None = None
    rtc: bool = False
    inference_noise_seed: int = Field(
        default=0,
        ge=0,
        le=2**32 - 1,
        description=(
            "Independent per-rollout inference RNG seed; this is deliberately "
            "separate from the checkpoint's training seed."
        ),
    )
    checkpoint_manifest: str | None = Field(
        default=None,
        description=(
            "Preregistered candidate snapshot manifest. When set, its validated "
            "model_path is loaded directly and exposed by /health."
        ),
    )
    flowdagger_adapter_path: str | None = Field(
        default=None,
        description=(
            "Path to a COMPLETE FlowDAgger noise-policy artifact. Required "
            "only when flowdagger_mode='adapter'."
        ),
    )
    flowdagger_record_dir: str | None = Field(
        default=None,
        description=(
            "Optional append-only metadata directory used by the pilot "
            "collect/adapter server modes."
        ),
    )
    flowdagger_mode: Literal["base", "collect", "adapter"] = Field(
        default="base",
        description=(
            "FlowDAgger deployment mode. 'base' preserves legacy /act, "
            "'collect' exposes exact sampled noise, and 'adapter' selects "
            "noise with the frozen adapter."
        ),
    )
    visual_dynamics_mode: Literal["off", "always", "ood"] = Field(
        default="off",
        description=(
            "Official-LPB VisualDynamicsModel steering mode. 'always' is the "
            "exploratory Psi0+dynamics method; 'ood' additionally requires a "
            "calibrated current-state barrier threshold."
        ),
    )
    visual_dynamics_checkpoint: str | None = Field(
        default=None,
        description="Final official-LPB VisualDynamicsModel model_100.pth.",
    )
    visual_dynamics_feature_root: str | None = Field(
        default=None,
        description="Directory containing frozen Psi0 feature_shards.",
    )
    visual_dynamics_transition_index: str | None = Field(
        default=None,
        description=(
            "feature_extraction_input.jsonl used to select expert-only memory."
        ),
    )
    visual_dynamics_threshold: float | None = Field(
        default=None,
        ge=0.0,
        description="Calibrated current visual barrier threshold for OOD mode.",
    )
    visual_dynamics_guidance_scale: float = Field(default=0.05, ge=0.0)
    visual_dynamics_guidance_steps: int = Field(default=3, ge=1, le=10)
    action_attention_dir: str | None = Field(
        default=None,
        description="Optional evaluation attention reports; disabled when unset.",
    )
    action_attention_every: int = Field(default=10, ge=1)
    action_attention_max_records: int = Field(default=20, ge=1)
    action_attention_layers: list[int] | None = None
    action_attention_steps: list[int] = Field(default_factory=lambda: [0])
    action_attention_heads: list[int] | None = None
    action_attention_queries: list[int] | None = None
    run_dir: str 
    ckpt_step: int 

    @model_validator(mode="after")
    def set_policy(self):
        for name in ("layers", "steps", "heads", "queries"):
            indices = getattr(self, f"action_attention_{name}")
            if indices is not None and (
                not indices or min(indices) < 0 or len(set(indices)) != len(indices)
            ):
                raise ValueError(f"action_attention_{name} must contain unique non-negative indices")
        if any(step >= 10 for step in self.action_attention_steps):
            raise ValueError("action_attention_steps must be below the server's 10 inference steps")
        if self.action_attention_dir is not None and self.visual_dynamics_mode != "off":
            raise ValueError("Action attention reports require ordinary decoding without visual dynamics guidance")
        if self.flowdagger_mode == "adapter" and not self.flowdagger_adapter_path:
            raise ValueError(
                "flowdagger_adapter_path is required in adapter mode"
            )
        if self.flowdagger_mode == "collect" and not self.flowdagger_record_dir:
            raise ValueError(
                "flowdagger_record_dir is required in collect mode"
            )
        if self.rtc and self.flowdagger_mode == "adapter":
            raise ValueError(
                "FlowDAgger non-RTC adapter cannot be combined with RTC"
            )
        if self.visual_dynamics_mode != "off":
            required = {
                "visual_dynamics_checkpoint": self.visual_dynamics_checkpoint,
                "visual_dynamics_feature_root": self.visual_dynamics_feature_root,
                "visual_dynamics_transition_index": (
                    self.visual_dynamics_transition_index
                ),
            }
            missing = [name for name, value in required.items() if not value]
            if missing:
                raise ValueError(
                    "VisualDynamicsModel steering is missing: " + ", ".join(missing)
                )
            if self.rtc:
                raise ValueError(
                    "VisualDynamicsModel flow guidance supports non-RTC inference only"
                )
            if self.flowdagger_mode != "base":
                raise ValueError(
                    "VisualDynamicsModel steering cannot be combined with "
                    "FlowDAgger collect/adapter modes"
                )
            if (
                self.visual_dynamics_mode == "ood"
                and self.visual_dynamics_threshold is None
            ):
                raise ValueError(
                    "visual_dynamics_threshold is required in OOD mode"
                )
        if self.policy is None:
            run_dir_path = Path(self.run_dir)
            self.policy = run_dir_path.parts[1]
        return self
        

class DataConfig(BaseModel):
    transform: DataTransform

    def __call__(self, split: str = "train", transform_kwargs={}, **kwargs) -> Any:
        raise NotImplementedError
    
class ModelConfig(BaseModel): 
    ...

class DummyModelConfig(ModelConfig):
    # boilerplate model config
    ...
class LaunchConfig(BaseModel):
    # NOTE: This class is used for type hinting only!
    # The real implmentation is DynamicLaunchConfig .

    exp: str = Field(..., description="Name of the experiment", frozen=True)
    seed: int | None = None
    auto_tag_run: bool = False
    eval: bool = False
    debug: bool = False
    timestamp: str | None = None

    log: LoggingConfig
    wandb: WandbConfig
    train: TrainConfig
    data: DataConfig
    model: ModelConfig
    
    def model_post_init(self, __context: Any) -> None:
        is_multinode = (
            "SLURM_NODELIST" in os.environ and
            len(os.environ["SLURM_NODELIST"].split(",")) > 1
        )
        if is_multinode:
            assert self.timestamp is not None, "Timestamp must be provided for multi-node training, eg., --timestamp=$(date +\"%y%m%d%H%M\")"

        def extract_timestamp(folder_name):
            parts = folder_name.split('.')
            return parts[-1] if len(parts) > 1 else ''

        if self.train.resume_from_checkpoint == "latest":
            """ auto resume by looking up timestamp or latest run folder """
            auto_resume_success = False
            trainer_dir = os.path.join(self.train.output_dir, self.train.name)
            if os.path.exists(trainer_dir):
                # Sort folders by timestamp (assumed to be last part after a dot)
                runs = dict(sorted({
                    extract_timestamp(f): os.path.join(trainer_dir, f) for f in os.listdir(trainer_dir)
                }.items(), reverse=True))

                if self.timestamp is not None and self.timestamp in runs:
                    print(f"Will resume latest run with specified timestamp: {self.timestamp}")
                    self.train.resume_from_checkpoint = runs[self.timestamp]
                    auto_resume_success = True

                """ elif len(runs) > 0:
                    latest_timestamp = next(iter(runs))
                    print(f"Will auto-resume from latest run with timestamp: {latest_timestamp}")
                    if self.timestamp is not None and self.timestamp != latest_timestamp:
                        print(f"Overriding timestamp {self.timestamp} with latest timestamp {latest_timestamp}")
                    self.timestamp = latest_timestamp
                    self.train.resume_from_checkpoint = runs[latest_timestamp]
                    auto_resume_success = True """
            
            if not auto_resume_success:
                self.train.resume_from_checkpoint = None

        if self.timestamp is None:
            self.timestamp = datetime.datetime.now().strftime("%y%m%d%H%M")

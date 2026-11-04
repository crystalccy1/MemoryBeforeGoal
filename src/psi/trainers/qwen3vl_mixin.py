# Derived from Psi0 (https://github.com/physical-superintelligence-lab/Psi0, Apache License 2.0) and modified; see NOTICE.
from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, cast
import re
import torch
import torch.nn as nn
import numpy as np
if TYPE_CHECKING:
    from psi.trainers import Trainer
    from psi.config.model_psi0 import Psi0ModelConfig
    from psi.config.model_qwen3vl import Qwen3VLModelConfig

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, Qwen2TokenizerFast, Qwen3VLProcessor
from accelerate import Accelerator
from torch.nn.utils.rnn import pad_sequence
# from psi.config.tokenizer import BinActionTokenizerConfig
from psi.utils import initialize_overwatch #, shorten, seed_everything
overwatch = initialize_overwatch(__name__)

class PaddedCollatorForActionPrediction:
    def __init__(self, model_max_length, pad_token_id, padding_side="right", pixel_values_dtype=torch.float32):
        self.model_max_length = model_max_length
        self.pad_token_id = pad_token_id
        self.padding_side = padding_side
        self.pixel_values_dtype = pixel_values_dtype

    def __call__(self, instances):
        
        # Extract sequences
        input_ids = [instance["input_ids"].squeeze(0) if instance["input_ids"].dim() == 2 else instance["input_ids"] 
                    for instance in instances]
        labels = [instance["labels"].squeeze(0) if instance["labels"].dim() == 2 else instance["labels"] 
                for instance in instances]
        pixel_values = [instance["pixel_values"] for instance in instances]
        dataset_names = [instance["dataset_name"] for instance in instances] if "dataset_name" in instances[0] else None

        # Only support right padding for now
        assert self.padding_side == "right", f"Invalid Tokenizer padding_side={self.padding_side}"

        # Pad sequences
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        # Truncate if longer than model_max_length
        input_ids = input_ids[:, :self.model_max_length]
        labels = labels[:, :self.model_max_length]

        # Attention mask based on pad token
        attention_mask = input_ids.ne(self.pad_token_id)

        # Stack pixel_values
        if isinstance(pixel_values[0], torch.Tensor):
            pixel_values = torch.stack(pixel_values).to(dtype=self.pixel_values_dtype)
        elif isinstance(pixel_values[0], dict):
            pixel_values = {k: torch.stack([pv[k] for pv in pixel_values]) for k in pixel_values[0]}
        else:
            raise ValueError(f"Unsupported pixel_values type: {type(pixel_values[0])}")

        # Stack image_grid_thw
        image_grid_thw = torch.stack([instance["image_grid_thw"].squeeze(0) for instance in instances])

        # Build output
        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values, 
            "image_grid_thw": image_grid_thw,
        }
        if dataset_names is not None:
            output["dataset_name"] = dataset_names

        raw_actions = np.stack([instance["raw_actions"] for instance in instances]) if "raw_actions" in instances[0] else None
        if raw_actions is not None:
            output["raw_actions"] = raw_actions

        actions_mask = torch.stack([instance["actions_mask"] for instance in instances]) if "actions_mask" in instances[0] else None
        if actions_mask is not None:
            output["actions_mask"] = actions_mask

        raw_images = torch.stack([torch.from_numpy(np.array(instance["raw_images"])) for instance in instances]) if "raw_images" in instances[0] else None
        if raw_images is not None:
            output["raw_images"] = raw_images

        return output

class PaddedCollatorForTogether:
    def __init__(self, model_max_length, pad_token_id, padding_side="right", pixel_values_dtype=torch.float32):
        self.model_max_length = model_max_length
        self.pad_token_id = pad_token_id
        self.padding_side = padding_side
        self.pixel_values_dtype = pixel_values_dtype

    def __call__(self, instances):
        
        # Extract sequences
        input_ids = [instance["input_ids"].squeeze(0) if instance["input_ids"].dim() == 2 else instance["input_ids"] 
                    for instance in instances]

        pixel_values = [instance["pixel_values"] for instance in instances]
        dataset_names = [instance["dataset_name"] for instance in instances] if "dataset_name" in instances[0] else None

        # Only support right padding for now
        assert self.padding_side == "right", f"Invalid Tokenizer padding_side={self.padding_side}"

        # Pad sequences
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)

        # Truncate if longer than model_max_length
        input_ids = input_ids[:, :self.model_max_length]

        # Attention mask based on pad token
        attention_mask = input_ids.ne(self.pad_token_id)

        # Stack pixel_values
        if isinstance(pixel_values[0], torch.Tensor):
            pixel_values = torch.stack(pixel_values).to(dtype=self.pixel_values_dtype)
        elif isinstance(pixel_values[0], dict):
            pixel_values = {k: torch.stack([pv[k] for pv in pixel_values]) for k in pixel_values[0]}
        else:
            raise ValueError(f"Unsupported pixel_values type: {type(pixel_values[0])}")

        # Stack image_grid_thw
        image_grid_thw = torch.stack([instance["image_grid_thw"].squeeze(0) for instance in instances])

        # Build output
        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values, 
            "image_grid_thw": image_grid_thw,
        }
        if dataset_names is not None:
            output["dataset_name"] = dataset_names

        raw_actions = np.stack([instance["raw_actions"] for instance in instances]) if "raw_actions" in instances[0] else None
        if raw_actions is not None:
            output["raw_actions"] = raw_actions

        # print(type(instances[0]["actions_mask"]), dataset_names)
        actions_mask = torch.stack([torch.from_numpy(instance["actions_mask"]) for instance in instances]) if "actions_mask" in instances[0] else None
        if actions_mask is not None:
            output["actions_mask"] = actions_mask

        raw_images = torch.stack([torch.from_numpy(np.array(instance["raw_images"])) for instance in instances]) if "raw_images" in instances[0] else None
        if raw_images is not None:
            output["raw_images"] = raw_images

        actions = torch.stack([torch.from_numpy(np.array(instance["actions"])) for instance in instances]) if "actions" in instances[0] else None
        if actions is not None:
            output["actions"] = actions

        states = torch.stack([torch.from_numpy(np.array(instance["states"])) for instance in instances]) if "states" in instances[0] else None
        if states is not None:
            output["states"] = states

        return output

class PaddedCollatorForFastWAM(PaddedCollatorForTogether):
    """Collate the additional fixed-shape video inputs used by FastWAM."""

    @staticmethod
    def _as_tensor(value):
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value)
        return torch.as_tensor(value)

    def __call__(self, instances):
        required = ("video", "image_is_pad", "action_is_pad")
        for index, instance in enumerate(instances):
            missing = [key for key in required if key not in instance]
            if missing:
                raise KeyError(
                    f"FastWAM instance {index} is missing required fields: {missing}"
                )

        output = super().__call__(instances)

        videos = torch.stack(
            [self._as_tensor(instance["video"]) for instance in instances]
        ).to(dtype=self.pixel_values_dtype)
        image_is_pad = torch.stack(
            [self._as_tensor(instance["image_is_pad"]) for instance in instances]
        ).to(dtype=torch.bool)
        action_is_pad = torch.stack(
            [self._as_tensor(instance["action_is_pad"]) for instance in instances]
        ).to(dtype=torch.bool)

        if videos.ndim != 5 or videos.shape[1] != 3:
            raise ValueError(
                "Collated FastWAM video must have shape (B,3,T,H,W), "
                f"got {tuple(videos.shape)}."
            )
        if image_is_pad.shape != (videos.shape[0], videos.shape[2]):
            raise ValueError(
                "image_is_pad must have shape (B,T_video), got "
                f"{tuple(image_is_pad.shape)} for video {tuple(videos.shape)}."
            )
        if action_is_pad.ndim != 2:
            raise ValueError(
                "action_is_pad must have shape (B,T_action), got "
                f"{tuple(action_is_pad.shape)}."
            )
        if (
            "actions" in output
            and action_is_pad.shape != output["actions"].shape[:2]
        ):
            raise ValueError(
                "action_is_pad shape must match collated action time dimensions, "
                f"got {tuple(action_is_pad.shape)} for actions "
                f"{tuple(output['actions'].shape)}."
            )

        output["video"] = videos
        output["image_is_pad"] = image_is_pad
        output["action_is_pad"] = action_is_pad
        return output

class QwenBinActionTokenizer:

    def __init__(
        self, tokenizer: Qwen2TokenizerFast, bins: int = 256, min_action: int = -1, max_action: int = 1
    ) -> None:
        self.tokenizer, self.n_bins, self.min_action, self.max_action = tokenizer, bins, min_action, max_action

        # Create Uniform Bins + Compute Bin Centers
        self.bins = np.linspace(min_action, max_action, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        self.action_token_begin_idx: int = len(tokenizer) # avoid conflict with existing tokens

    def __call__(self, action: np.ndarray) -> tuple[List[int], int]:
        action = np.clip(action, a_min=float(self.min_action), a_max=float(self.max_action))
        discretized_action = np.digitize(action, self.bins)
        discretized_action = discretized_action.flatten()
        action_token_ids = (self.action_token_begin_idx + discretized_action).tolist()
        return action_token_ids
    
    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        """
        Returns continuous actions for discrete action token IDs.
        """
        discretized_actions = action_token_ids - self.action_token_begin_idx   #self.tokenizer.vocab_size - action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        return self.bin_centers[discretized_actions]

    @property
    def vocab_size(self) -> int:
        return self.n_bins

class Qwen3vlMixin:

    def init_qwen3vl_models(self):
        trainer = cast("Trainer", self)

        model_cfg: Qwen3VLModelConfig = trainer.cfg.model  # type: ignore
        overwatch.info(f"Loading pretrained from {model_cfg.model_name_or_path}")

        try:
            # default: Load the model on the available device(s)
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_cfg.model_name_or_path,
                attn_implementation="flash_attention_2",
                dtype=torch.bfloat16,
                local_files_only=True 
            )
            overwatch.info(f"Load VLM from {model_cfg.model_name_or_path} successfully.")
        except Exception as e:
            # safe for multi-node training, please download weights in advance
            overwatch.critical(f"Failed to load VLM from {model_cfg.model_name_or_path}. \n"
                               f"Please download weights in advance using: 'python scripts/predownload_qwen3vl.py'\n"
                               f"Auto downloading is disabled in multi-node training to prevent conflicts. \n")
            raise e

        if not hasattr(self.model.config, "hidden_size"):
            self.model.config.hidden_size = self.model.config.text_config.hidden_size

        self.vlm_processor = AutoProcessor.from_pretrained(model_cfg.model_name_or_path)
        self.tokenizer = self.vlm_processor.tokenizer

        if trainer.cfg.train.lora:
            from peft import LoraConfig, get_peft_model, TaskType
            print("LoRA enabled")

            for p in self.model.parameters():
                p.requires_grad = False

            lora_config = LoraConfig(
                r=model_cfg.lora_r or 64,
                lora_alpha=model_cfg.lora_alpha or 128,
                lora_dropout=model_cfg.lora_dropout or 0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen 的 attention 线性层
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.model = get_peft_model(self.model, lora_config)
        else:
            if model_cfg.tune_mm_vision:
                for n, p in self.model.visual.named_parameters():
                    p.requires_grad = True
            else:
                for n, p in self.model.visual.named_parameters():
                    p.requires_grad = False

            if model_cfg.tune_mm_mlp:
                for n, p in self.model.visual.merger.named_parameters():
                    p.requires_grad = True
            else:
                for n, p in self.model.visual.merger.named_parameters():
                    p.requires_grad = False

            if model_cfg.tune_mm_llm:
                for n, p in self.model.language_model.named_parameters():
                    p.requires_grad = True
                self.model.lm_head.requires_grad = True # type:ignore
            else:
                for n, p in self.model.language_model.named_parameters():
                    p.requires_grad = False
                self.model.lm_head.requires_grad = False # type:ignore
    
    def prepare_qwen3vl(self, accelerator: Accelerator):
        trainer = cast("Trainer", self)
        self.model, self.optimizer, self.lr_scheduler = accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler
        )

        if trainer.train_cfg.data_parallel == "deepspeed":
            # Gradient Checkpoint Setup
            if trainer.model_cfg.gradient_checkpointing: # type: ignore
                assert "DeepSpeedEngine" in trainer.model.__class__.__name__, "deepspeed is not properly initialized!"
                if hasattr(trainer.model, "enable_input_require_grads"): 
                    trainer.model.enable_input_require_grads()
                else:
                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)
                    trainer.model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        
        if trainer.cfg.train.overfit_single_batch:
            overwatch.warning("Overfitting a single batch: reusing first batch every step. set cfg.data.image_aug = False for true memorization.")
            first_batch = next(iter(self.train_dataloader))
            class SingleBatchLoader:
                def __iter__(self): 
                    while True:
                        yield first_batch
                def __len__(self):
                    return 1
            self.train_dataloader = SingleBatchLoader()
            
        self.accelerator = accelerator
        return self.train_dataloader


    def create_qwen3vl_optimizer(self):
        trainer = cast("Trainer", self)
        opt_model = self.model
        decay_parameters = self.get_decay_parameter_names(opt_model)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]

        model_cfg: Qwen3VLModelConfig | Psi0ModelConfig = trainer.cfg.model  # type: ignore
        if model_cfg.mm_projector_lr is not None and model_cfg.mm_projector_lr != 0:
            projector_parameters = [
                name for name, _ in opt_model.named_parameters() if "merger" in name
            ]
            if model_cfg.vision_tower_lr is not None and model_cfg.vision_tower_lr != 0:
                vision_tower_parameters = [
                    name for name, _ in opt_model.named_parameters() if "visual" in name
                ]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and n not in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": model_cfg.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and n in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": model_cfg.weight_decay,
                        "lr": model_cfg.vision_tower_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and n not in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and n in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": model_cfg.vision_tower_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": model_cfg.weight_decay,
                        "lr": model_cfg.mm_projector_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": model_cfg.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": model_cfg.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": model_cfg.weight_decay,
                        "lr": model_cfg.mm_projector_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": model_cfg.mm_projector_lr,
                    },
                ]
        else:
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": model_cfg.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]
        return optimizer_grouped_parameters

    ####  adatped from transformers.Trainer  ####
    def get_decay_parameter_names(self, model) -> list[str]:
        """
        Get all parameter names that weight decay will be applied to.

        This function filters out parameters in two ways:
        1. By layer type (instances of layers specified in ALL_LAYERNORM_LAYERS)
        2. By parameter name patterns (containing 'bias', or variation of 'norm')
        """
        forbidden_name_patterns = [r"bias", r"layernorm", r"rmsnorm", r"(?:^|\.)norm(?:$|\.)", r"_norm(?:$|\.)"]
        decay_parameters = self.get_parameter_names(model, [nn.LayerNorm], forbidden_name_patterns)
        return decay_parameters

    def get_parameter_names(self, model, forbidden_layer_types, forbidden_layer_names=None):
        """
        Returns the names of the model parameters that are not inside a forbidden layer.
        """
        forbidden_layer_patterns = (
            [re.compile(pattern) for pattern in forbidden_layer_names] if forbidden_layer_names is not None else []
        )
        result = []
        for name, child in model.named_children():
            child_params = self.get_parameter_names(child, forbidden_layer_types, forbidden_layer_names)
            result += [
                f"{name}.{n}"
                for n in child_params
                if not isinstance(child, tuple(forbidden_layer_types))
                and not any(pattern.search(f"{name}.{n}".lower()) for pattern in forbidden_layer_patterns)
            ]
        # Add model specific parameters that are not in any child
        result += [
            k for k in model._parameters if not any(pattern.search(k.lower()) for pattern in forbidden_layer_patterns)
        ]

        return result

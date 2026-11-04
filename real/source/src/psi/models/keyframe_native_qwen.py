from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from torchvision.transforms import v2

from psi.config.augmentation import CenterCrop, ResizeImage


class QwenVisionMergerNativeFrameProducer:
    """Extract the native 7x11 Qwen vision-merger grid without spatial pooling.

    This is a separate data contract from strict 16-token FrameSamp. Features
    retain Qwen's merger token order; they are not full-VLM hidden states.
    Singleton calls preserve the existing offline/online batching convention.
    The number of historical frames is a selection policy, not an encoder
    parameter, so the same bank can support seven-frame causal selection.
    """

    SCHEMA_VERSION = "psi0-keyframe-native-qwen-producer/v1"

    def __init__(
        self,
        *,
        model: Any,
        processor: Any,
        device: str | torch.device,
        resize_hw: tuple[int, int] = (180, 320),
        expected_grid_hw: tuple[int, int] = (7, 11),
        feature_dim: int = 2048,
    ) -> None:
        if tuple(resize_hw) != (180, 320):
            raise ValueError("native77 preprocessing requires resize_hw=(180, 320)")
        if tuple(expected_grid_hw) != (7, 11):
            raise ValueError("native77 requires expected_grid_hw=(7, 11)")
        if feature_dim != 2048:
            raise ValueError("native77 requires feature_dim=2048")
        self.model = model
        self.processor = processor
        self.device = torch.device(device)
        self.resize_hw = (180, 320)
        self.expected_grid_hw = (7, 11)
        self.tokens_per_frame = 77
        self.feature_dim = 2048
        self.attention_backend = getattr(
            getattr(model, "config", None), "_attn_implementation", None
        )
        self.spatial_merge_size = int(model.model.visual.spatial_merge_size)
        if self.spatial_merge_size <= 0:
            raise ValueError("Qwen spatial_merge_size must be positive")
        self.transform = v2.Compose(
            [
                ResizeImage(size=self.resize_hw)(),
                CenterCrop(size=self.resize_hw)(),
            ]
        )

    @staticmethod
    def _as_rgb_pil(frame: np.ndarray | Image.Image) -> Image.Image:
        if isinstance(frame, Image.Image):
            return frame.convert("RGB")
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(f"raw RGB frame must have shape [H,W,3], got {array.shape}")
        if array.dtype != np.uint8:
            raise TypeError(f"raw RGB frame must use uint8, got {array.dtype}")
        return Image.fromarray(np.ascontiguousarray(array))

    @torch.inference_mode()
    def encode(
        self,
        frames: Sequence[np.ndarray | Image.Image] | np.ndarray,
        instruction: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(frames, np.ndarray) and frames.ndim == 3:
            items: list[np.ndarray | Image.Image] = [frames]
        else:
            items = list(frames)
        if len(items) != 1:
            raise ValueError("Native Qwen producer encodes exactly one raw frame per call")
        image = self.transform(self._as_rgb_pil(items[0]))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": str(instruction)},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(
            [messages], image_patch_size=16
        )
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        image_grid_thw = inputs["image_grid_thw"].to(self.device)
        if tuple(image_grid_thw.shape) != (1, 3):
            raise RuntimeError("Native Qwen requires exactly one image_grid_thw row")
        if image_grid_thw.dtype not in (torch.int32, torch.int64):
            raise RuntimeError("Native Qwen image_grid_thw must have an integer dtype")
        temporal, raw_h, raw_w = (int(value.item()) for value in image_grid_thw[0])
        if temporal != 1:
            raise RuntimeError("Native Qwen image temporal grid must equal one")
        if raw_h % self.spatial_merge_size or raw_w % self.spatial_merge_size:
            raise RuntimeError("Qwen image grid is not divisible by spatial_merge_size")
        grid_hw = (raw_h // self.spatial_merge_size, raw_w // self.spatial_merge_size)
        if grid_hw != self.expected_grid_hw:
            raise RuntimeError(
                f"Native Qwen merged grid must be {self.expected_grid_hw}, got {grid_hw}; "
                "check the checkpoint and image-processor preprocessing contract"
            )
        image_features, _ = self.model.get_image_features(
            pixel_values=inputs["pixel_values"].to(self.device),
            image_grid_thw=image_grid_thw,
        )
        if len(image_features) != 1:
            raise RuntimeError("Qwen returned the wrong number of image feature groups")
        feature = image_features[0]
        expected_shape = (self.tokens_per_frame, self.feature_dim)
        if not isinstance(feature, torch.Tensor) or tuple(feature.shape) != expected_shape:
            actual_shape = getattr(feature, "shape", None)
            raise RuntimeError(
                f"Unexpected native Qwen vision-merger feature shape: {actual_shape} "
                f"!= {expected_shape}"
            )
        if not feature.is_floating_point():
            raise RuntimeError("Qwen vision-merger features must have a floating-point dtype")
        # Preserve all native tokens in their original order, with no pooling,
        # interpolation, replication, or spatial transposition.
        native = feature.to(dtype=torch.bfloat16, device="cpu").contiguous()
        return native.unsqueeze(0), torch.tensor([grid_hw], dtype=torch.int64)

    def parity_contract(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "encoder": "psi0_qwen3vl_vision_merger",
            "feature_tap": "Qwen3VL.get_image_features vision-merger output",
            "attention_backend": self.attention_backend,
            "input_color_space": "RGB",
            "input_dtype": "uint8",
            "resize_hw": list(self.resize_hw),
            "center_crop_hw": list(self.resize_hw),
            "resize_interpolation": "nearest",
            "vision_info_image_patch_size": 16,
            "spatial_merge_size": self.spatial_merge_size,
            "expected_grid_hw": list(self.expected_grid_hw),
            "tokens_per_frame": self.tokens_per_frame,
            "feature_dim": self.feature_dim,
            "pool": "none",
            "token_order": "native_qwen_vision_merger",
            "batching_policy": "canonical_singleton_per_frame",
            "output_dtype": "bfloat16",
            "output_device": "cpu",
            "instruction_conditions_features": False,
        }


def qwen_native_frame_producer_source_path() -> Path:
    """Return the implementation module for feature-bank provenance binding."""

    path = Path(__file__).resolve(strict=True)
    if path.name != "keyframe_native_qwen.py":
        raise RuntimeError(f"Unexpected native Qwen producer module: {path}")
    return path

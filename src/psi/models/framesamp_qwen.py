from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from qwen_vl_utils import process_vision_info
from torchvision.transforms import v2

from psi.config.augmentation import CenterCrop, ResizeImage


class QwenVisionMergerFrameProducer:
    """Canonical offline/online producer for strict Psi-0 FrameSamp v2.

    Both feature-bank generation and deployment call this implementation.
    Encoding is intentionally singleton-per-frame: the released Qwen merger
    BF16 output was observed to be batch-shape dependent on MI210, so batching
    would break offline/online feature parity.
    """

    SCHEMA_VERSION = "psi0-framesamp-qwen-producer/v2"

    def __init__(
        self,
        *,
        model: Any,
        processor: Any,
        device: str | torch.device,
        resize_hw: tuple[int, int] = (180, 320),
        tokens_per_frame: int = 16,
        feature_dim: int = 2048,
    ) -> None:
        target_side = int(round(tokens_per_frame**0.5))
        if target_side * target_side != tokens_per_frame:
            raise ValueError("tokens_per_frame must be a square number")
        self.model = model
        self.processor = processor
        self.device = torch.device(device)
        self.resize_hw = tuple(int(value) for value in resize_hw)
        self.tokens_per_frame = int(tokens_per_frame)
        self.feature_dim = int(feature_dim)
        self.target_side = target_side
        self.attention_backend = getattr(
            getattr(model, "config", None), "_attn_implementation", None
        )
        self.spatial_merge_size = int(model.model.visual.spatial_merge_size)
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
        return Image.fromarray(np.ascontiguousarray(array), mode="RGB")

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
            raise ValueError(
                "Canonical FrameSamp producer encodes exactly one raw frame per call"
            )
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
        pixel_values = inputs["pixel_values"].to(self.device)
        image_grid_thw = inputs["image_grid_thw"].to(self.device)
        image_features, _ = self.model.get_image_features(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        if len(image_features) != 1:
            raise RuntimeError("Qwen returned the wrong number of image feature groups")
        feature = image_features[0]
        grid = image_grid_thw[0]
        if int(grid[0].item()) != 1:
            raise RuntimeError(f"FrameSamp image temporal grid must equal one: {grid}")
        merged_h = int(grid[1].item()) // self.spatial_merge_size
        merged_w = int(grid[2].item()) // self.spatial_merge_size
        expected = (merged_h * merged_w, self.feature_dim)
        if tuple(feature.shape) != expected:
            raise RuntimeError(
                "Unexpected Qwen vision-merger feature shape: "
                f"{tuple(feature.shape)} != {expected}"
            )
        spatial = feature.float().transpose(0, 1).reshape(
            1, self.feature_dim, merged_h, merged_w
        )
        pooled = F.adaptive_avg_pool2d(
            spatial, output_size=(self.target_side, self.target_side)
        )
        pooled = (
            pooled.reshape(self.feature_dim, -1)
            .transpose(0, 1)
            .to(dtype=torch.bfloat16, device="cpu")
            .contiguous()
        )
        grid_hw = torch.tensor([[merged_h, merged_w]], dtype=torch.int64)
        return pooled.unsqueeze(0), grid_hw

    def parity_contract(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "encoder": "psi0_qwen3vl_vision_merger",
            "attention_backend": self.attention_backend,
            "resize_hw": list(self.resize_hw),
            "resize_interpolation": "nearest",
            "tokens_per_frame": self.tokens_per_frame,
            "feature_dim": self.feature_dim,
            "pool": "float32_adaptive_mean_4x4",
            "batching_policy": "canonical_singleton_per_frame",
            "output_dtype": "bfloat16",
        }


def qwen_frame_producer_source_path() -> Path:
    """Return the defining module, never a decorator-generated wrapper file."""

    path = Path(__file__).resolve(strict=True)
    if path.name != "framesamp_qwen.py":
        raise RuntimeError(f"Unexpected FrameSamp Qwen producer module: {path}")
    return path

"""Streaming (online) SAM3 text-prompt tracking for the SAM3-v2 memory runtime.

Wraps the pinned causal adapter (sam_adapter.create_predictor) so that frames can be pushed one at a time.
- A StreamingFrameLoader stands in for SAM3's image-folder loader. It is used only for resource paths that start
  with "stream://".
- It has a fixed capacity and preprocesses each frame exactly like AsyncImageFrameLoader:
  PIL RGB -> TF.resize(image_size) -> to_tensor -> fp16 -> (x - mean) / std -> cuda unless offloaded.
- Each prompt keeps its own causal session, as the offline tracker does. Its propagate_in_video generator is advanced
  by exactly one frame per push.
- The adapter enforces detector prefetch off and hotstart_delay 0, so no frame is read before it is pushed.
  Reading one early raises an error instead of silently using a missing frame.

Shared image backbone (share_backbone=True): every prompt session runs the detector's image backbone on the same
frame. The first prompt of a frame computes it; the other prompts of that frame reuse its output after checking with
torch.equal that the backbone input is the same image. Only the containers are copied; nothing downstream writes
into these tensors (checked by the exact parity test, which compares every object with the offline tracker).

Frames can be tracked in slices: push(rgb), then step_prompt(k) for every prompt in order, then collect().
step(rgb) does all three at once. Both give the same result; the offline parity test
(real-sonic-sam3v2-sparse-20260930/tools/stream_parity.py) matched every object of the offline tracker exactly.
"""
from __future__ import annotations

import time
import uuid

import numpy as np
import torch

STREAM_PREFIX = "stream://"


class StreamingFrameLoader:
    def __init__(self, capacity, image_size, offload_video_to_cpu, img_mean, img_std):
        self.capacity = int(capacity)
        self.image_size = int(image_size)
        self.offload = bool(offload_video_to_cpu)
        self.img_mean = torch.tensor(img_mean, dtype=torch.float16)[:, None, None]
        self.img_std = torch.tensor(img_std, dtype=torch.float16)[:, None, None]
        self.images = [None] * self.capacity
        self.count = 0
        self.video_height = None
        self.video_width = None

    def push(self, rgb: np.ndarray) -> int:
        from PIL import Image
        import torchvision.transforms.functional as TF
        if self.count >= self.capacity:
            raise RuntimeError("stream capacity exceeded")
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("push expects an HxWx3 uint8 RGB frame")
        pil = Image.fromarray(rgb, mode="RGB")
        img = TF.to_tensor(TF.resize(pil, size=(self.image_size, self.image_size))).to(dtype=torch.float16)
        img -= self.img_mean
        img /= self.img_std
        if not self.offload:
            img = img.cuda()
        self.images[self.count] = img
        self.video_height, self.video_width = pil.height, pil.width
        self.count += 1
        return self.count - 1

    def __getitem__(self, index):
        if index >= self.count or self.images[index] is None:
            raise RuntimeError(f"SAM3 requested frame {index} before it was pushed ({self.count} available)")
        return self.images[index]

    def __len__(self):
        return self.capacity


_LOADERS = {}
_PATCHED = False
_SHARED = {"key": None, "cache_key": None, "image": None, "out": None, "hits": 0, "misses": 0}


def _copy_containers(value):
    if isinstance(value, dict):
        return {k: _copy_containers(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_copy_containers(v) for v in value)
    return value


def _install_backbone_share(predictor):
    backbone = predictor.model.detector.backbone
    if getattr(backbone, "_stream_shared_forward_image", False):
        return
    original = backbone.forward_image

    def forward_image(image, *args, **kwargs):
        key = _SHARED["key"]
        if key is None or args or kwargs:
            return original(image, *args, **kwargs)
        cached = _SHARED["image"]
        if (_SHARED["cache_key"] == key and cached is not None and cached.shape == image.shape
                and cached.dtype == image.dtype and cached.device == image.device and torch.equal(cached, image)):
            _SHARED["hits"] += 1
            return _copy_containers(_SHARED["out"])
        out = original(image)
        _SHARED.update(cache_key=key, image=image, out=out)
        _SHARED["misses"] += 1
        return _copy_containers(out)

    backbone.forward_image = forward_image
    backbone._stream_shared_forward_image = True


def shared_backbone_stats():
    return {"hits": _SHARED["hits"], "misses": _SHARED["misses"]}


def _install_patch():
    global _PATCHED
    if _PATCHED:
        return
    import sam3.model.sam3_video_inference as vi
    original = vi.load_resource_as_video_frames

    def load(resource_path, image_size, offload_video_to_cpu, img_mean=(0.5, 0.5, 0.5), img_std=(0.5, 0.5, 0.5), **kwargs):
        if isinstance(resource_path, str) and resource_path.startswith(STREAM_PREFIX):
            loader = _LOADERS[resource_path]
            if loader.image_size != image_size or loader.offload != bool(offload_video_to_cpu):
                raise ValueError("stream loader created with other image size or offload setting")
            return loader, loader.video_height, loader.video_width
        return original(resource_path, image_size, offload_video_to_cpu, img_mean, img_std, **kwargs)

    vi.load_resource_as_video_frames = load
    _PATCHED = True


class StreamingTracker:
    """One causal SAM3 session per prompt; step(rgb) returns the encoded objects of that frame, per prompt."""

    def __init__(self, predictor, prompts, namespace, capacity=4096, offload=True, share_backbone=False):
        import sam_adapter
        _install_patch()
        self.share_backbone = bool(share_backbone)
        if self.share_backbone:
            _install_backbone_share(predictor)
        self.predictor, self.prompts, self.namespace = predictor, list(prompts), namespace
        self.offload = bool(offload)
        model = predictor.model
        self.key = STREAM_PREFIX + uuid.uuid4().hex
        _LOADERS[self.key] = self.loader = StreamingFrameLoader(capacity, model.image_size, self.offload,
                                                                model.image_mean, model.image_std)
        self.encode = sam_adapter._encode_objects
        self.sessions, self.generators = [], []
        self.frames = 0
        self.pending = None
        self.objects = None

    def _start(self):
        for index, prompt in enumerate(self.prompts):
            sid = self.predictor.handle_request({"type": "start_session", "resource_path": self.key,
                                                 "offload_video_to_cpu": self.offload,
                                                 "offload_state_to_cpu": self.offload})["session_id"]
            self.predictor.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": 0, "text": prompt})
            self.sessions.append(sid)
            self.generators.append(self.predictor.handle_stream_request({
                "type": "propagate_in_video", "session_id": sid, "start_frame_index": 0,
                "propagation_direction": "forward", "evict_cached_frame_outputs": True}))

    def push(self, rgb) -> int:
        if self.pending is not None:
            raise RuntimeError("previous frame is not collected yet")
        index = self.loader.push(rgb)
        if index == 0:
            self._start()
        self.pending, self.objects = index, [None] * len(self.prompts)
        return index

    def step_prompt(self, k):
        if self.pending is None or self.objects[k] is not None or any(o is None for o in self.objects[:k]):
            raise RuntimeError("prompts must be stepped once each, in order, after push")
        if self.share_backbone:
            _SHARED["key"] = (self.key, self.pending)
        try:
            response = next(self.generators[k])
        finally:
            _SHARED["key"] = None
        if response["frame_index"] != self.pending:
            raise RuntimeError(f"stream out of step: got {response['frame_index']}, expected {self.pending}")
        self.objects[k] = self.encode(response["outputs"], self.prompts[k], f"{self.namespace}/{k}")

    def collect(self):
        if self.pending is None or any(o is None for o in self.objects):
            raise RuntimeError("collect before every prompt was stepped")
        index, objects = self.pending, [o for per_prompt in self.objects for o in per_prompt]
        self.frames, self.pending, self.objects = index + 1, None, None
        if _SHARED["cache_key"] is not None and _SHARED["cache_key"][0] == self.key:
            _SHARED.update(cache_key=None, image=None, out=None)
        return {"frame_index": index, "objects": objects}

    def step(self, rgb):
        t0 = time.perf_counter()
        self.push(rgb)
        for k in range(len(self.prompts)):
            self.step_prompt(k)
        out = self.collect()
        out["seconds"] = time.perf_counter() - t0
        return out

    def close(self):
        for gen in self.generators:
            try:
                gen.close()
            except Exception:
                pass
        for sid in self.sessions:
            try:
                self.predictor.handle_request({"type": "close_session", "session_id": sid})
            except Exception:
                pass
        _LOADERS.pop(self.key, None)
        self.loader.images = []

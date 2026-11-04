"""Online SAM3-v2 memory for one policy session: the streaming counterpart of the offline sparse bank.

Same computation as real-sonic-sam3v2-sparse-20260930/tools/memory_runtime.py (OnlineObjMem), whose memory
matched the offline sparse bank with zero difference in every tensor. Here it runs in slices so a server can
fit it into the idle time between policy inferences:

    begin(rgb, raw_frame)   push the frame into the streaming SAM3 tracker (CPU)
    step_prompt(k)          one SAM3 step for prompt k (GPU, ~100 ms on the RTX 5090)
    finish()                BackgroundCompensation, CausalSelector.update, Qwen vision-merger native features,
                            per-slot mask pooling, and the memory tensors for the policy

process(rgb, raw_frame) runs all slices at once. Memory frames are the frames the server tracks, one every
`stride` raw frames; raw_frame is the frame's raw 30 Hz index since the session start and only has to increase.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F

SCORE_THRESHOLD = 0.35


def quality(obj):
    return float(obj["score"]) * (0.6 + 0.4 * math.sqrt(float(obj["mask_area_fraction"])))


class SlicedObjMem:
    def __init__(self, predictor, producer, instruction, prompts, stride, tokens=77, max_frames=7, offload=True,
                 share_backbone=False):
        from sam3_stream import StreamingTracker
        from selector import CausalSelector
        from run import BackgroundCompensation
        self.prompts, self.slots, self.tokens, self.max_frames = list(prompts), len(prompts), tokens, max_frames
        self.stride = int(stride)
        self.tracker = StreamingTracker(predictor, self.prompts, namespace="online", offload=offload,
                                        share_backbone=share_backbone)
        self.selector = CausalSelector(max_frames=max_frames, event_fraction=0.75,
                                       event_min_gap=max(1, math.ceil(5 / self.stride)), score_threshold=0.35)
        self.compensation = BackgroundCompensation()
        self.producer, self.instruction = producer, instruction
        self.raw, self.native, self.ofeat, self.ogeom, self.ovalid = [], [], [], [], []
        self.selection = None
        self.current = None  # (rgb, raw_frame, next prompt index) of the frame being tracked

    # -- slices ---------------------------------------------------------------------------------------------
    def begin(self, rgb: np.ndarray, raw_frame: int):
        if self.current is not None:
            raise RuntimeError("previous memory frame is not finished")
        if self.raw and raw_frame <= self.raw[-1]:
            raise ValueError(f"raw frames must increase: {self.raw[-1]} -> {raw_frame}")
        self.tracker.push(rgb)
        self.current = [rgb, int(raw_frame), 0]

    def next_slice(self):
        """'begin' (nothing pending), ('prompt', k) or 'finish'."""
        if self.current is None:
            return None
        k = self.current[2]
        return ("prompt", k) if k < self.slots else ("finish", None)

    def step_prompt(self, k):
        if self.current is None or self.current[2] != k:
            raise RuntimeError("prompt slices must run in order")
        self.tracker.step_prompt(k)
        self.current[2] += 1

    def finish(self) -> dict:
        import cv2
        from pycocotools import mask as mask_utils
        if self.current is None or self.current[2] != self.slots:
            raise RuntimeError("finish before every prompt was tracked")
        rgb, raw_frame, _ = self.current
        t0 = time.perf_counter()
        res = self.tracker.collect()
        i = res["frame_index"]
        record = {"frame_index": i, "available_frame": i, "raw_frame": int(raw_frame),
                  "capture_time_seconds": raw_frame / 30.0, "objects": res["objects"]}
        record.update(self.compensation.update(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), record["objects"]))
        row = self.selector.update(record)
        t1 = time.perf_counter()
        native, _ = self.producer.encode(rgb, self.instruction)
        native = native[0].detach().to("cpu", torch.bfloat16)  # [77, 2048]
        t2 = time.perf_counter()
        weights = np.zeros((self.slots, self.tokens), dtype=np.float32)
        geometry = np.zeros((self.slots, 8), dtype=np.float32)
        valid = np.zeros(self.slots, dtype=bool)
        for k, prompt in enumerate(self.prompts):
            cands = [o for o in record["objects"] if o["prompt"] == prompt
                     and float(o["score"]) >= SCORE_THRESHOLD and float(o["mask_area_fraction"]) > 0]
            if not cands:
                continue
            obj = max(cands, key=quality)
            rle = dict(obj["mask_rle"]); rle["counts"] = rle["counts"].encode("ascii")
            binary = mask_utils.decode(rle).astype(np.float32)
            cell = F.adaptive_avg_pool2d(torch.from_numpy(binary)[None, None], (7, 11)).reshape(77).numpy()
            if cell.sum() <= 0 or obj.get("centroid_xy") is None:
                continue
            weights[k] = cell
            x0, y0, x1, y1 = (float(v) for v in obj["bbox_xyxy"])
            cx, cy = (float(v) for v in obj["centroid_xy"])
            geometry[k] = [x0, y0, x1, y1, cx, cy, float(obj["mask_area_fraction"]), 1.0]
            valid[k] = True
        norm = torch.from_numpy(weights / np.maximum(weights.sum(-1, keepdims=True), 1e-12))
        pooled = torch.einsum("fsc,fcd->fsd", norm[None], native.float()[None]).to(torch.bfloat16)[0]
        pooled[torch.from_numpy(~valid)] = 0
        self.raw.append(int(raw_frame)); self.native.append(native); self.ofeat.append(pooled)
        self.ogeom.append(torch.from_numpy(geometry)); self.ovalid.append(torch.from_numpy(valid))
        sel = list(row["selected_raw_frame_indices"])[:row["selection_count"]]
        if len(sel) != min(self.max_frames, i + 1) or sel[-1] != i:
            raise RuntimeError(f"selector returned an unexpected selection at {i}: {sel}")
        self.selection = sel
        self.current = None
        t3 = time.perf_counter()
        return {"frame_index": i, "raw_frame": int(raw_frame), "selected_raw_frames": [self.raw[j] for j in sel],
                "valid_slots": valid.tolist(), "selector_ms": (t1 - t0) * 1e3, "native_ms": (t2 - t1) * 1e3,
                "pool_ms": (t3 - t2) * 1e3}

    def process(self, rgb: np.ndarray, raw_frame: int) -> dict:
        self.begin(rgb, raw_frame)
        for k in range(self.slots):
            self.step_prompt(k)
        return self.finish()

    # -- policy input ---------------------------------------------------------------------------------------
    def memory(self) -> dict:
        if self.selection is None:
            raise RuntimeError("no frame processed yet")
        tokens, slots = self.tokens, self.slots
        budget, obudget = self.max_frames * tokens, self.max_frames * slots
        out = {"memory_features": torch.zeros((budget, 2048), dtype=torch.bfloat16),
               "memory_mask": torch.zeros(budget, dtype=torch.bool),
               "memory_raw_frame_indices": torch.full((budget,), -1, dtype=torch.int64),
               "memory_spatial_indices": torch.full((budget,), -1, dtype=torch.int64),
               "memory_object_features": torch.zeros((obudget, 2048), dtype=torch.bfloat16),
               "memory_object_geometry": torch.zeros((obudget, 8), dtype=torch.float32),
               "memory_object_mask": torch.zeros(obudget, dtype=torch.bool),
               "memory_object_raw_frame_indices": torch.full((obudget,), -1, dtype=torch.int64),
               "memory_object_slots": torch.full((obudget,), -1, dtype=torch.int64)}
        for n, j in enumerate(self.selection):
            a, b = n * tokens, n * slots
            out["memory_features"][a:a + tokens] = self.native[j]
            out["memory_mask"][a:a + tokens] = True
            out["memory_raw_frame_indices"][a:a + tokens] = self.raw[j]
            out["memory_spatial_indices"][a:a + tokens] = torch.arange(tokens)
            out["memory_object_features"][b:b + slots] = self.ofeat[j]
            out["memory_object_geometry"][b:b + slots] = self.ogeom[j]
            out["memory_object_mask"][b:b + slots] = self.ovalid[j]
            out["memory_object_raw_frame_indices"][b:b + slots] = self.raw[j]
            out["memory_object_slots"][b:b + slots] = torch.arange(slots)
        om = out["memory_object_mask"]
        out["memory_object_features"][~om] = 0
        out["memory_object_geometry"][~om] = 0
        out["memory_object_raw_frame_indices"][~om] = -1
        out["memory_object_slots"][~om] = -1
        return out

    def close(self):
        self.tracker.close()

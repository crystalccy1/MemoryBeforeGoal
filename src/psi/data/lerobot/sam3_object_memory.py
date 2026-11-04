"""SAM3-tracked object memory on top of the native 7x77 keyframe bank.

A sidecar per task stores, for every raw execution frame:
  * a causal SAM3 event selection of 7 keyframes (current frame included), and
  * up to ``OBJECT_SLOTS`` object tokens: the SAM3 mask pooled over the frame's
    native 7x11 Qwen vision-merger grid, plus box/centroid/area/visibility.

Object slots are assigned causally, in order of first reliable appearance of a
SAM3 identity (prompt, object_id). Mask pooling uses the geometry of the Qwen
preprocessing (whole-image rescale, no crop/pad): merged cell (r, c) covers
source rows [r*H/7, (r+1)*H/7) and columns [c*W/11, (c+1)*W/11).
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from psi.data.lerobot.framesamp_features import assemble_memory_tokens
from psi.data.lerobot.keyframe_native import (
    FEATURE_DIM, GRID_HW, MAX_FRAMES, TOKENS_PER_FRAME, validate_selection,
)
from psi.data.lerobot.keyframe_native_training import NativeKeyframeTrainingBank, PaddedCollatorForNativeKeyframe

SCHEMA = "psi0-sam3-object-memory/v1"
OBJECT_SLOTS = 2
GEOMETRY_FIELDS = ("x0", "y0", "x1", "y1", "cx", "cy", "area_fraction", "visible")
GEOMETRY_DIM = len(GEOMETRY_FIELDS)
SCORE_THRESHOLD = 0.35
OBJECT_BUDGET = MAX_FRAMES * OBJECT_SLOTS
OBJECT_FIELDS = (
    "memory_object_features", "memory_object_geometry", "memory_object_mask",
    "memory_object_raw_frame_indices", "memory_object_slots",
)
EPISODE_TENSORS = ("raw_frame_indices", "sam3_raw_frame_indices", "sam3_frame_mask", "object_features",
                   "object_geometry", "object_valid", "object_cell_weights")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_coco_rle(rle: dict) -> np.ndarray:
    """Decode a compressed COCO RLE (as written by pycocotools) to a bool [H, W] mask."""
    height, width = (int(v) for v in rle["size"])
    counts = rle["counts"]
    if isinstance(counts, str):
        values, position = [], 0
        while position < len(counts):
            value, shift, more = 0, 0, True
            while more:
                char = ord(counts[position]) - 48
                value |= (char & 0x1F) << (5 * shift)
                more = bool(char & 0x20)
                position += 1
                shift += 1
                if not more and char & 0x10:
                    value |= -1 << (5 * shift)
            if len(values) > 2:
                value += values[-2]
            values.append(value)
        counts = values
    counts = np.asarray(counts, dtype=np.int64)
    if np.any(counts < 0) or counts.sum() != height * width:
        raise ValueError("Corrupt RLE: run lengths do not cover the image")
    flat = np.repeat(np.arange(len(counts)) % 2 == 1, counts)
    return flat.reshape(width, height).T  # COCO RLE is column-major


def cell_weights(masks: np.ndarray) -> np.ndarray:
    """Area fraction of each 7x11 merged cell covered by each mask: [N, H, W] -> [N, 77]."""
    tensor = torch.from_numpy(np.ascontiguousarray(masks, dtype=np.float32))[:, None]
    pooled = F.adaptive_avg_pool2d(tensor, GRID_HW)
    return pooled.reshape(len(masks), TOKENS_PER_FRAME).numpy()


def eligible(obj: dict) -> bool:
    return float(obj.get("score", 0.0)) >= SCORE_THRESHOLD and float(obj.get("mask_area_fraction", 0.0)) > 0


def assign_slots(records: list[dict], slots: int = OBJECT_SLOTS) -> tuple[list[dict[int, dict]], dict]:
    """Causally map SAM3 identities to slots by first eligible appearance."""
    order: dict[tuple[str, str], int] = {}
    dropped: set[tuple[str, str]] = set()
    per_frame = []
    for record in records:
        present = {}
        for obj in record["objects"]:
            if not eligible(obj):
                continue
            key = (obj["prompt"], obj["object_id"])
            if key not in order:
                if len(order) < slots:
                    order[key] = len(order)
                else:
                    dropped.add(key)
                    continue
            slot = order[key]
            if slot in present:
                raise ValueError("Duplicate SAM3 identity within one frame")
            present[slot] = obj
        per_frame.append(present)
    audit = {"assigned_identities": [list(k) for k in order], "dropped_identities": sorted(map(list, dropped))}
    return per_frame, audit


def build_object_tensors(records: list[dict], features: torch.Tensor, slots: int = OBJECT_SLOTS,
                         chunk: int = 128) -> tuple[dict[str, torch.Tensor], dict]:
    """Pool SAM3 masks over native 7x11 features. ``features``: [F, 77, 2048] bf16."""
    frames = len(records)
    if features.shape != (frames, TOKENS_PER_FRAME, FEATURE_DIM):
        raise ValueError("Features and tracks disagree on frame count or grid")
    per_frame, audit = assign_slots(records, slots)
    weights = np.zeros((frames, slots, TOKENS_PER_FRAME), dtype=np.float32)
    geometry = np.zeros((frames, slots, GEOMETRY_DIM), dtype=np.float32)
    valid = np.zeros((frames, slots), dtype=bool)
    for t, present in enumerate(per_frame):
        if not present:
            continue
        masks = np.stack([decode_coco_rle(obj["mask_rle"]) for obj in present.values()])
        cells = cell_weights(masks)
        for (slot, obj), cell in zip(present.items(), cells):
            if cell.sum() <= 0:
                continue
            weights[t, slot] = cell
            x0, y0, x1, y1 = (float(v) for v in obj["bbox_xyxy"])
            cx, cy = (float(v) for v in obj["centroid_xy"])
            geometry[t, slot] = [x0, y0, x1, y1, cx, cy, float(obj["mask_area_fraction"]), 1.0]
            valid[t, slot] = True
    pooled = torch.zeros((frames, slots, FEATURE_DIM), dtype=torch.bfloat16)
    normalized = weights / np.maximum(weights.sum(-1, keepdims=True), 1e-12)
    for start in range(0, frames, chunk):
        stop = min(frames, start + chunk)
        block = torch.einsum("fsc,fcd->fsd", torch.from_numpy(normalized[start:stop]),
                             features[start:stop].float())
        pooled[start:stop] = block.to(torch.bfloat16)
    pooled[torch.from_numpy(~valid)] = 0
    audit.update(valid_fraction_per_slot=valid.mean(0).round(4).tolist(), frames=frames)
    return {"object_features": pooled, "object_geometry": torch.from_numpy(geometry),
            "object_valid": torch.from_numpy(valid),
            "object_cell_weights": torch.from_numpy(weights).to(torch.float16)}, audit


def load_selector(source_dir: str | Path):
    import importlib.util
    path = Path(source_dir) / "selector.py"
    spec = importlib.util.spec_from_file_location("sam3_selector", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(module)
    return module.select_records


def sam3_selection(records: list[dict], select_records, selector_config: dict) -> tuple[torch.Tensor, torch.Tensor, dict]:
    result = select_records(records, max_frames=MAX_FRAMES, **selector_config)
    ids = torch.from_numpy(result["selected_raw_frame_indices"]).long()
    mask = torch.from_numpy(result["selected_frame_mask"]).bool()
    validate_selection(ids, mask, len(records))
    for length in sorted({1, min(16, len(records)), max(1, len(records) // 2)}):
        prefix = select_records(records[:length], max_frames=MAX_FRAMES, **selector_config)
        if not np.array_equal(prefix["selected_raw_frame_indices"], result["selected_raw_frame_indices"][:length]):
            raise ValueError("SAM3 selector prefix replay mismatch: selection is not causal")
    return ids, mask, result["summary"]


class Sam3ObjectSidecar:
    """Verified reader for a per-task SAM3 object-memory sidecar."""

    def __init__(self, root: str | Path, expected_manifest_sha256: str, expected_task: str,
                 expected_native_manifest_sha256: str, cache_episodes: int = 8) -> None:
        self.root = Path(root).resolve(strict=True)
        if sha256_file(self.root / "manifest.json") != expected_manifest_sha256:
            raise ValueError("SAM3 object sidecar manifest checksum mismatch")
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        ready = json.loads((self.root / "READY").read_text())
        m = self.manifest
        if (ready.get("manifest_sha256") != expected_manifest_sha256 or m.get("status") != "complete"
                or m.get("schema_version") != SCHEMA or m.get("task") != expected_task
                or m.get("native_manifest_sha256") != expected_native_manifest_sha256
                or m.get("object_slots") != OBJECT_SLOTS or m.get("geometry_fields") != list(GEOMETRY_FIELDS)):
            raise ValueError("SAM3 object sidecar identity/contract mismatch")
        self.records = {int(r["episode_index"]): r for r in m["outputs"]}
        self._verified: dict[int, tuple[int, int, int]] = {}
        for episode, row in self.records.items():
            path = self._path(row)
            if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
                raise ValueError(f"SAM3 object shard changed: {path}")
            stat = path.stat()
            self._verified[episode] = (stat.st_size, stat.st_mtime_ns, stat.st_ino)
        self._cache: OrderedDict[int, dict[str, torch.Tensor]] = OrderedDict()
        self._cache_size = cache_episodes

    def _path(self, row: dict) -> Path:
        path = (self.root / row["relative_path"]).resolve(strict=True)
        path.relative_to(self.root)
        return path

    def episode(self, episode: int) -> dict[str, torch.Tensor]:
        if episode in self._cache:
            self._cache.move_to_end(episode)
            return self._cache[episode]
        row = self.records[episode]
        path = self._path(row)
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns, stat.st_ino) != self._verified[episode]:
            raise ValueError("SAM3 object shard changed after verification")
        with safe_open(path, framework="pt", device="cpu") as handle:
            data = {key: handle.get_tensor(key) for key in handle.keys()}
        frames = int(row["raw_frames"])
        if (set(data) != set(EPISODE_TENSORS) or data["object_features"].shape != (frames, OBJECT_SLOTS, FEATURE_DIM)
                or data["object_features"].dtype != torch.bfloat16
                or not torch.equal(data["raw_frame_indices"], torch.arange(frames))):
            raise ValueError("SAM3 object shard violates its contract")
        validate_selection(data["sam3_raw_frame_indices"], data["sam3_frame_mask"], frames)
        self._cache[episode] = data
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return data


def gather_object_memory(data: dict[str, torch.Tensor], selected: torch.Tensor) -> dict[str, torch.Tensor]:
    """Frame-major [7*S] object tokens for selected raw frames (valid prefix only)."""
    count = len(selected)
    features = torch.zeros((OBJECT_BUDGET, FEATURE_DIM), dtype=torch.bfloat16)
    geometry = torch.zeros((OBJECT_BUDGET, GEOMETRY_DIM), dtype=torch.float32)
    mask = torch.zeros(OBJECT_BUDGET, dtype=torch.bool)
    raw = torch.full((OBJECT_BUDGET,), -1, dtype=torch.int64)
    slots = torch.full((OBJECT_BUDGET,), -1, dtype=torch.int64)
    n = count * OBJECT_SLOTS
    valid = data["object_valid"][selected].reshape(n)
    features[:n] = data["object_features"][selected].reshape(n, FEATURE_DIM)
    geometry[:n] = data["object_geometry"][selected].reshape(n, GEOMETRY_DIM)
    mask[:n] = valid
    raw[:n] = selected.repeat_interleave(OBJECT_SLOTS)
    slots[:n] = torch.arange(OBJECT_SLOTS).repeat(count)
    features[~mask] = 0
    geometry[~mask] = 0
    raw[~mask] = -1
    slots[~mask] = -1
    return {"memory_object_features": features, "memory_object_geometry": geometry,
            "memory_object_mask": mask, "memory_object_raw_frame_indices": raw, "memory_object_slots": slots}


class Sam3ObjectNativeTrainingBank(NativeKeyframeTrainingBank):
    """Native 7x77 features selected by SAM3 events, plus SAM3 object tokens."""

    def __init__(self, root, *, expected_manifest_sha256, expected_task, expected_dataset_revision,
                 object_root, object_manifest_sha256):
        super().__init__(root, expected_manifest_sha256=expected_manifest_sha256, expected_task=expected_task,
                         expected_dataset_revision=expected_dataset_revision, selector="uniform")
        self.selector = "sam3"
        self.objects = Sam3ObjectSidecar(object_root, object_manifest_sha256, expected_task,
                                         expected_manifest_sha256)
        if {e: int(r["raw_frames"]) for e, r in self.objects.records.items()} != {
                e: int(r["raw_frames"]) for e, r in self.records.items()}:
            raise ValueError("SAM3 sidecar and native bank cover different episodes/frames")

    def memory_for_frame(self, episode_index, current_frame_index):
        if type(episode_index) is not int or type(current_frame_index) is not int:
            raise TypeError("Memory episode/frame identifiers must be integers")
        features, _ = self._episode(episode_index)
        data = self.objects.episode(episode_index)
        ids = data["sam3_raw_frame_indices"][current_frame_index]
        raw = ids[data["sam3_frame_mask"][current_frame_index]]
        result = assemble_memory_tokens(features["features"][raw], raw, max_memory_frames=MAX_FRAMES,
                                        tokens_per_frame=TOKENS_PER_FRAME, feature_dim=FEATURE_DIM)
        memory = {name: result[name] for name in ("memory_features", "memory_mask",
                                                   "memory_raw_frame_indices", "memory_spatial_indices")}
        memory.update(gather_object_memory(data, raw))
        memory["selected_raw_frame_indices"] = raw.clone()
        return memory


DECISION_LABELS_SCHEMA = "psi0-decision-labels/v1"
DECISION_IGNORE = -100


class Sam3ObjectDecisionTrainingBank(Sam3ObjectNativeTrainingBank):
    """As Sam3ObjectNativeTrainingBank, plus ``decision_label`` per frame: answer - 1 (0-based class) from the
    episode's first supervised frame on (the recorded decision step), DECISION_IGNORE before it."""

    def __init__(self, root, *, decision_labels, decision_labels_sha256, **kwargs):
        super().__init__(root, **kwargs)
        if sha256_file(decision_labels) != decision_labels_sha256:
            raise ValueError("Decision labels changed")
        labels = json.loads(Path(decision_labels).read_text())
        if labels.get("schema") != DECISION_LABELS_SCHEMA or labels.get("task") != kwargs["expected_task"]:
            raise ValueError("Decision labels have the wrong schema or task")
        self.decision_num_classes = int(labels["num_classes"])
        self.decision = {int(e): (int(v["answer"]), int(v["start"])) for e, v in labels["episodes"].items()}
        if set(self.decision) != set(self.records):
            raise ValueError("Decision labels and native bank cover different episodes")
        if any(not 1 <= a <= self.decision_num_classes or s < 0 for a, s in self.decision.values()):
            raise ValueError("Decision label out of range")

    def memory_for_frame(self, episode_index, current_frame_index):
        memory = super().memory_for_frame(episode_index, current_frame_index)
        answer, start = self.decision[episode_index]
        memory["decision_label"] = torch.tensor(answer - 1 if current_frame_index >= start else DECISION_IGNORE,
                                                dtype=torch.int64)
        return memory


class PaddedCollatorForNativeKeyframeObject(PaddedCollatorForNativeKeyframe):
    def __call__(self, instances):
        for sample in instances:
            missing = [name for name in OBJECT_FIELDS if name not in sample]
            if missing:
                raise KeyError(f"Object memory sample is missing {missing}")
            f, g, m, r, s = (sample[name] for name in OBJECT_FIELDS)
            if (f.shape != (OBJECT_BUDGET, FEATURE_DIM) or f.dtype != torch.bfloat16
                    or g.shape != (OBJECT_BUDGET, GEOMETRY_DIM) or m.shape != (OBJECT_BUDGET,) or m.dtype != torch.bool
                    or r.dtype != torch.int64 or s.dtype != torch.int64):
                raise ValueError("Object memory sample violates its shape/dtype contract")
            if torch.count_nonzero(f[~m]) or torch.any(r[~m] != -1) or torch.any(s[~m] != -1):
                raise ValueError("Object memory padding must be zero / -1")
            current = int(sample["framesamp_frame_index"])
            if torch.any(r[m] > current) or torch.any(r[m] < 0):
                raise ValueError("Object memory contains a future frame")
        result = super().__call__(instances)
        for name in OBJECT_FIELDS:
            result[name] = torch.stack([sample[name] for sample in instances])
        labelled = ["decision_label" in sample for sample in instances]
        if any(labelled):
            if not all(labelled):
                raise ValueError("Decision labels must be present for every sample or none")
            result["decision_label"] = torch.stack([sample["decision_label"] for sample in instances])
        return result


def iter_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)

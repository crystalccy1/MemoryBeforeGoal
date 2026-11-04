"""Online, causal SAM3 keyframe + object memory for the policy server.

For every executed frame t, strictly in order, this module builds what the training data loader built offline:

    F[t]    = QwenVisionMergerNativeFrameProducer.encode(rgb_t)    [77, 2048] bf16 native Qwen vision-merger tokens
    record  = TrackerClient.track(...)[t]                          SAM3 objects + background-compensation residuals
    row     = selector.update(record)                              adaptive causal keyframe selection (1..7 frames, ends at t)
    present = slots.update(record), or selector.slot_objects(record) for the tasks whose selector assigns the slots
    pooled  = pool_frame(present, F[t])                            mask-weighted pooling of F[t] per object slot

At a request it assembles the memory at the current frame from the selected frames (``assemble_memory_tokens``) and their
object tokens (``gather_object_memory``), applies the training collator checks, and returns the nine memory tensors with a
batch dimension of 1. The pooling, slot assignment and assembly are copies of the training data code in
``psi.data.lerobot.sam3_object_memory`` (marked below), so the online memory equals the offline one for the same frames.

Invariants (a violation raises; once a frame was accepted it also poisons the store until the server restarts, because
SAM3, the background compensation and the selector change their state in place and cannot roll back):
    * frames are uint8 RGB (360, 640, 3); raw indices are contiguous from 0; the last uploaded frame is the current image;
    * every frame goes through tracker -> selector -> slots -> pooling exactly once; one tracker session at a time;
    * each selection has 1..min(7, t+1) sorted, unique frames and ends at t; object tokens never reference a later frame.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import json
import time
from typing import Any

import numpy as np
import torch

from psi.data.lerobot.framesamp_features import assemble_memory_tokens
from psi.data.lerobot.sam3_object_memory import (
    FEATURE_DIM, GEOMETRY_DIM, GRID_HW, MAX_FRAMES, OBJECT_BUDGET, OBJECT_FIELDS, OBJECT_SLOTS, TOKENS_PER_FRAME,
    cell_weights, decode_coco_rle, eligible, gather_object_memory,
)
from psi.deploy.framesamp_online import FrameSampOnlineSessionStore

from wholebodymem.selectors import load_selector_class
from wholebodymem.tasks import TaskSpec
from wholebodymem.tracker_client import TrackerClient

FRAME_SHAPE = (360, 640, 3)
FPS = 50
MIN_N_DECL = 16
PROMPT_SEP = "|"
AVAILABILITY_MODE = "synchronous_replay; no asynchronous latency simulation"
RECORD_KEYS = frozenset({   # keys of a training track record
    "frame_index", "available_frame", "capture_time_seconds", "availability_mode", "objects",
    "tracking_elapsed_wall_seconds", "background_alignment_valid", "background_inlier_fraction",
})
OBJECT_KEYS = frozenset({   # sam_adapter._encode_objects
    "object_id", "prompt", "score", "score_kind", "relevance", "bbox_xyxy", "mask_area_fraction",
    "centroid_xy", "mask_rle",
})
OBJECT_OPTIONAL_KEYS = frozenset({"motion_residual", "mask_change_residual"})   # background compensation
VISUAL_FIELDS = ("memory_features", "memory_mask", "memory_raw_frame_indices", "memory_spatial_indices")
MEMORY_FIELDS = VISUAL_FIELDS + OBJECT_FIELDS
MEMORY_BUDGET = MAX_FRAMES * TOKENS_PER_FRAME
MEMORY_VIEW = "rgb_head_stereo_left"


# ----------------------------------------------------------------------------------------------
# Per-frame replays of the training data code
# ----------------------------------------------------------------------------------------------

def normalize_record(record: dict, *, t: int, prompt: str, namespace: str) -> dict:
    """Serialise the record as the training tracks were written (JSON) and read it back, then check its schema."""
    record = json.loads(json.dumps(record, sort_keys=True, allow_nan=False))
    if not isinstance(record, dict) or set(record) != RECORD_KEYS:
        raise RuntimeError(f"SAM3 record {t} keys differ from training: {sorted(record) if isinstance(record, dict) else record!r}")
    if (type(record["frame_index"]) is not int or record["frame_index"] != t or record["available_frame"] != t
            or record["capture_time_seconds"] != t / FPS or record["availability_mode"] != AVAILABILITY_MODE):
        raise RuntimeError(f"SAM3 record {t} is not the synchronous record for frame {t}")
    if not isinstance(record["objects"], list):
        raise RuntimeError(f"SAM3 record {t} objects must be a list")
    for obj in record["objects"]:
        keys = set(obj) if isinstance(obj, dict) else set()
        if not OBJECT_KEYS <= keys <= OBJECT_KEYS | OBJECT_OPTIONAL_KEYS:
            raise RuntimeError(f"SAM3 record {t} object keys differ from training: {sorted(keys)}")
        # one SAM3 session per prompt: prompt i lives in ".../<i>:<id>"
        prompts = prompt.split(PROMPT_SEP)
        base = namespace.rsplit("/", 1)[0]
        ok = any(obj["prompt"] == p and str(obj["object_id"]).startswith(f"{base}/{i}:") for i, p in enumerate(prompts))
        if not ok or (len(prompts) == 1 and not str(obj["object_id"]).startswith(namespace + ":")):
            raise RuntimeError(f"SAM3 record {t} object {obj['object_id']!r}/{obj['prompt']!r} is not {namespace}/{prompt!r}")
        rle = obj["mask_rle"]
        if not isinstance(rle, dict) or rle.get("size") != list(FRAME_SHAPE[:2]) or not isinstance(rle.get("counts"), str):
            raise RuntimeError(f"SAM3 record {t} mask_rle is not a {FRAME_SHAPE[:2]} compressed COCO RLE")
    return record


class IncrementalSlots:
    """Causal slot assignment by first appearance, keeping assign_slots' ``order`` and ``dropped`` across frames."""

    def __init__(self, slots: int = OBJECT_SLOTS) -> None:
        self.slots = int(slots)
        self.order: dict[tuple[str, str], int] = {}
        self.dropped: set[tuple[str, str]] = set()

    def update(self, record: dict) -> dict[int, dict]:
        slots, order, dropped = self.slots, self.order, self.dropped
        # --- copy of the assign_slots body in psi/data/lerobot/sam3_object_memory.py ---
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
        # --- end copy ---
        return present

    def audit(self) -> dict:
        order, dropped = self.order, self.dropped
        return {"assigned_identities": [list(k) for k in order], "dropped_identities": sorted(map(list, dropped))}


def pool_frame(present: dict[int, dict], features_t: torch.Tensor, slots: int = OBJECT_SLOTS,
               chunk: int = 128, source_features=None) -> dict[str, torch.Tensor]:
    """build_object_tensors pooling for a single frame t. ``features_t``: [77, 2048] bf16."""
    if tuple(features_t.shape) != (TOKENS_PER_FRAME, FEATURE_DIM) or features_t.dtype != torch.bfloat16:
        raise ValueError("pool_frame needs native [77,2048] bf16 tokens")
    frames, per_frame, features = 1, [present], features_t[None]
    # --- copy of build_object_tensors in psi/data/lerobot/sam3_object_memory.py, with frames = 1 ---
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
    # --- end copy ---
    if source_features is not None:   # hidden-object slots: pool a persisted object over its source frame, visible = 0
        for slot, obj in present.items():
            s = obj.get("_source_frame")
            if s is None or not geometry[0, slot, 7]:
                continue
            w = torch.from_numpy(np.ascontiguousarray(normalized[0, slot]))
            pooled[0, slot] = torch.einsum("c,cd->d", w, source_features(int(s)).float()).to(torch.bfloat16)
            geometry[0, slot, 7] = 0.0
    return {"object_features": pooled[0], "object_geometry": torch.from_numpy(geometry)[0],
            "object_valid": torch.from_numpy(valid)[0],
            "object_cell_weights": torch.from_numpy(weights).to(torch.float16)[0]}


def check_selection(selected: list[int], t: int) -> None:
    """Per-row form of keyframe_native.validate_selection: 1..min(7, t+1) sorted frames ending at t."""
    if (not 1 <= len(selected) <= min(MAX_FRAMES, t + 1) or selected[0] < 0 or selected[-1] != t
            or any(b <= a for a, b in zip(selected, selected[1:]))):
        raise RuntimeError(f"SAM3 selection at frame {t} violates the causal keyframe contract: {selected}")


def check_memory_contract(memory: dict[str, torch.Tensor], current: int) -> None:
    """The invariants the training collators applied to every sample."""
    if set(memory) != set(MEMORY_FIELDS):
        raise KeyError(f"memory fields differ: {sorted(memory)}")
    # --- the native keyframe collator (psi/data/lerobot/keyframe_native_training.py) ---
    features, mask, raw, spatial = (memory[name] for name in VISUAL_FIELDS)
    if features.dtype != torch.bfloat16 or mask.dtype != torch.bool:
        raise TypeError("Native memory requires bfloat16 features and bool masks")
    if any(memory[name].dtype != torch.int64 for name in VISUAL_FIELDS[2:]):
        raise TypeError("Native memory identifiers must be int64")
    cap = min(MAX_FRAMES, current + 1) * TOKENS_PER_FRAME
    if (features.shape != (MEMORY_BUDGET, FEATURE_DIM)
            or any(item.shape != (MEMORY_BUDGET,) for item in (mask, raw, spatial))):
        raise ValueError("Native memory shape differs from its seven-frame budget")
    count = int(mask.sum())
    if (count % TOKENS_PER_FRAME or not TOKENS_PER_FRAME <= count <= cap
            or not torch.equal(mask, torch.arange(MEMORY_BUDGET) < count)):
        raise ValueError("Native memory padding must hold 1..min(7,t+1) whole right-padded frames")
    if (not torch.isfinite(features).all() or torch.count_nonzero(features[~mask])
            or not torch.all(raw[~mask] == -1) or not torch.all(spatial[~mask] == -1)):
        raise ValueError("Native memory values or padding are invalid")
    grouped_raw = raw[:count].reshape(-1, TOKENS_PER_FRAME)
    selected = grouped_raw[:, 0]
    if (not torch.equal(grouped_raw, selected[:, None].expand_as(grouped_raw))
            or not torch.equal(spatial[:count], torch.arange(TOKENS_PER_FRAME).repeat(len(selected)))
            or selected[0] < 0 or selected[-1] != current
            or not torch.all(torch.diff(selected) > 0)):
        raise ValueError("Native memory must contain chronological causal frames including current")
    # --- the object collator (psi/data/lerobot/sam3_object_memory.py) ---
    f, g, m, r, s = (memory[name] for name in OBJECT_FIELDS)
    if (f.shape != (OBJECT_BUDGET, FEATURE_DIM) or f.dtype != torch.bfloat16
            or g.shape != (OBJECT_BUDGET, GEOMETRY_DIM) or m.shape != (OBJECT_BUDGET,) or m.dtype != torch.bool
            or r.dtype != torch.int64 or s.dtype != torch.int64):
        raise ValueError("Object memory sample violates its shape/dtype contract")
    if torch.count_nonzero(f[~m]) or torch.any(r[~m] != -1) or torch.any(s[~m] != -1):
        raise ValueError("Object memory padding must be zero / -1")
    if torch.any(r[m] > current) or torch.any(r[m] < 0):
        raise ValueError("Object memory contains a future frame")
    # --- end; added: geometry dtype and slot/frame ranges ---
    if g.dtype != torch.float32 or torch.any(s[m] >= OBJECT_SLOTS) or r.shape != (OBJECT_BUDGET,) or s.shape != (OBJECT_BUDGET,):
        raise ValueError("Object memory geometry/slot identifiers violate the contract")


# ----------------------------------------------------------------------------------------------
# Session store
# ----------------------------------------------------------------------------------------------

@dataclass
class _Episode:
    episode_index: int
    selector: Any
    slots: IncrementalSlots
    token: int
    next_t: int = 0
    requests: int = 0
    features: torch.Tensor | None = None             # [N_decl, 77, 2048] bf16, indexed by raw frame
    object_features: torch.Tensor | None = None      # [N_decl, 2, 2048] bf16
    object_geometry: torch.Tensor | None = None      # [N_decl, 2, 8] f32
    object_valid: torch.Tensor | None = None         # [N_decl, 2] bool
    object_cell_weights: torch.Tensor | None = None  # [N_decl, 2, 77] f16 (diagnostics only)
    selections: list[list[int]] = field(default_factory=list)
    valid_object_tokens: int = 0

    def allocate(self, n_decl: int) -> None:
        self.features = torch.zeros((n_decl, TOKENS_PER_FRAME, FEATURE_DIM), dtype=torch.bfloat16)
        self.object_features = torch.zeros((n_decl, OBJECT_SLOTS, FEATURE_DIM), dtype=torch.bfloat16)
        self.object_geometry = torch.zeros((n_decl, OBJECT_SLOTS, GEOMETRY_DIM), dtype=torch.float32)
        self.object_valid = torch.zeros((n_decl, OBJECT_SLOTS), dtype=torch.bool)
        self.object_cell_weights = torch.zeros((n_decl, OBJECT_SLOTS, TOKENS_PER_FRAME), dtype=torch.float16)


class MemorySessionStore(FrameSampOnlineSessionStore):
    """Episode-local SAM3 keyframe + object memory keyed by client session (see the module docstring)."""

    SCHEMA_VERSION = "wholebodymem-online-memory/v1"

    def __init__(self, producer: Any, tracker: Any, task: TaskSpec, *, max_sessions: int = 2,
                 memory_log: str | None = None) -> None:
        super().__init__(producer, max_memory_frames=MAX_FRAMES, tokens_per_frame=TOKENS_PER_FRAME,
                         feature_dim=FEATURE_DIM, max_sessions=max_sessions)
        if tracker.prompt != task.sam3_prompt or tracker.namespace != task.track_namespace or tracker.n_decl != task.n_decl:
            raise ValueError("SAM3 tracker prompt/namespace/N_decl disagree with the task")
        self.task = task
        self.tracker = tracker
        self.selector_cls = load_selector_class(task.selector)
        self.selector_kwargs = task.selector_kwargs
        self.n_decl = task.n_decl
        self.prompt, self.namespace = task.sam3_prompt, task.track_namespace
        self.selector_slots = task.selector_slots
        self.memory_log = memory_log
        self.completed_episode_resets = 0
        self._sessions: OrderedDict[str, _Episode] = OrderedDict()
        self._next_token = 0
        self._tracker_owner: tuple[str, int] | None = None
        self._poisoned: dict[str, Any] | None = None
        self._latency: dict[str, list[float]] = {}  # name -> [count, total, max, last]

    # -- bookkeeping ------------------------------------------------------------------------
    def _observe(self, name: str, seconds: float) -> None:
        stat = self._latency.setdefault(name, [0, 0.0, 0.0, 0.0])
        stat[0] += 1
        stat[1] += seconds
        stat[2] = max(stat[2], seconds)
        stat[3] = seconds

    @staticmethod
    def _validate_episode_index(episode_index: Any) -> int:
        episode_index = FrameSampOnlineSessionStore._validate_episode_index(episode_index)
        if episode_index < 0:
            raise ValueError("episode_index must be non-negative (it names the episode's SAM3 objects)")
        return episode_index

    def _new_episode(self, episode_index: int) -> _Episode:
        self._next_token += 1
        return _Episode(episode_index, self.selector_cls(**self.selector_kwargs), IncrementalSlots(OBJECT_SLOTS),
                        self._next_token)

    def _publish(self, session_id: str, state: _Episode) -> None:
        self._sessions[session_id] = state
        self._sessions.move_to_end(session_id)
        while len(self._sessions) > self.max_sessions:
            evicted_id, evicted = self._sessions.popitem(last=False)
            if self._tracker_owner == (evicted_id, evicted.token):
                self._tracker_owner = None

    def _poison(self, reason: str) -> None:
        if self._poisoned is None:
            self._poisoned = {"reasons": [], "at_unix": time.time(), "tracker_close_errors": []}
        self._poisoned["reasons"].append(reason)
        self._tracker_owner = None
        try:
            if self.tracker.alive():
                self.tracker.close_session()
        except Exception as exc:  # recorded only: the store is already failed closed
            self._poisoned["tracker_close_errors"].append(f"{type(exc).__name__}: {exc}")

    def _require_usable(self) -> None:
        if self._poisoned is not None:
            raise RuntimeError("memory session store is poisoned (restart the server): "
                               + " | ".join(self._poisoned["reasons"]))
        if not self.tracker.alive():
            raise RuntimeError("SAM3 tracker child is not alive")

    # -- FrameSampOnlineSessionStore protocol ----------------------------------------------------
    def reset(self, session_id: Any, episode_index: Any) -> None:
        self._require_usable()
        session_id = self._validate_session_id(session_id)
        episode_index = self._validate_episode_index(episode_index)
        self._publish(session_id, self._new_episode(episode_index))  # the tracker resets lazily at frame 0

    def snapshot(self, session_id: Any):
        # There is no rollback point: tracker, background compensation and selector mutate in place.
        session_id = self._validate_session_id(session_id)
        state = self._sessions.get(session_id)
        return {"session_id": session_id, "token": None if state is None else state.token,
                "next_t": None if state is None else state.next_t}

    def restore(self, session_id: Any, snapshot) -> None:
        session_id = self._validate_session_id(session_id)
        self._poison(f"request on session {session_id!r} failed after snapshot {snapshot}; "
                     "SAM3/background-compensation/selector state cannot be rolled back")

    def append_and_assemble(self, *, session_id, episode_index, reset, raw_frames,
                            raw_frame_indices, current_rgb, instruction):
        self._require_usable()
        try:
            return self._append_and_assemble(session_id=session_id, episode_index=episode_index, reset=reset,
                                             raw_frames=raw_frames, raw_frame_indices=raw_frame_indices,
                                             current_rgb=current_rgb, instruction=instruction)
        except BaseException as exc:
            self._poison(f"append_and_assemble({session_id!r}) failed: {type(exc).__name__}: {exc}")
            raise

    def _append_and_assemble(self, *, session_id, episode_index, reset, raw_frames,
                             raw_frame_indices, current_rgb, instruction):
        started = time.perf_counter()
        session_id = self._validate_session_id(session_id)
        if not isinstance(reset, bool):
            raise TypeError("reset flag must be boolean")
        episode_index = self._validate_episode_index(episode_index)
        state = self._new_episode(episode_index) if reset else self._sessions.get(session_id)
        if state is None:
            raise RuntimeError("the first request of an episode requires reset=true")
        if state.episode_index != episode_index:
            raise RuntimeError("episode changed without reset")
        frames, indices = np.asarray(raw_frames), np.asarray(raw_frame_indices)
        if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
            raise ValueError("memory requires uint8 [N,H,W,3] raw RGB")
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("raw indices must be a 1-D integer array")
        if len(frames) == 0 or len(frames) != len(indices):
            raise ValueError("one raw index per uploaded frame is required")
        expected = np.arange(state.next_t, state.next_t + len(frames), dtype=np.int64)
        if not np.array_equal(indices, expected):
            raise RuntimeError("raw execution frames are missing, duplicated, or out of order")
        current = np.asarray(current_rgb)
        if current.dtype != np.uint8 or not np.array_equal(current, frames[-1]):
            raise RuntimeError("the upload must end at the current image")
        if tuple(frames.shape[1:]) != FRAME_SHAPE:
            raise ValueError(f"memory requires {FRAME_SHAPE} frames, got {tuple(frames.shape[1:])}")
        if state.next_t + len(frames) > self.n_decl:
            raise RuntimeError(f"episode exceeds N_decl={self.n_decl}")

        self._publish(session_id, state)  # mutable from here on: any failure poisons the store
        if state.features is None:
            state.allocate(self.n_decl)
        native = []
        for frame in frames:
            tick = time.perf_counter()
            tokens, grid = self.producer.encode(frame, instruction)
            if tuple(tokens.shape) != (1, TOKENS_PER_FRAME, FEATURE_DIM) or tokens.dtype != torch.bfloat16:
                raise RuntimeError("frame producer returned incompatible tokens")
            if not torch.isfinite(tokens.float()).all():
                raise RuntimeError("frame producer returned nonfinite tokens")
            if grid.dtype != torch.int64 or grid.tolist() != [list(GRID_HW)]:
                raise RuntimeError("frame producer requires the ordered 7x11 source grid")
            native.append(tokens[0].detach().cpu().contiguous())
            self._observe("qwen_s_per_frame", time.perf_counter() - tick)

        tick = time.perf_counter()
        if state.next_t == 0:
            self.tracker.reset(session_id, episode_index)
            self._tracker_owner = (session_id, state.token)
        elif self._tracker_owner != (session_id, state.token):
            raise RuntimeError("SAM3 tracker session belongs to another buffer; only one session is active")
        records = self.tracker.track(session_id, episode_index, state.next_t, frames)
        if len(records) != len(frames):
            raise RuntimeError("SAM3 tracker returned the wrong number of records")
        self._observe("tracker_rpc_s_per_request", time.perf_counter() - tick)
        for name, values in (self.tracker.last_timing or {}).items():
            for value in values:
                self._observe(f"{name}_per_frame", value)

        namespace = f"{self.namespace}/{episode_index:06d}/0"
        row = None
        for record, tokens in zip(records, native):
            tick = time.perf_counter()
            t = state.next_t
            record = normalize_record(record, t=t, prompt=self.prompt, namespace=namespace)
            row = state.selector.update(record)
            selected = [int(i) for i in row["selected_raw_frame_indices"]]
            check_selection(selected, t)
            present = state.selector.slot_objects(record) if self.selector_slots else state.slots.update(record)
            pooled = pool_frame(present, tokens, source_features=lambda s: state.features[s])
            state.features[t] = tokens
            for name in ("object_features", "object_geometry", "object_valid", "object_cell_weights"):
                getattr(state, name)[t] = pooled[name]
            state.selections.append(selected)
            state.next_t = t + 1
            self._observe("select_pool_s_per_frame", time.perf_counter() - tick)
        if self.memory_log:   # optional per-request log of the memory frames
            with open(self.memory_log, "a") as stream:
                stream.write(json.dumps({"session": str(session_id), "episode_index": int(episode_index),
                                         "t": int(state.next_t - 1),
                                         "selected": [int(i) for i in row["selected_raw_frame_indices"]],
                                         "objects_last": len(records[-1].get("objects", [])) if records else 0}) + "\n")

        t = state.next_t - 1
        raw = torch.tensor(row["selected_raw_frame_indices"], dtype=torch.int64)
        if (raw.ndim != 1 or not 1 <= raw.shape[0] <= min(MAX_FRAMES, t + 1) or int(raw[-1]) != t or int(raw[0]) < 0
                or not torch.all(torch.diff(raw) > 0)):
            raise RuntimeError(f"SAM3 selection at frame {t} violates the causal contract")
        # --- the memory_for_frame assembly of psi/data/lerobot/sam3_object_memory.py, on this episode's tables ---
        result = assemble_memory_tokens(state.features[raw], raw, max_memory_frames=MAX_FRAMES,
                                        tokens_per_frame=TOKENS_PER_FRAME, feature_dim=FEATURE_DIM)
        memory = {name: result[name] for name in ("memory_features", "memory_mask",
                                                   "memory_raw_frame_indices", "memory_spatial_indices")}
        memory.update(gather_object_memory({"object_valid": state.object_valid,
                                            "object_features": state.object_features,
                                            "object_geometry": state.object_geometry}, raw))
        # --- end ---
        check_memory_contract(memory, current=t)
        state.requests += 1
        state.valid_object_tokens = int(memory["memory_object_mask"].sum())
        self.total_encoded_frames += len(native)
        if state.requests == 1:
            self.completed_episode_resets += 1
        self._observe("append_and_assemble_s_per_request", time.perf_counter() - started)
        return {name: memory[name].unsqueeze(0) for name in MEMORY_FIELDS}

    # -- diagnostics ------------------------------------------------------------------------------
    def episode_tensors(self, session_id: Any) -> dict[str, torch.Tensor]:
        """This buffer's per-frame tables in the training object-memory layout."""
        state = self._sessions[self._validate_session_id(session_id)]
        frames = state.next_t
        ids = torch.full((frames, MAX_FRAMES), -1, dtype=torch.int64)
        mask = torch.zeros((frames, MAX_FRAMES), dtype=torch.bool)
        for t, selected in enumerate(state.selections):
            ids[t, :len(selected)] = torch.tensor(selected, dtype=torch.int64)
            mask[t, :len(selected)] = True
        empty = state.features is None
        return {
            "raw_frame_indices": torch.arange(frames),
            "sam3_raw_frame_indices": ids, "sam3_frame_mask": mask,
            **{name: (torch.empty(0) if empty else getattr(state, name)[:frames].clone())
               for name in ("object_features", "object_geometry", "object_valid", "object_cell_weights")},
        }

    def health(self) -> dict[str, Any]:
        tracker = self.tracker
        return {
            "schema_version": self.SCHEMA_VERSION, "task": self.task.name, "sessions": len(self._sessions),
            "max_sessions": self.max_sessions, "total_encoded_frames": self.total_encoded_frames,
            "memory_frames": MAX_FRAMES, "tokens_per_frame": TOKENS_PER_FRAME, "memory_budget": MEMORY_BUDGET,
            "object_slots": OBJECT_SLOTS, "object_budget": OBJECT_BUDGET, "source_grid_hw": list(GRID_HW),
            "selector": self.task.selector, "selector_kwargs": dict(self.selector_kwargs),
            "prompt": self.prompt, "namespace": self.namespace, "n_decl": self.n_decl,
            "nominal_execution_hz": FPS, "completed_episode_resets": self.completed_episode_resets,
            "tracker_alive": bool(tracker.alive()), "tracker_pid": getattr(tracker, "pid", None),
            "tracker_identity": getattr(tracker, "identity", None),
            "poisoned": None if self._poisoned is None else json.loads(json.dumps(self._poisoned)),
            "latency_seconds": {name: {"count": int(c), "mean": s / c if c else None, "max": m, "last": last}
                                for name, (c, s, m, last) in list(self._latency.items())},
            "buffers": {sid: {"episode_index": v.episode_index, "raw_frames": v.next_t, "requests": v.requests,
                              "selected_raw_frame_indices": v.selections[-1] if v.selections else [],
                              "valid_object_tokens": v.valid_object_tokens}
                        for sid, v in list(self._sessions.items())},
        }


# ----------------------------------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------------------------------

def require_object_model(model: Any) -> None:
    """Object-memory model with the native 7x77 layout and a 2-slot, 8-dim object encoder."""
    from psi.models import psi0_keyframe_native_object as object_module
    if not isinstance(model, object_module.Psi0KeyframeNativeObjectModel):
        raise TypeError("serving requires a Psi0KeyframeNativeObjectModel")
    header = model.action_header
    abi = header.keyframe_native_contract_version
    if abi.dtype != torch.int64 or abi.tolist() != [1, 7, 77, 7, 11]:
        raise ValueError("checkpoint has incompatible keyframe feature geometry")
    encoder = header.object_memory_encoder
    if (header.object_budget, encoder.object_budget, encoder.slots, encoder.geometry_dim, encoder.feature_dim) != (
            OBJECT_BUDGET, OBJECT_BUDGET, OBJECT_SLOTS, GEOMETRY_DIM, FEATURE_DIM):
        raise ValueError("checkpoint has an incompatible object-memory encoder")


def make_memory_store(model, config, device, task: TaskSpec, *, sam_repo, sam3_checkpoint, socket_dir,
                      tracker_python=None, trim_past_keep: int = 16, check_weights_sha256: bool = True,
                      startup_timeout_s: float = 900.0, memory_log: str | None = None) -> MemorySessionStore:
    """Build the frame producer from the policy's own VLM, start the SAM3 tracker and return the session store."""
    require_object_model(model)
    from psi.models.keyframe_native_qwen import QwenVisionMergerNativeFrameProducer

    m = config.model
    if m.native_keyframe_selector != "sam3" or m.object_memory_enabled is not True or m.object_memory_slots != OBJECT_SLOTS:
        raise ValueError("the checkpoint was not trained with SAM3 keyframes and 2-slot object memory")
    if (m.framesamp_memory_budget, m.framesamp_tokens_per_frame, tuple(m.native_keyframe_grid_hw)) != (539, 77, (7, 11)):
        raise ValueError("unexpected memory geometry (expected 7 frames x 77 tokens, 7x11 grid)")
    if m.framesamp_online_memory_view != MEMORY_VIEW:
        raise ValueError(f"the memory is built from the {MEMORY_VIEW!r} view")
    if list(config.data.train_repo_ids) != [task.env_id]:
        raise ValueError(f"the checkpoint was trained on {list(config.data.train_repo_ids)}, not on {task.env_id}")
    producer = QwenVisionMergerNativeFrameProducer(model=model.vlm_model, processor=model.vlm_processor, device=device,
                                                   resize_hw=tuple(config.data.transform.model.resize.size))
    tracker = TrackerClient(python=tracker_python, sam_repo=sam_repo, checkpoint=sam3_checkpoint,
                            prompt=task.sam3_prompt, namespace=task.track_namespace, n_decl=task.n_decl,
                            socket_dir=socket_dir, trim_past_keep=trim_past_keep,
                            check_weights_sha256=check_weights_sha256, startup_timeout_s=startup_timeout_s)
    try:
        tracker.hello()
        return MemorySessionStore(producer, tracker, task, memory_log=memory_log)
    except BaseException:
        tracker.close()
        raise

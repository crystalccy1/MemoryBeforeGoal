"""Causal, frozen-rule SAM tracking evidence selector for FrameSamp.

This module consumes observations; it never runs SAM or infers contact/success.
Every returned row is an immutable decision using only evidence available then.
"""

from __future__ import annotations

import copy
import heapq
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass
class _ObjectState:
    object_id: str
    prompt: str
    last_frame: int
    last_quality: float
    last_score: float
    last_area: float
    score_kind: str = "unspecified"
    missing: bool = False
    last_motion_event: int = -1_000_000_000
    last_mask_event: int = -1_000_000_000


class CausalSelector:
    """Select up to ``max_frames`` raw RGB frames for each decision frame.

    ``frame_index`` must be 0, 1, 2, ... within an episode. An observation whose
    ``available_frame`` exceeds ``frame_index`` is queued, not used early.
    Event evidence receives at most floor(max_frames * event_fraction) slots;
    remaining slots provide temporal coverage, including the current frame.
    Object identities are (prompt, object_id) tuples even if ids are namespaced.

    ``motion_residual``, when supplied, must be camera-compensated displacement
    divided by image diagonal. Values >= 0.03 can create motion candidates.
    Raw bbox displacement is deliberately not a motion-event input.
    Optional ``mask_change_residual`` is 1-IoU after valid background-affine
    registration. Values > 0.35 propose visual-change evidence, not interaction.
    """

    MOTION_RESIDUAL_THRESHOLD = 0.03
    MASK_CHANGE_RESIDUAL_THRESHOLD = 0.35
    _EVENT_PRIORITY = {
        "first_reliable_observation": 1.0,
        "reliable_reobservation": 1.25,
        "last_reliable_before_missing_observation": 1.5,
        "compensated_motion": 1.1,
        "compensated_mask_change": 1.0,
    }

    def __init__(self, max_frames: int = 32, event_fraction: float = 0.75,
                 event_min_gap: int = 5, score_threshold: float = 0.35,
                 pin_first_sighting: bool = False):
        # pin_first_sighting: the frame of each prompt's first reliable observation keeps one event slot from then on
        # (it cannot be pushed out by later events). Off by default, so existing configurations are unchanged.
        if not isinstance(pin_first_sighting, bool):
            raise ValueError("pin_first_sighting must be a bool")
        self.pin_first_sighting = pin_first_sighting
        if isinstance(max_frames, bool) or int(max_frames) != max_frames or max_frames < 1:
            raise ValueError("max_frames must be a positive integer")
        if not math.isfinite(event_fraction) or not 0 <= event_fraction <= 1:
            raise ValueError("event_fraction must be in [0, 1]")
        if isinstance(event_min_gap, bool) or int(event_min_gap) != event_min_gap or event_min_gap < 0:
            raise ValueError("event_min_gap must be a nonnegative integer")
        if not math.isfinite(score_threshold) or not 0 <= score_threshold <= 1:
            raise ValueError("score_threshold must be in [0, 1]")
        self.max_frames = int(max_frames)
        self.event_fraction = float(event_fraction)
        self.event_min_gap = int(event_min_gap)
        self.score_threshold = float(score_threshold)
        self.reset()

    def reset(self) -> None:
        """Start a new independent episode, discarding previous tracking state."""
        self._frame = -1
        self._objects: dict[tuple[str, str], _ObjectState] = {}
        self._candidates: dict[int, list[dict[str, Any]]] = {}
        self._pending: list[tuple[int, int, list[dict[str, Any]]]] = []
        self._last_evidence_source = -1
        self._last_evidence_reliable_count = 0
        self._last_evidence_object_count = 0
        self._last_evidence_score_kinds: list[str] = []
        self._ever_reliable = False
        self._snapshot: dict[str, Any] | None = None
        self._pins: dict[str, int] = {}   # prompt -> frame of its first reliable observation

    @staticmethod
    def _integer(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        return int(value)

    @staticmethod
    def _unit(value: Any, name: str) -> float:
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be finite and in [0, 1]")
        return value

    def _validate_objects(self, objects: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        validated = []
        identities = set()
        for obj in objects:
            object_id = str(obj["object_id"])
            prompt = str(obj.get("prompt", ""))
            identity = (prompt, object_id)
            if identity in identities:
                raise ValueError(f"duplicate (prompt, object_id) in observation: {identity!r}")
            identities.add(identity)
            bbox = [self._unit(v, "bbox_xyxy coordinate") for v in obj["bbox_xyxy"]]
            if len(bbox) != 4 or bbox[2] < bbox[0] or bbox[3] < bbox[1]:
                raise ValueError("bbox_xyxy must contain four ordered normalized coordinates")
            residual = obj.get("motion_residual")
            if residual is not None:
                residual = float(residual)
                if not math.isfinite(residual) or residual < 0:
                    raise ValueError("motion_residual must be finite and nonnegative")
            mask_residual = obj.get("mask_change_residual")
            if mask_residual is not None:
                mask_residual = self._unit(mask_residual, "mask_change_residual")
            validated.append({
                "object_id": object_id, "prompt": prompt, "identity": identity,
                "score": self._unit(obj["score"], "score"),
                "relevance": self._unit(obj.get("relevance", 1.0), "relevance"),
                "area": self._unit(obj["mask_area_fraction"], "mask_area_fraction"),
                "bbox_xyxy": bbox, "motion_residual": residual,
                "mask_change_residual": mask_residual,
                "score_kind": str(obj.get("score_kind", "unspecified")),
            })
        return validated

    @staticmethod
    def _quality(obj: dict[str, Any]) -> float:
        # Area is a weak quality proxy; do not let it dominate small objects.
        return obj["relevance"] * obj["score"] * (0.6 + 0.4 * math.sqrt(obj["area"]))

    def _event(self, state: _ObjectState, reason: str, source_frame: int,
               available_frame: int, observed_source_frame: int,
               event_log: list[dict[str, Any]]) -> None:
        event = {
            "source_frame": source_frame,
            "available_frame": available_frame,
            "observed_source_frame": observed_source_frame,
            "object_id": state.object_id, "prompt": state.prompt,
            "object_key": json.dumps([state.prompt, state.object_id],
                                     ensure_ascii=False, separators=(",", ":")),
            "reason": reason,
            "quality": float(state.last_quality),
            "score_kind": state.score_kind,
            "priority": self._EVENT_PRIORITY[reason],
        }
        if (self.pin_first_sighting and reason == "first_reliable_observation"
                and state.prompt not in self._pins):
            self._pins[state.prompt] = source_frame
        self._candidates.setdefault(source_frame, []).append(event)
        event_log.append(copy.deepcopy(event))

    def _consume(self, source: int, available: int, objects: list[dict[str, Any]],
                 log: list[dict[str, Any]]) -> None:
        reliable = {o["identity"]: o for o in objects
                    if o["score"] >= self.score_threshold and o["area"] > 0
                    and o["relevance"] > 0}
        self._last_evidence_reliable_count = len(reliable)
        self._last_evidence_object_count = len(objects)
        self._last_evidence_score_kinds = sorted({o["score_kind"] for o in objects})
        self._ever_reliable |= bool(reliable)
        for identity, obj in reliable.items():
            quality = self._quality(obj)
            state = self._objects.get(identity)
            if state is None:
                state = _ObjectState(obj["object_id"], obj["prompt"], source,
                                     quality, obj["score"], obj["area"], obj["score_kind"])
                self._objects[identity] = state
                reason = "first_reliable_observation"
            else:
                reason = "reliable_reobservation" if state.missing else None
                state.last_frame = source
                state.last_quality = quality
                state.last_score = obj["score"]
                state.last_area = obj["area"]
                state.score_kind = obj["score_kind"]
                state.missing = False
            if reason:
                self._event(state, reason, source, available, source, log)
            residual = obj["motion_residual"]
            if (residual is not None and residual >= self.MOTION_RESIDUAL_THRESHOLD
                    and source - state.last_motion_event >= self.event_min_gap):
                self._event(state, "compensated_motion", source, available, source, log)
                state.last_motion_event = source
            mask_residual = obj["mask_change_residual"]
            if (mask_residual is not None
                    and mask_residual > self.MASK_CHANGE_RESIDUAL_THRESHOLD
                    and source - state.last_mask_event >= self.event_min_gap):
                self._event(state, "compensated_mask_change", source, available, source, log)
                state.last_mask_event = source

        for identity, state in self._objects.items():
            if identity not in reliable and not state.missing:
                # A missing/low-confidence observation is not physical absence.
                self._event(state, "last_reliable_before_missing_observation",
                            state.last_frame, available, source, log)
                state.missing = True
        self._last_evidence_source = source

    def _ranked_events(self, current: int) -> list[int]:
        pinned = sorted(f for f in self._pins.values() if f < current)
        frames = [f for f in self._candidates if f < current and f not in pinned]
        def rank(frame: int) -> tuple[float, int]:
            evidence = self._candidates[frame]
            # Weak recency tie-break only: old unique evidence remains eligible.
            return (max(e["priority"] * e["quality"] for e in evidence), frame)
        frames.sort(key=rank, reverse=True)
        selected = []
        selected_by_object: dict[str, list[int]] = {}
        effective_budget = min(self.max_frames, current + 1)
        event_budget = min(effective_budget - 1,
                           math.floor(effective_budget * self.event_fraction))
        for frame in pinned[:event_budget]:   # pinned first sightings take their slots before ranking
            selected.append(frame)
            for key in {e["object_key"] for e in self._candidates[frame]}:
                selected_by_object.setdefault(key, []).append(frame)
        for frame in frames:
            if len(selected) >= event_budget:
                break
            keys = {e["object_key"] for e in self._candidates[frame]}
            # Suppress nearby evidence for the same identity. A distinct object
            # can justify a nearby frame, since it carries different evidence.
            novel = any(not any(abs(frame - old) < self.event_min_gap
                                for old in selected_by_object.get(key, []))
                        for key in keys)
            if not novel:
                continue
            selected.append(frame)
            for key in keys:
                selected_by_object.setdefault(key, []).append(frame)
        return selected

    @staticmethod
    def _uniform(current: int, count: int) -> list[int]:
        if count == 1:
            return [current]
        return np.rint(np.linspace(0, current, count)).astype(np.int64).tolist()

    @staticmethod
    def _fill_largest_gaps(selected: set[int], current: int, target: int) -> None:
        while len(selected) < target:
            ordered = sorted(selected)
            choices = []
            if ordered[0] > 0:
                choices.append((ordered[0], 0))
            for left, right in zip(ordered, ordered[1:]):
                if right - left > 1:
                    midpoint = (left + right) // 2
                    choices.append((min(midpoint - left, right - midpoint), midpoint))
            if ordered[-1] < current:
                choices.append((current - ordered[-1], current))
            if not choices:
                raise RuntimeError("could not fill temporal coverage despite available frames")
            selected.add(max(choices)[1])

    def update(self, record: dict[str, Any]) -> dict[str, Any]:
        current = self._integer(record["frame_index"], "frame_index")
        if current != self._frame + 1:
            raise ValueError(f"expected contiguous frame_index {self._frame + 1}, got {current}; "
                             "call reset() for a new episode")
        available = self._integer(record.get("available_frame", current), "available_frame")
        if available < current:
            raise ValueError("available_frame cannot precede source frame_index")
        objects = self._validate_objects(record.get("objects", []))
        self._frame = current
        heapq.heappush(self._pending, (available, current, objects))
        log: list[dict[str, Any]] = []
        flags = []
        while self._pending and self._pending[0][0] <= current:
            evidence_available, source, ready_objects = heapq.heappop(self._pending)
            if source <= self._last_evidence_source:
                flags.append("stale_delayed_evidence_ignored")
                continue
            self._consume(source, evidence_available, ready_objects, log)
        if self._pending:
            flags.append("tracking_evidence_pending")
        if self._last_evidence_source < 0:
            flags.append("no_tracking_evidence_available")
        elif self._last_evidence_source < current:
            flags.append("tracking_evidence_lags_current_frame")
        if self._last_evidence_reliable_count == 0:
            flags.append("no_reliable_objects_in_available_observation")
        if "initial_detection_probability" in self._last_evidence_score_kinds:
            flags.append("score_is_initial_detection_probability_not_current_confidence")
        target = min(self.max_frames, current + 1)
        event_indices = self._ranked_events(current)
        uniform_fallback = not self._ever_reliable or not event_indices
        if uniform_fallback:
            selected = set(self._uniform(current, target))
            if not self._ever_reliable:
                flags.append("no_reliable_tracking_history_uniform_fallback")
        else:
            selected = set(event_indices)
            coverage_slots = max(1, target - len(selected))
            selected.update(self._uniform(current, coverage_slots))
            selected.add(current)
            self._fill_largest_gaps(selected, current, target)
            if self._last_evidence_reliable_count == 0:
                flags.append("retaining_history_during_missing_observation")
        indices = sorted(selected)
        if len(indices) != target or indices[-1] != current:
            raise RuntimeError("internal selection budget or current-frame invariant failed")
        result = {
            "frame_index": current,
            "latest_tracking_source_frame": self._last_evidence_source,
            "observed_object_count": self._last_evidence_object_count,
            "eligible_object_count": self._last_evidence_reliable_count,
            "score_kinds": list(self._last_evidence_score_kinds),
            "using_uniform_fallback": uniform_fallback,
            "selected_raw_frame_indices": indices,
            "selected_frame_mask": [True] * len(indices),
            "selection_count": len(indices),
            "event_selection_count": len(event_indices),
            "event_frame_indices": sorted(event_indices),
            "event_reasons": log,
            "selected_event_reasons": {
                str(f): copy.deepcopy(self._candidates[f])
                for f in indices if f in self._candidates
            },
            "quality_flags": sorted(set(flags)),
        }
        self._snapshot = copy.deepcopy(result)
        return result

    def snapshot(self) -> dict[str, Any] | None:
        """Return an independent copy of the last decision, or None after reset."""
        return copy.deepcopy(self._snapshot)


def select_records(records: Iterable[dict[str, Any]], **selector_kwargs: Any) -> dict[str, Any]:
    """Process one episode and return padded [T, K] NumPy arrays plus audit logs.

    Padding is -1 with a False mask. Do not gather padded ids before masking.
    Invoke separately per episode; no state crosses episode boundaries.
    """
    selector = SELECTOR_CLASS(**selector_kwargs)
    rows, object_slots = [], []
    for record in records:
        rows.append(selector.update(record))
        # per-frame object slots of the binding policy (None: the training first-appearance policy applies)
        object_slots.append(selector.slot_objects(record) if hasattr(selector, "slot_objects") else None)
    indices = np.full((len(rows), selector.max_frames), -1, dtype=np.int64)
    mask = np.zeros_like(indices, dtype=bool)
    counts = np.zeros(len(rows), dtype=np.int32)
    for i, row in enumerate(rows):
        count = row["selection_count"]
        indices[i, :count] = row["selected_raw_frame_indices"]
        mask[i, :count] = True
        counts[i] = count
    return {
        "selected_raw_frame_indices": indices,
        "selected_frame_mask": mask,
        "selection_counts": counts,
        "event_log": [row["event_reasons"] for row in rows],
        "quality_flags": [row["quality_flags"] for row in rows],
        "rows": rows,
        "object_slots": object_slots,
        "summary": {
            "decision_count": len(rows),
            "no_observed_objects_fraction": float(np.mean([
                row["observed_object_count"] == 0 for row in rows])) if rows else None,
            "no_eligible_objects_fraction": float(np.mean([
                row["eligible_object_count"] == 0 for row in rows])) if rows else None,
            "uniform_fallback_fraction": float(np.mean([
                row["using_uniform_fallback"] for row in rows])) if rows else None,
            "event_count": sum(len(row["event_reasons"]) for row in rows),
        },
    }


# ===================== container binding =====================
import itertools

BINDING_DOC = """Container-binding keyframe selector: remember which container holds the object, also while the containers move.
Subclass of CausalSelector with pin_first_sighting on and generic rules (no task names):
 1. flicker suppression: per identity, a last_reliable_before_missing_observation / reliable_reobservation event becomes a
    candidate only if that identity had no such event in the last FLICKER_COOLDOWN frames (a carried object occluding the
    containers otherwise produces ~30 such events per episode, priority 1.5 / 1.25, and they push the cue frames out);
 2. containment: when an identity of a non-container prompt (e.g. the ball) goes missing while its last bbox centre lies in the
    expanded bbox of a reliable container-prompt identity (e.g. a cup), a 'contained' event (priority 2.0) is pinned at the
    object's last frame and that cup becomes the bound container;
 3. container following by motion continuity, not by SAM3 identity (identical cups swap places, SAM3 ids jump between
    them and duplicate at every crossing). At the binding, one track per physical container is seeded from the reliable
    container boxes (deduped by IoU >= DEDUP_IOU), in the first frame with the usual number of containers. Every frame the
    deduped boxes are assigned to the tracks (optimal, brute force) on (box centre x, box top y) - the top edge of a cup is the
    part not cut by a cup passing in front - against a constant-velocity prediction, with a per-frame gate GATE and a width
    tolerance WIDTH_TOL (width averaged slowly, so a box cut by occlusion does not reset it); an unmatched (occluded) track
    coasts on its velocity (holding it in place loses the cup that passes in front during a crossing). The bound track is the
    seed nearest the contained object's last centre. A track unmatched for LOST_FRAMES frames ends the following.
 4. container settled: once all tracks are matched and none moved by more than SETTLE_EPS over SETTLE_FRAMES frames after some
    motion, the bound track's left-to-right rank among the tracks is read. The first such rest, and any later rest at a
    DIFFERENT rank (the container sat out a swap and was moved again), emits 'container_settled' (1.9) and re-pins it; a rest
    at the same rank (e.g. after the robot walked) changes nothing. The pinned frame shows where the container ended up.
self.settles lists (frame, rank); self.container_ids maps frame -> SAM3 identity of the box assigned to the bound track.
self.binding lists (frame, identity, reason)."""

FLICKER = ("last_reliable_before_missing_observation", "reliable_reobservation")


def _centre(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def _feat(b):
    return ((b[0] + b[2]) / 2, b[1])


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


class BindingSelector(CausalSelector):
    _EVENT_PRIORITY = dict(CausalSelector._EVENT_PRIORITY, contained=2.0, container_settled=1.9, container_motion=1.8)

    def __init__(self, *args, container_prompts=("blue cup",), flicker_cooldown=100, expand=0.5, suppress_flicker=True,
                 bind=True, min_observations=5, settle_frames=25, settle_eps=0.01, dedup_iou=0.6, gate=0.06, width_tol=0.35,
                 seed_frames=30, lost_frames=50, reacq_gate=0.3, dup_eps=0.012, max_speed=0.02, **kw):
        self.min_observations = int(min_observations)
        self.settle_frames, self.settle_eps = int(settle_frames), float(settle_eps)
        self.container_prompts = set(container_prompts)
        self.flicker_cooldown, self.expand = int(flicker_cooldown), float(expand)
        self.suppress_flicker, self.bind = bool(suppress_flicker), bool(bind)
        self.dedup_iou, self.gate, self.width_tol = float(dedup_iou), float(gate), float(width_tol)
        self.seed_frames, self.lost_frames = int(seed_frames), int(lost_frames)
        self.reacq_gate, self.dup_eps, self.max_speed = float(reacq_gate), float(dup_eps), float(max_speed)
        kw.setdefault("pin_first_sighting", True)
        super().__init__(*args, **kw)

    def reset(self) -> None:
        super().reset()
        self._bbox: dict = {}          # identity -> bbox at its last reliable frame
        self._first_seen: dict = {}    # identity -> first reliable frame
        self._n_obs: dict = {}         # identity -> number of reliable observations
        self._flicker_last: dict = {}
        self._container = None         # SAM3 identity of the box currently assigned to the bound track
        self.binding: list = []
        self._n_hist: dict = {}        # number of deduped containers per frame -> count (the usual number = all visible)
        self._seed = None              # (frame, contained object's last centre) waiting for a frame with all containers
        self._tracks = None            # [{'p': (x, top), 'w': width, 'miss': frames unmatched, 'hist': [(frame, p)]}]
        self._bound = None             # index of the bound track
        self._moved = False
        self.settles: list = []        # (frame, rank)
        self.seed_rank = None
        self.container_ids: dict = {}

    def _event(self, state, reason, source_frame, available_frame, observed_source_frame, event_log) -> None:
        ident = (state.prompt, state.object_id)
        if self.suppress_flicker and reason in FLICKER:
            last = self._flicker_last.get(ident)
            if last is not None and source_frame - last < self.flicker_cooldown:
                return
            self._flicker_last[ident] = source_frame
        if reason == "compensated_motion" and self._container is not None and ident == self._container:
            reason = "container_motion"
        super()._event(state, reason, source_frame, available_frame, observed_source_frame, event_log)

    def _consume(self, source, available, objects, log) -> None:
        was_missing = {k: s.missing for k, s in self._objects.items()}
        super()._consume(source, available, objects, log)
        rel = {o["identity"]: o for o in objects
               if o["score"] >= self.score_threshold and o["area"] > 0 and o["relevance"] > 0}
        for ident, o in rel.items():
            self._first_seen.setdefault(ident, source)
            self._n_obs[ident] = self._n_obs.get(ident, 0) + 1
        if self.bind:
            # 2. containment: a non-container identity that went missing in this observation
            for ident, s in self._objects.items():
                if not (s.missing and not was_missing.get(ident, True)) or s.prompt in self.container_prompts:
                    continue
                if ident not in self._bbox or self._n_obs.get(ident, 0) < self.min_observations:
                    continue
                cx, cy = _centre(self._bbox[ident])
                best = None
                for cid, co in rel.items():
                    if co["prompt"] not in self.container_prompts:
                        continue
                    x0, y0, x1, y1 = co["bbox_xyxy"]
                    w, h = x1 - x0, y1 - y0
                    if x0 - self.expand * w <= cx <= x1 + self.expand * w and y0 - self.expand * h <= cy <= y1 + self.expand * h:
                        d = abs((x0 + x1) / 2 - cx)
                        if best is None or d < best[0]:
                            best = (d, cid)
                if best is not None:
                    self._container = best[1]
                    self.container_ids[source] = best[1]
                    self._seed, self._tracks, self._bound, self._moved = (s.last_frame, (cx, cy)), None, None, False
                    super()._event(self._objects[best[1]], "contained", s.last_frame, available, source, log)
                    self._pins["__contained__"] = s.last_frame
                    self.binding.append((s.last_frame, best[1], "contained"))
            dets = self._containers(rel)
            if dets:
                self._n_hist[len(dets)] = self._n_hist.get(len(dets), 0) + 1
            n_all = max(self._n_hist, key=lambda k: (self._n_hist[k], k)) if self._n_hist else 0
            # 3. seed the tracks in the first frame with all containers, then follow by continuity
            if self._seed is not None and self._tracks is None:
                if source - self._seed[0] > self.seed_frames:
                    self._seed = None
                elif len(dets) == n_all and n_all >= 2:
                    bc = self._seed[1]
                    self._tracks = [{"p": _feat(b), "v": (0.0, 0.0), "w": b[2] - b[0], "miss": 0, "hist": [(source, _feat(b))],
                                     "id": i} for i, b in dets]
                    self._bound = min(range(len(dets)), key=lambda k: abs(self._tracks[k]["p"][0] - bc[0]))
                    self.seed_rank = sorted(range(len(dets)), key=lambda k: self._tracks[k]["p"][0]).index(self._bound) + 1
                    self._container = dets[self._bound][0]
                    self.container_ids[source] = self._container
                    self.binding.append((source, dets[self._bound][0], f"tracks_seeded rank {self.seed_rank}"))
                    self._seed = None
            elif self._tracks is not None:
                self._follow(source, dets, available, log, n_all)
        for ident, o in rel.items():
            self._bbox[ident] = o["bbox_xyxy"]

    def _containers(self, rel):
        cs = sorted(((i, o) for i, o in rel.items() if o["prompt"] in self.container_prompts), key=lambda io: -io[1]["score"])
        keep = []
        for i, o in cs:
            q = _feat(o["bbox_xyxy"])
            # a duplicate SAM3 box of the same cup: high IoU, or the same centre x and top edge (a second box cut short by
            # the cup in front)
            if all(_iou(o["bbox_xyxy"], b) < self.dedup_iou and
                   (abs(q[0] - _feat(b)[0]) > self.dup_eps or abs(q[1] - _feat(b)[1]) > self.dup_eps) for _, b in keep):
                keep.append((i, o["bbox_xyxy"]))
        return keep

    def _follow(self, source, dets, available, log, n_all) -> None:
        tr = self._tracks
        pred = [(min(1.0, max(0.0, t["p"][0] + t["v"][0])), min(1.0, max(0.0, t["p"][1] + t["v"][1]))) for t in tr]
        best, bcost = {}, None
        if len(dets) <= 6:
            for perm in itertools.permutations(list(range(len(dets))) + [None] * len(tr), len(tr)):
                cost = 0.0
                for k, d in enumerate(perm):
                    if d is None:
                        cost += self.gate; continue
                    b = dets[d][1]; q = _feat(b); w = b[2] - b[0]
                    dd = ((q[0] - pred[k][0]) ** 2 + (q[1] - pred[k][1]) ** 2) ** 0.5
                    if dd > self.gate or abs(w - tr[k]["w"]) > self.width_tol * tr[k]["w"]:
                        cost = None; break
                    cost += dd
                if cost is not None and (bcost is None or cost < bcost):
                    bcost, best = cost, {k: d for k, d in enumerate(perm) if d is not None}
        # re-acquisition (a cup passing in front can lose its box for ~20 frames and come back as a new
        # SAM3 id beyond the per-frame gate): the containers are conserved, so when every container is visible as a distinct
        # box, the unmatched boxes go to the unmatched (coasting) tracks by the nearest prediction, within REACQ_GATE.
        if len(dets) == len(tr) and len(best) < len(tr):
            free_t = [k for k in range(len(tr)) if k not in best]
            free_d = [d for d in range(len(dets)) if d not in best.values()]
            pairs = None
            for perm in itertools.permutations(free_d, len(free_t)):
                cost = sum(((_feat(dets[d][1])[0] - pred[k][0]) ** 2 + (_feat(dets[d][1])[1] - pred[k][1]) ** 2) ** 0.5
                           for k, d in zip(free_t, perm))
                if pairs is None or cost < pairs[0]:
                    pairs = (cost, list(zip(free_t, perm)))
            # one missing track and one free box: the same container (conservation), whatever the prediction says
            if pairs is not None and (len(free_t) == 1 or all(
                    ((_feat(dets[d][1])[0] - pred[k][0]) ** 2 + (_feat(dets[d][1])[1] - pred[k][1]) ** 2) ** 0.5
                    <= self.reacq_gate for k, d in pairs[1])):
                for k, d in pairs[1]:
                    best[k] = d
                    tr[k]["v"] = (0.0, 0.0)          # a re-acquired track restarts its velocity
        for k, t in enumerate(tr):
            if k in best:
                i, b = dets[best[k]]
                q = _feat(b)
                cap = lambda v: max(-self.max_speed, min(self.max_speed, v))
                t["v"] = (cap(0.5 * t["v"][0] + 0.5 * (q[0] - t["p"][0])), cap(0.5 * t["v"][1] + 0.5 * (q[1] - t["p"][1])))
                t["p"], t["w"], t["miss"], t["id"] = q, 0.9 * t["w"] + 0.1 * (b[2] - b[0]), 0, i
            else:
                t["p"], t["v"] = pred[k], (0.9 * t["v"][0], 0.9 * t["v"][1])
                t["miss"] += 1
            t["hist"] = [h for h in t["hist"] if source - h[0] <= self.settle_frames] + [(source, t["p"])]
        if tr[self._bound]["miss"] > self.lost_frames:
            self._tracks = None          # the containers left the view: stop following
            return
        self._container = tr[self._bound]["id"] if tr[self._bound]["miss"] == 0 else None
        if self._container is not None:
            self.container_ids[source] = self._container
        # 4. settled: all tracks matched and still over the window, after some motion
        span = max(max(abs(a[1][0] - b[1][0]), abs(a[1][1] - b[1][1])) for t in tr for a in t["hist"] for b in t["hist"])
        if span > self.settle_eps:
            self._moved = True
        elif (self._moved and all(t["miss"] == 0 for t in tr) and all(source - t["hist"][0][0] >= self.settle_frames - 1 for t in tr)):
            self._moved = False
            rank = sorted(range(len(tr)), key=lambda k: tr[k]["p"][0]).index(self._bound) + 1
            if not self.settles or self.settles[-1][1] != rank:
                self.settles.append((source, rank))
                state = self._objects.get(self._container)
                if state is not None:
                    super()._event(state, "container_settled", source, available, source, log)
                self._pins["__settled__"] = source
                self.binding.append((source, self._container, "container_settled"))

    def slot_objects(self, record) -> dict:
        """Object slots of the binding policy for this record's frame (call right after update(record)).
        slot 0: the box assigned to the bound container's track at this frame (absent before the binding, while the
                container is occluded and after the following ended);
        slot 1: the highest-scoring eligible object of a non-container prompt (e.g. the ball while visible).
        Eligible = training's sam3_object_memory.eligible (score >= 0.35 and mask area > 0). Returns {slot: record object}."""
        t = int(record["frame_index"])
        ok = [o for o in record.get("objects", []) if float(o.get("score", 0.0)) >= 0.35 and float(o.get("mask_area_fraction", 0.0)) > 0]
        present = {}
        cid = self.container_ids.get(t)
        if cid is not None:
            hit = [o for o in ok if (str(o.get("prompt", "")), str(o["object_id"])) == cid]
            if hit:
                present[0] = hit[0]
        other = [o for o in ok if str(o.get("prompt", "")) not in self.container_prompts]
        if other:
            present[1] = max(other, key=lambda o: (float(o["score"]), str(o["object_id"])))
        return present


SELECTOR_CLASS = BindingSelector


# ===================== hidden-object memory =====================
HIDDEN_DOC = """Hidden-object keyframe selector. Subclass of the pin-capable CausalSelector with pin_first_sighting on and
four generic rules (no task names):
 1. gone: an identity with >= MIN_OBSERVATIONS reliable observations that has had no reliable observation for GONE_FRAMES
    frames is gone; its last reliable box decides how it went:
 2. left the view (last box within BORDER of an image edge): a disturbance. Its candidate events are dropped from the event
    pool (from the frame this becomes known on), so it no longer takes memory slots;
 3. hidden (last box inside the view): the object was occluded by something else. Its LAST CLEAR VIEW (the reliable observation
    with the largest mask area among its last CLEAR_WINDOW frames before it went missing) gets a 'hidden' event (priority 2.0)
    and is pinned in memory. Only the first hidden object is remembered (later hides do not replace it);
 4. object slots (slot_objects): slot 0 = the hidden object's last clear view, persisted after it disappeared (record object
    with '_source_frame' = that frame and '_hidden' = True: pool it over that frame's native tokens, geometry 'visible' = 0),
    or the live object if it is reliably seen again; slot 1 = the highest-scoring eligible object that is neither the hidden
    object nor an object that left the view. Before any hide, slot 0 is absent.
self.hidden = (identity, clear_frame, gone_on_frame) once known; self.left_view lists (identity, last_frame, gone_on_frame)."""


class HiddenSelector(CausalSelector):
    _EVENT_PRIORITY = dict(CausalSelector._EVENT_PRIORITY, hidden=2.0)
    BORDER, GONE_FRAMES, CLEAR_WINDOW, MIN_OBSERVATIONS = 0.04, 25, 40, 5

    def __init__(self, *args, container_prompts=(), **kw):   # container_prompts: accepted for the binding-alias plumbing, unused
        kw.setdefault("pin_first_sighting", True)
        super().__init__(*args, **kw)

    def reset(self) -> None:
        super().reset()
        self._seen: dict = {}          # identity -> list of (frame, area, bbox) reliable observations (last CLEAR_WINDOW frames)
        self._n_obs: dict = {}
        self._decided: set = set()     # identities already classified as hidden / left (until seen again)
        self._raw: dict = {}           # identity -> {frame: raw record object} of the last CLEAR_WINDOW frames (filled by slot_objects)
        self._left: set = set()
        self.hidden = None
        self.left_view: list = []
        self._hidden_obj = None        # raw record object of the hidden object's clear view

    def _consume(self, source, available, objects, log) -> None:
        super()._consume(source, available, objects, log)
        rel = {o["identity"]: o for o in objects if o["score"] >= self.score_threshold and o["area"] > 0 and o["relevance"] > 0}
        for ident, o in rel.items():
            self._n_obs[ident] = self._n_obs.get(ident, 0) + 1
            h = self._seen.setdefault(ident, [])
            h.append((source, o["area"], o["bbox_xyxy"]))
            while h and source - h[0][0] >= self.CLEAR_WINDOW:
                h.pop(0)
            if ident in self._decided:       # seen again: it is back (a hidden object reappears live in slot 0)
                self._decided.discard(ident); self._left.discard(ident)
        for ident, h in self._seen.items():
            if ident in rel or ident in self._decided or not h or self._n_obs.get(ident, 0) < self.MIN_OBSERVATIONS:
                continue
            last_f, _, b = h[-1]
            if source - last_f < self.GONE_FRAMES:
                continue
            self._decided.add(ident)
            if b[0] < self.BORDER or b[2] > 1 - self.BORDER or b[1] < self.BORDER or b[3] > 1 - self.BORDER:
                self._left.add(ident); self.left_view.append((ident, last_f, source))
                key = json.dumps([ident[0], ident[1]], ensure_ascii=False, separators=(",", ":"))
                for f in list(self._candidates):
                    self._candidates[f] = [e for e in self._candidates[f] if e["object_key"] != key]
                    if not self._candidates[f] and f not in self._pins.values():
                        del self._candidates[f]
            elif self.hidden is None:
                clear_f = max(h, key=lambda x: (x[1], x[0]))[0]       # largest mask area, latest on ties
                self.hidden = (ident, clear_f, source)
                state = self._objects[ident]
                super()._event(state, "hidden", clear_f, available, source, log)
                self._pins["__hidden__"] = clear_f

    def slot_objects(self, record) -> dict:
        t = int(record["frame_index"])
        ok = [o for o in record.get("objects", []) if float(o.get("score", 0.0)) >= 0.35 and float(o.get("mask_area_fraction", 0.0)) > 0]
        for o in ok:
            ident = (str(o.get("prompt", "")), str(o["object_id"]))
            d = self._raw.setdefault(ident, {}); d[t] = o
            for f in [f for f in d if t - f >= self.CLEAR_WINDOW]:
                del d[f]
        present = {}
        if self.hidden is not None:
            ident, clear_f, _ = self.hidden
            live = [o for o in ok if (str(o.get("prompt", "")), str(o["object_id"])) == ident]
            if live:
                present[0] = live[0]
            else:
                if self._hidden_obj is None:
                    self._hidden_obj = dict(self._raw[ident][clear_f], _source_frame=int(clear_f), _hidden=True)
                present[0] = self._hidden_obj
        hid = self.hidden[0] if self.hidden is not None else None
        other = [o for o in ok if (str(o.get("prompt", "")), str(o["object_id"])) not in self._left
                 and (str(o.get("prompt", "")), str(o["object_id"])) != hid]
        if other:
            present[1] = max(other, key=lambda o: (float(o["score"]), str(o["object_id"])))
        return present


SELECTOR_CLASS = HiddenSelector

# ===================== flexible keyframes, cap 7 (hidden object) =====================
class FlexHiddenSelector(HiddenSelector):
    """Hidden-object selector + fewer keyframes when fewer are enough. Same evidence intake and event ranking as CausalSelector, but a
    row is NOT filled up to max_frames: it holds the ranked events (at most max_frames - COVERAGE, same per-identity gap
    suppression) plus COVERAGE uniform frames {0, t/2, t} (fewer early on), so its length varies from 1 to max_frames.
    Applied over HiddenSelector (events, pins, hiding rules and slot_objects unchanged)."""
    COVERAGE = 3

    def __init__(self, *args, **kw):
        kw["event_fraction"] = (kw.get("max_frames", 7) - self.COVERAGE) / kw.get("max_frames", 7) + 1e-9
        super().__init__(*args, **kw)

    def update(self, record: dict[str, Any]) -> dict[str, Any]:
        current = self._integer(record["frame_index"], "frame_index")
        if current != self._frame + 1:
            raise ValueError(f"expected contiguous frame_index {self._frame + 1}, got {current}; "
                             "call reset() for a new episode")
        available = self._integer(record.get("available_frame", current), "available_frame")
        if available < current:
            raise ValueError("available_frame cannot precede source frame_index")
        objects = self._validate_objects(record.get("objects", []))
        self._frame = current
        heapq.heappush(self._pending, (available, current, objects))
        log: list[dict[str, Any]] = []
        flags = []
        while self._pending and self._pending[0][0] <= current:
            evidence_available, source, ready_objects = heapq.heappop(self._pending)
            if source <= self._last_evidence_source:
                flags.append("stale_delayed_evidence_ignored")
                continue
            self._consume(source, evidence_available, ready_objects, log)
        if self._pending:
            flags.append("tracking_evidence_pending")
        if self._last_evidence_source < 0:
            flags.append("no_tracking_evidence_available")
        elif self._last_evidence_source < current:
            flags.append("tracking_evidence_lags_current_frame")
        if self._last_evidence_reliable_count == 0:
            flags.append("no_reliable_objects_in_available_observation")
        if "initial_detection_probability" in self._last_evidence_score_kinds:
            flags.append("score_is_initial_detection_probability_not_current_confidence")
        event_indices = self._ranked_events(current)
        uniform_fallback = not self._ever_reliable or not event_indices
        selected = set(self._uniform(current, min(self.COVERAGE, current + 1)))
        selected.add(current)
        if uniform_fallback:
            if not self._ever_reliable:
                flags.append("no_reliable_tracking_history_uniform_fallback")
        else:
            selected.update(event_indices)
            if self._last_evidence_reliable_count == 0:
                flags.append("retaining_history_during_missing_observation")
        indices = sorted(selected)
        if not 1 <= len(indices) <= min(self.max_frames, current + 1) or indices[-1] != current:
            raise RuntimeError("internal flexible selection budget or current-frame invariant failed")
        result = {
            "frame_index": current,
            "latest_tracking_source_frame": self._last_evidence_source,
            "observed_object_count": self._last_evidence_object_count,
            "eligible_object_count": self._last_evidence_reliable_count,
            "score_kinds": list(self._last_evidence_score_kinds),
            "using_uniform_fallback": uniform_fallback,
            "selected_raw_frame_indices": indices,
            "selected_frame_mask": [True] * len(indices),
            "selection_count": len(indices),
            "event_selection_count": len(event_indices),
            "event_frame_indices": sorted(event_indices),
            "event_reasons": log,
            "selected_event_reasons": {
                str(f): copy.deepcopy(self._candidates[f])
                for f in indices if f in self._candidates
            },
            "quality_flags": sorted(set(flags)),
        }
        self._snapshot = copy.deepcopy(result)
        return result


SELECTOR_CLASS = FlexHiddenSelector

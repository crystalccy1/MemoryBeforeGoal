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
                 event_min_gap: int = 5, score_threshold: float = 0.35):
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
        frames = [f for f in self._candidates if f < current]
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
    selector = SELECTOR_CLASS(**selector_kwargs)   # the last SELECTOR_CLASS assignment below
    rows = [selector.update(record) for record in records]
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


# ===================== flexible keyframes, cap 7 =====================
class FlexSelector(CausalSelector):
    """Fewer keyframes when fewer are enough. Same evidence intake and event ranking as CausalSelector, but a
    row is NOT filled up to max_frames: it holds the ranked events (at most max_frames - COVERAGE, same per-identity gap
    suppression) plus COVERAGE uniform frames {0, t/2, t} (fewer early on), so its length varies from 1 to max_frames.
    This is the adaptive keyframe memory of the WholeBodyMem checkpoints."""
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


SELECTOR_CLASS = FlexSelector

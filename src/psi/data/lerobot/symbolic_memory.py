from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch


TRAINING_SCHEMA = "psi0_symbolic_policy_training_rows/v1"
PREP_SCHEMA = "psi0_symbolic_policy_prep/v1"
SUPPORTED_LABEL_TIER = "action_derived_oracle_proxy"
SUPPORTED_MODES = {"correct", "empty", "shuffled", "wrong"}
WRONG_SUBGOAL_MAP = {0: 4, 1: 3, 2: 0, 3: 1, 4: 0}


class SymbolicMemoryDataError(ValueError):
    """Raised when the prepared symbolic-memory contract is unsafe or stale."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scalar_int(value: Any, field: str) -> int:
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise SymbolicMemoryDataError(f"{field} must contain one value")
        return int(value.item())
    array = np.asarray(value)
    if array.size != 1:
        raise SymbolicMemoryDataError(f"{field} must contain one value")
    return int(array.reshape(-1)[0])


@dataclass(frozen=True)
class SymbolicMemoryRecord:
    symbolic_instruction: str
    current_subgoal: str
    subgoal_id: int
    phase_action_mask: np.ndarray
    label_tier: str
    mode: str
    correct_subgoal_id: int
    correct_current_subgoal: str
    control_matches_correct: bool
    control_source_row: int | None


class SymbolicMemorySidecar:
    """Verified O(1) lookup over the prepared per-frame symbolic sidecar."""

    def __init__(
        self,
        rows_path: str | Path,
        *,
        expected_task: str,
        expected_rows: int,
        expected_action_horizon: int,
        mode: str = "correct",
        seed: int = 292285,
    ) -> None:
        if mode not in SUPPORTED_MODES:
            raise SymbolicMemoryDataError(
                f"Unsupported symbolic-memory mode: {mode}"
            )
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise SymbolicMemoryDataError("Symbolic-memory seed must be >= 0")
        raw_path = Path(rows_path).expanduser()
        try:
            path = raw_path.resolve(strict=True)
        except OSError as exc:
            raise SymbolicMemoryDataError(
                f"Symbolic rows are unavailable: {raw_path}"
            ) from exc
        if not path.is_file() or path.suffix != ".parquet":
            raise SymbolicMemoryDataError(
                f"Symbolic rows must be one Parquet file: {path}"
            )
        manifest_path = path.parent / "training_manifest.json"
        ready_path = path.parent / "READY"
        if not manifest_path.is_file() or not ready_path.is_file():
            raise SymbolicMemoryDataError(
                "Symbolic rows require adjacent training_manifest.json and READY"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            ready = json.loads(ready_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SymbolicMemoryDataError(
                "Symbolic manifest or READY is invalid JSON"
            ) from exc
        output = manifest.get("outputs", {}).get(path.name)
        if (
            manifest.get("schema_version") != PREP_SCHEMA
            or manifest.get("task") != expected_task
            or not isinstance(output, dict)
            or output.get("rows") != expected_rows
            or output.get("bytes") != path.stat().st_size
            or output.get("sha256") != _sha256_file(path)
            or ready.get("manifest") != manifest_path.name
            or ready.get("manifest_bytes") != manifest_path.stat().st_size
            or ready.get("manifest_sha256") != _sha256_file(manifest_path)
        ):
            raise SymbolicMemoryDataError(
                "Symbolic rows do not match their frozen manifest"
            )
        labels = manifest.get("labels")
        policy_contract = manifest.get("policy_contract")
        readiness = manifest.get("readiness")
        if (
            not isinstance(labels, dict)
            or labels.get("label_tier") != SUPPORTED_LABEL_TIER
            or labels.get("is_privileged_oracle") is not False
            or not isinstance(policy_contract, dict)
            or policy_contract.get("action_horizon") != expected_action_horizon
            or not isinstance(readiness, dict)
            or readiness.get("data_preparation_ready") is not True
            or readiness.get("training_started") is not False
        ):
            raise SymbolicMemoryDataError(
                "Symbolic label tier, horizon, or readiness contract is invalid"
            )

        required_columns = [
            "schema_version",
            "task",
            "episode_index",
            "frame_index",
            "dataset_index",
            "baseline_instruction",
            "symbolic_instruction",
            "current_subgoal",
            "subgoal_id",
            "phase_action_mask",
            "label_tier",
        ]
        table = pq.read_table(path, columns=required_columns)
        if table.num_rows != expected_rows:
            raise SymbolicMemoryDataError(
                f"Symbolic row count mismatch: {table.num_rows} != {expected_rows}"
            )
        schema_values = set(table["schema_version"].to_pylist())
        task_values = set(table["task"].to_pylist())
        label_tiers = set(table["label_tier"].to_pylist())
        if (
            schema_values != {TRAINING_SCHEMA}
            or task_values != {expected_task}
            or label_tiers != {SUPPORTED_LABEL_TIER}
        ):
            raise SymbolicMemoryDataError(
                "Symbolic Parquet schema, task, or label tier drifted"
            )

        self.path = path
        self.manifest_path = manifest_path
        self.manifest = manifest
        self.rows_sha256 = str(output["sha256"])
        self.action_horizon = expected_action_horizon
        self.mode = mode
        self.seed = seed
        self._episode_index = np.asarray(
            table["episode_index"].to_numpy(), dtype=np.int64
        )
        self._frame_index = np.asarray(
            table["frame_index"].to_numpy(), dtype=np.int64
        )
        self._dataset_index = np.asarray(
            table["dataset_index"].to_numpy(), dtype=np.int64
        )
        self._baseline_instruction = table["baseline_instruction"].to_pylist()
        self._symbolic_instruction = table["symbolic_instruction"].to_pylist()
        self._current_subgoal = table["current_subgoal"].to_pylist()
        self._subgoal_id = np.asarray(
            table["subgoal_id"].to_numpy(), dtype=np.int64
        )
        self._phase_action_mask = np.asarray(
            table["phase_action_mask"].to_pylist(), dtype=np.bool_
        )
        if self._phase_action_mask.shape != (
            expected_rows,
            expected_action_horizon,
        ):
            raise SymbolicMemoryDataError(
                "Symbolic phase_action_mask shape mismatch"
            )
        keys = list(zip(self._episode_index.tolist(), self._frame_index.tolist()))
        self._key_to_row = {key: row for row, key in enumerate(keys)}
        if len(self._key_to_row) != expected_rows:
            raise SymbolicMemoryDataError("Duplicate symbolic episode/frame key")
        phase_ids = sorted(set(self._subgoal_id.tolist()))
        self._phase_text: dict[int, str] = {}
        for phase_id in phase_ids:
            values = {
                str(self._current_subgoal[row])
                for row in np.flatnonzero(self._subgoal_id == phase_id)
            }
            if len(values) != 1:
                raise SymbolicMemoryDataError(
                    f"Subgoal text drift for phase {phase_id}: {sorted(values)}"
                )
            self._phase_text[phase_id] = values.pop()
        if set(phase_ids) == set(WRONG_SUBGOAL_MAP):
            self._wrong_subgoal_map = dict(WRONG_SUBGOAL_MAP)
        else:
            self._wrong_subgoal_map = {
                phase_id: phase_ids[(index + 1) % len(phase_ids)]
                for index, phase_id in enumerate(phase_ids)
            }
        rng = np.random.default_rng(seed)
        self._shuffled_row = rng.permutation(expected_rows)
        shuffled_matches = int(
            np.sum(self._subgoal_id == self._subgoal_id[self._shuffled_row])
        )
        self.control_stats = {
            "mode": mode,
            "seed": seed,
            "rows": expected_rows,
            "shuffled_matches_correct": shuffled_matches,
            "shuffled_match_fraction": shuffled_matches / expected_rows,
            "wrong_matches_correct": 0,
        }

    def __len__(self) -> int:
        return len(self._episode_index)

    def lookup(self, episode_index: int, frame_index: int) -> SymbolicMemoryRecord:
        row = self._key_to_row.get((episode_index, frame_index))
        if row is None:
            raise SymbolicMemoryDataError(
                f"Missing symbolic row: episode={episode_index}, frame={frame_index}"
            )
        correct_subgoal_id = int(self._subgoal_id[row])
        correct_current_subgoal = str(self._current_subgoal[row])
        control_source_row: int | None = None
        if self.mode == "correct":
            subgoal_id = correct_subgoal_id
            current_subgoal = correct_current_subgoal
            symbolic_instruction = str(self._symbolic_instruction[row])
        elif self.mode == "empty":
            subgoal_id = -1
            current_subgoal = ""
            symbolic_instruction = (
                f"{self._baseline_instruction[row]}\nCurrent Subgoal:"
            )
        elif self.mode == "shuffled":
            control_source_row = int(self._shuffled_row[row])
            subgoal_id = int(self._subgoal_id[control_source_row])
            current_subgoal = str(self._current_subgoal[control_source_row])
            symbolic_instruction = (
                f"{self._baseline_instruction[row]}\n"
                f"Current Subgoal: {current_subgoal}"
            )
        else:
            subgoal_id = self._wrong_subgoal_map[correct_subgoal_id]
            current_subgoal = self._phase_text[subgoal_id]
            symbolic_instruction = (
                f"{self._baseline_instruction[row]}\n"
                f"Current Subgoal: {current_subgoal}"
            )
        return SymbolicMemoryRecord(
            symbolic_instruction=symbolic_instruction,
            current_subgoal=current_subgoal,
            subgoal_id=subgoal_id,
            phase_action_mask=self._phase_action_mask[row].copy(),
            label_tier=SUPPORTED_LABEL_TIER,
            mode=self.mode,
            correct_subgoal_id=correct_subgoal_id,
            correct_current_subgoal=correct_current_subgoal,
            control_matches_correct=subgoal_id == correct_subgoal_id,
            control_source_row=control_source_row,
        )

    def augment_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        if "episode_index" not in sample or "frame_index" not in sample:
            raise SymbolicMemoryDataError(
                "LeRobot sample is missing episode_index or frame_index"
            )
        episode_index = _scalar_int(sample["episode_index"], "episode_index")
        frame_index = _scalar_int(sample["frame_index"], "frame_index")
        record = self.lookup(episode_index, frame_index)
        result = dict(sample)
        result.update(
            {
                "symbolic_instruction": record.symbolic_instruction,
                "current_subgoal": record.current_subgoal,
                "subgoal_id": torch.tensor(record.subgoal_id, dtype=torch.int64),
                "phase_action_mask": torch.from_numpy(record.phase_action_mask),
                "symbolic_memory_label_tier": record.label_tier,
                "symbolic_memory_mode": record.mode,
                "correct_subgoal_id": torch.tensor(
                    record.correct_subgoal_id, dtype=torch.int64
                ),
                "correct_current_subgoal": record.correct_current_subgoal,
                "control_matches_correct": torch.tensor(
                    record.control_matches_correct, dtype=torch.bool
                ),
                "control_source_row": torch.tensor(
                    -1
                    if record.control_source_row is None
                    else record.control_source_row,
                    dtype=torch.int64,
                ),
            }
        )
        return result

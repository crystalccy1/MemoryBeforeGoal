"""Auditable identity and RNG helpers for the Psi0 SIMPLE policy server.

This module deliberately has no dependency on ``Psi0Model`` or FastAPI so the
startup contract can be tested without loading a checkpoint onto a GPU.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import random
import re
import stat
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


SIMPLE_SNAPSHOT_MARKER = "psi0_simple_snapshot.json"
THIN_SIMPLE_COMPLETE_TEXT = "verified thin SIMPLE snapshot"
INFERENCE_NOISE_PROTOCOL = "per_rollout_global_rng_reset_v1"
IDENTITY_SCHEMA_VERSION = 1
RNG_AUDIT_SCHEMA_VERSION = 1
MAX_INFERENCE_NOISE_SEED = 2**32 - 1
SAFE_CHECKPOINT_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/@:-]{0,511}")
CHECKPOINT_MANIFEST_KEYS = (
    "checkpoint_id",
    "step",
    "model_sha256",
    "model_bytes",
    "source_run",
    "train_seed",
    "COMPLETE",
    "model_path",
)


class SimpleServerAuditError(RuntimeError):
    """The policy server cannot establish an auditable startup contract."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise SimpleServerAuditError(f"{label} is not a lowercase SHA-256 digest")
    return digest


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SimpleServerAuditError(f"Cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SimpleServerAuditError(f"{label} must contain a JSON object: {path}")
    return value


def file_identity(
    path: Path,
    *,
    label: str,
    allow_symlink: bool = False,
    chunk_bytes: int = 8 * 1024 * 1024,
) -> dict[str, Any]:
    """Return bytes/SHA-256 while rejecting missing or changing artifacts."""

    path = Path(path).expanduser()
    if path.is_symlink() and not allow_symlink:
        raise SimpleServerAuditError(f"{label} must not be a symlink: {path}")
    try:
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise SimpleServerAuditError(f"{label} is not a regular file: {path}")
            digest = hashlib.sha256()
            measured_bytes = 0
            while chunk := stream.read(chunk_bytes):
                digest.update(chunk)
                measured_bytes += len(chunk)
            after = os.fstat(stream.fileno())
    except SimpleServerAuditError:
        raise
    except OSError as exc:
        raise SimpleServerAuditError(f"Cannot read {label}: {path}: {exc}") from exc

    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, name) != getattr(after, name) for name in stable_fields):
        raise SimpleServerAuditError(f"{label} changed while it was hashed: {path}")
    if measured_bytes != before.st_size or measured_bytes <= 0:
        raise SimpleServerAuditError(
            f"{label} byte count is invalid: measured={measured_bytes}, stat={before.st_size}"
        )
    return {
        "path": str(path.resolve()),
        "bytes": measured_bytes,
        "sha256": digest.hexdigest(),
    }


def _validate_thin_snapshot_marker(
    *,
    run_dir: Path,
    checkpoint_step: int,
    model: Mapping[str, Any],
    argv: Mapping[str, Any],
    run_config: Mapping[str, Any],
) -> dict[str, Any]:
    marker_path = run_dir / SIMPLE_SNAPSHOT_MARKER
    marker_file = file_identity(
        marker_path,
        label="official thin SIMPLE snapshot marker",
    )
    marker = _read_json_object(
        marker_path,
        label="official thin SIMPLE snapshot marker",
    )
    if marker.get("schema_version") != 1:
        raise SimpleServerAuditError(
            f"Unsupported official thin SIMPLE snapshot schema: {marker_path}"
        )
    if marker.get("checkpoint_step") != checkpoint_step:
        raise SimpleServerAuditError(
            "Official thin SIMPLE snapshot checkpoint_step does not match the requested step"
        )
    if marker.get("checkpoint_bytes") != model["bytes"]:
        raise SimpleServerAuditError(
            "Official thin SIMPLE snapshot checkpoint_bytes does not match model.safetensors"
        )
    if marker.get("checkpoint_sha256") != model["sha256"]:
        raise SimpleServerAuditError(
            "Official thin SIMPLE snapshot checkpoint_sha256 does not match model.safetensors"
        )
    if marker.get("patched_argv_sha256") != argv["sha256"]:
        raise SimpleServerAuditError(
            "Official thin SIMPLE snapshot argv digest does not match argv.txt"
        )

    run_config_path = run_dir / "run_config.json"
    run_config_object = _read_json_object(run_config_path, label="run config")
    canonical_config_sha256 = _json_sha256(run_config_object)
    if marker.get("patched_run_config_sha256") != canonical_config_sha256:
        raise SimpleServerAuditError(
            "Official thin SIMPLE snapshot config digest does not match run_config.json"
        )

    snapshot_id = _require_sha256(marker.get("snapshot_id"), label="snapshot_id")
    snapshot_identity = {
        key: value
        for key, value in marker.items()
        if key not in {"schema_version", "created_at", "snapshot_id"}
    }
    if _json_sha256(snapshot_identity) != snapshot_id:
        raise SimpleServerAuditError(
            f"Official thin SIMPLE snapshot_id does not bind its identity: {marker_path}"
        )
    return {
        **marker_file,
        "snapshot_id": snapshot_id,
        "canonical_run_config_sha256": canonical_config_sha256,
    }


def inspect_simple_checkpoint(run_dir: Path, checkpoint_step: int) -> dict[str, Any]:
    """Verify a trainer checkpoint or prepared official thin SIMPLE snapshot.

    A trainer checkpoint is publishable only when ``COMPLETE`` contains the
    exact requested step. The official thin snapshot text is accepted only
    when its provenance JSON independently binds the model and metadata.
    """

    run_dir = Path(run_dir).expanduser().resolve()
    if isinstance(checkpoint_step, bool) or not isinstance(checkpoint_step, int):
        raise SimpleServerAuditError("checkpoint_step must be an integer")
    if checkpoint_step < 0:
        raise SimpleServerAuditError("checkpoint_step must be non-negative")
    if not run_dir.is_dir() or run_dir.is_symlink():
        raise SimpleServerAuditError(f"run_dir is not a real directory: {run_dir}")

    checkpoint_dir = run_dir / "checkpoints" / f"ckpt_{checkpoint_step}"
    if not checkpoint_dir.is_dir() or checkpoint_dir.is_symlink():
        raise SimpleServerAuditError(
            f"checkpoint directory does not exist or is a symlink: {checkpoint_dir}"
        )

    model = file_identity(
        checkpoint_dir / "model.safetensors",
        label="checkpoint model",
        allow_symlink=True,
    )
    run_config = file_identity(run_dir / "run_config.json", label="run config")
    argv = file_identity(run_dir / "argv.txt", label="launch argv")
    complete_path = checkpoint_dir / "COMPLETE"
    complete = file_identity(complete_path, label="checkpoint COMPLETE marker")
    try:
        complete_text = complete_path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise SimpleServerAuditError(
            f"Cannot read checkpoint COMPLETE marker: {complete_path}: {exc}"
        ) from exc

    snapshot_marker: dict[str, Any] | None = None
    if complete_text == str(checkpoint_step):
        completion_protocol = "trainer_complete_exact_step_v1"
    elif complete_text == THIN_SIMPLE_COMPLETE_TEXT:
        completion_protocol = "verified_thin_simple_snapshot_v1"
        snapshot_marker = _validate_thin_snapshot_marker(
            run_dir=run_dir,
            checkpoint_step=checkpoint_step,
            model=model,
            argv=argv,
            run_config=run_config,
        )
    else:
        raise SimpleServerAuditError(
            "Checkpoint COMPLETE marker must contain the exact step or the verified "
            f"thin SIMPLE snapshot marker, got {complete_text!r}: {complete_path}"
        )

    run_binding = {
        "run_config": {"bytes": run_config["bytes"], "sha256": run_config["sha256"]},
        "argv": {"bytes": argv["bytes"], "sha256": argv["sha256"]},
    }
    checkpoint_binding = {
        "checkpoint_step": checkpoint_step,
        "completion_protocol": completion_protocol,
        "complete": {"bytes": complete["bytes"], "sha256": complete["sha256"]},
        "model": {"bytes": model["bytes"], "sha256": model["sha256"]},
        "snapshot_id": None if snapshot_marker is None else snapshot_marker["snapshot_id"],
    }
    return {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "run_dir": str(run_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_step": checkpoint_step,
        "completion_protocol": completion_protocol,
        "run_sha256": _json_sha256(run_binding),
        "checkpoint_sha256": _json_sha256(checkpoint_binding),
        "model_sha256": model["sha256"],
        "artifacts": {
            "model": model,
            "run_config": run_config,
            "argv": argv,
            "complete": complete,
            "snapshot_marker": snapshot_marker,
        },
    }


def _absolute_lexical_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise SimpleServerAuditError(f"{label} must be a non-empty absolute path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise SimpleServerAuditError(f"{label} must be an absolute path: {value!r}")
    return Path(os.path.abspath(path))


def inspect_checkpoint_manifest(
    manifest_path: Path,
    checkpoint_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify and normalize a preregistered candidate-checkpoint manifest."""

    manifest_path = Path(manifest_path).expanduser()
    manifest_file = file_identity(
        manifest_path,
        label="candidate checkpoint manifest",
    )
    try:
        raw_manifest = manifest_path.read_bytes()
        manifest = json.loads(raw_manifest)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SimpleServerAuditError(
            f"Cannot read candidate checkpoint manifest: {manifest_path}: {exc}"
        ) from exc
    if (
        len(raw_manifest) != manifest_file["bytes"]
        or hashlib.sha256(raw_manifest).hexdigest() != manifest_file["sha256"]
    ):
        raise SimpleServerAuditError(
            f"Candidate checkpoint manifest changed while it was read: {manifest_path}"
        )
    if not isinstance(manifest, dict):
        raise SimpleServerAuditError("Candidate checkpoint manifest must be a JSON object")
    missing = [key for key in CHECKPOINT_MANIFEST_KEYS if key not in manifest]
    if missing:
        raise SimpleServerAuditError(
            f"Candidate checkpoint manifest is missing required keys: {missing}"
        )

    checkpoint_id = manifest["checkpoint_id"]
    if (
        not isinstance(checkpoint_id, str)
        or SAFE_CHECKPOINT_ID_RE.fullmatch(checkpoint_id) is None
    ):
        raise SimpleServerAuditError(
            "checkpoint_id must match [A-Za-z0-9][A-Za-z0-9._/@:-]{0,511}"
        )
    step = manifest["step"]
    expected_step = checkpoint_identity.get("checkpoint_step")
    if isinstance(step, bool) or not isinstance(step, int) or step != expected_step:
        raise SimpleServerAuditError(
            f"Candidate checkpoint manifest step mismatch: {step!r} != {expected_step!r}"
        )

    model = checkpoint_identity["artifacts"]["model"]
    model_sha256 = _require_sha256(
        manifest["model_sha256"],
        label="candidate manifest model_sha256",
    )
    if model_sha256 != model["sha256"]:
        raise SimpleServerAuditError(
            "Candidate checkpoint manifest model_sha256 does not match the served model"
        )
    model_bytes = manifest["model_bytes"]
    if (
        isinstance(model_bytes, bool)
        or not isinstance(model_bytes, int)
        or model_bytes != model["bytes"]
    ):
        raise SimpleServerAuditError(
            "Candidate checkpoint manifest model_bytes does not match the served model"
        )

    source_run_input = _absolute_lexical_path(
        manifest["source_run"],
        label="candidate manifest source_run",
    )
    if source_run_input.is_symlink() or not source_run_input.is_dir():
        raise SimpleServerAuditError(
            f"Candidate manifest source_run must be a real directory: {source_run_input}"
        )
    source_run = source_run_input.resolve()
    model_path = _absolute_lexical_path(
        manifest["model_path"],
        label="candidate manifest model_path",
    )
    complete_value = manifest["COMPLETE"]
    if (
        isinstance(complete_value, bool)
        or not isinstance(complete_value, int)
        or complete_value != expected_step
    ):
        raise SimpleServerAuditError(
            "Candidate manifest COMPLETE must equal the exact integer checkpoint step"
        )
    expected_complete_path = model_path.parent / "COMPLETE"
    complete_marker_value = manifest.get(
        "complete_marker_path",
        str(expected_complete_path),
    )
    complete_path = _absolute_lexical_path(
        complete_marker_value,
        label="candidate manifest complete_marker_path",
    )
    allowed_checkpoint_dirs = {
        source_run / "checkpoints" / f"ckpt_{expected_step}",
        source_run / "model_snapshots" / f"ckpt_{expected_step}",
    }
    if model_path.parent not in allowed_checkpoint_dirs:
        raise SimpleServerAuditError(
            "Candidate manifest model_path is outside the canonical full-checkpoint "
            "and model-only snapshot layouts"
        )
    if model_path.name != "model.safetensors":
        raise SimpleServerAuditError("Candidate manifest model_path has the wrong filename")
    if complete_path != expected_complete_path:
        raise SimpleServerAuditError(
            "Candidate manifest complete_marker_path is not beside model.safetensors"
        )
    if not model_path.is_file():
        raise SimpleServerAuditError(
            f"Candidate manifest model_path is not a file: {model_path}"
        )
    if model_path.resolve() != Path(str(model["path"])).resolve():
        raise SimpleServerAuditError(
            "Candidate manifest model_path does not resolve to the served model"
        )
    if complete_path.is_symlink() or not complete_path.is_file():
        raise SimpleServerAuditError(
            f"Candidate manifest COMPLETE must be a regular nonsymlink file: {complete_path}"
        )
    try:
        complete_text = complete_path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise SimpleServerAuditError(
            f"Cannot read candidate manifest COMPLETE: {complete_path}: {exc}"
        ) from exc
    if complete_text != str(expected_step):
        raise SimpleServerAuditError(
            "Candidate manifest COMPLETE must contain the exact checkpoint step"
        )

    train_seed = manifest["train_seed"]
    if isinstance(train_seed, bool) or not isinstance(train_seed, int):
        raise SimpleServerAuditError("Candidate manifest train_seed must be an integer")
    run_config_path = source_run / "run_config.json"
    try:
        raw_config = run_config_path.read_bytes()
        run_config = json.loads(raw_config)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SimpleServerAuditError(
            f"Cannot read run_config for candidate train_seed: {run_config_path}: {exc}"
        ) from exc
    run_config_artifact = checkpoint_identity["artifacts"]["run_config"]
    if (
        len(raw_config) != run_config_artifact["bytes"]
        or hashlib.sha256(raw_config).hexdigest() != run_config_artifact["sha256"]
    ):
        raise SimpleServerAuditError(
            "run_config changed after checkpoint identity was established"
        )
    source_argv_path = source_run / "argv.txt"
    source_argv = file_identity(source_argv_path, label="candidate source launch argv")
    served_argv = checkpoint_identity["artifacts"]["argv"]
    if (
        source_argv["bytes"] != served_argv["bytes"]
        or source_argv["sha256"] != served_argv["sha256"]
    ):
        raise SimpleServerAuditError(
            "Candidate source argv.txt does not match the served run metadata"
        )
    if not isinstance(run_config, dict) or run_config.get("seed") != train_seed:
        raise SimpleServerAuditError(
            "Candidate manifest train_seed does not match run_config.json"
        )

    normalized_identity = {
        "checkpoint_id": checkpoint_id,
        "step": step,
        "model_sha256": model_sha256,
        "model_bytes": model_bytes,
        "source_run": str(source_run),
        "train_seed": train_seed,
        "COMPLETE": step,
        "model_path": str(model_path),
        "complete_marker_path": str(expected_complete_path),
    }
    return {
        "identity": normalized_identity,
        "artifact": manifest_file,
    }


def inspect_manifest_backed_checkpoint(
    run_dir: Path,
    checkpoint_step: int,
    manifest_path: Path,
) -> dict[str, Any]:
    """Establish identity directly from a model-only/full candidate manifest.

    This path is alias-free: the manifest's validated ``model_path`` is the
    exact file that the server must pass to ``Psi0Model.from_pretrained``.
    """

    run_dir = Path(run_dir).expanduser().resolve()
    manifest_path = Path(manifest_path).expanduser()
    manifest = _read_json_object(
        manifest_path,
        label="candidate checkpoint manifest",
    )
    missing = [key for key in CHECKPOINT_MANIFEST_KEYS if key not in manifest]
    if missing:
        raise SimpleServerAuditError(
            f"Candidate checkpoint manifest is missing required keys: {missing}"
        )
    if (
        isinstance(checkpoint_step, bool)
        or not isinstance(checkpoint_step, int)
        or manifest["step"] != checkpoint_step
    ):
        raise SimpleServerAuditError(
            "Candidate checkpoint manifest step does not match --ckpt-step"
        )
    source_run = _absolute_lexical_path(
        manifest["source_run"],
        label="candidate manifest source_run",
    ).resolve()
    if source_run != run_dir:
        raise SimpleServerAuditError(
            "--run-dir must resolve to candidate manifest source_run when loading "
            "an alias-free model-only snapshot"
        )
    model_path = _absolute_lexical_path(
        manifest["model_path"],
        label="candidate manifest model_path",
    )
    complete_value = manifest["COMPLETE"]
    if (
        isinstance(complete_value, bool)
        or not isinstance(complete_value, int)
        or complete_value != checkpoint_step
    ):
        raise SimpleServerAuditError(
            "Candidate checkpoint manifest COMPLETE must equal --ckpt-step"
        )
    complete_path = _absolute_lexical_path(
        manifest.get("complete_marker_path", str(model_path.parent / "COMPLETE")),
        label="candidate manifest complete_marker_path",
    )

    model = file_identity(
        model_path,
        label="manifest-backed checkpoint model",
        allow_symlink=True,
    )
    run_config = file_identity(run_dir / "run_config.json", label="run config")
    argv = file_identity(run_dir / "argv.txt", label="launch argv")
    complete = file_identity(
        complete_path,
        label="manifest-backed checkpoint COMPLETE marker",
    )
    try:
        complete_text = complete_path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise SimpleServerAuditError(
            f"Cannot read manifest-backed checkpoint COMPLETE: {complete_path}: {exc}"
        ) from exc
    if complete_text != str(checkpoint_step):
        raise SimpleServerAuditError(
            "Manifest-backed checkpoint COMPLETE must contain the exact step"
        )

    manifest_file = file_identity(
        manifest_path,
        label="candidate checkpoint manifest",
    )
    run_binding = {
        "run_config": {"bytes": run_config["bytes"], "sha256": run_config["sha256"]},
        "argv": {"bytes": argv["bytes"], "sha256": argv["sha256"]},
    }
    checkpoint_binding = {
        "checkpoint_step": checkpoint_step,
        "completion_protocol": "candidate_checkpoint_manifest_v1",
        "complete": {"bytes": complete["bytes"], "sha256": complete["sha256"]},
        "model": {"bytes": model["bytes"], "sha256": model["sha256"]},
        "manifest": {
            "bytes": manifest_file["bytes"],
            "sha256": manifest_file["sha256"],
        },
    }
    checkpoint_identity = {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "run_dir": str(run_dir),
        "checkpoint_dir": str(model_path.parent),
        "checkpoint_step": checkpoint_step,
        "completion_protocol": "candidate_checkpoint_manifest_v1",
        "run_sha256": _json_sha256(run_binding),
        "checkpoint_sha256": _json_sha256(checkpoint_binding),
        "model_sha256": model["sha256"],
        "artifacts": {
            "model": model,
            "run_config": run_config,
            "argv": argv,
            "complete": complete,
            "snapshot_marker": None,
            "checkpoint_manifest": manifest_file,
        },
    }
    manifest_audit = inspect_checkpoint_manifest(
        manifest_path,
        checkpoint_identity,
    )
    return {
        "checkpoint_identity": checkpoint_identity,
        "checkpoint_manifest_identity": manifest_audit["identity"],
        "checkpoint_manifest_artifact": manifest_audit["artifact"],
        "model_path": manifest_audit["identity"]["model_path"],
    }


def validate_inference_noise_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("inference_noise_seed must be an integer")
    if not 0 <= seed <= MAX_INFERENCE_NOISE_SEED:
        raise ValueError(
            f"inference_noise_seed must be in [0, {MAX_INFERENCE_NOISE_SEED}]"
        )
    return seed


def _tensor_bytes(value: torch.Tensor) -> bytes:
    return value.detach().to(device="cpu").contiguous().numpy().tobytes()


def capture_global_rng_state(*, include_cuda: bool | None = None) -> dict[str, Any]:
    """Hash all global RNG states without advancing them."""

    if include_cuda is None:
        include_cuda = torch.cuda.is_available()
    if include_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA RNG state was requested but CUDA is unavailable")

    source_digests: dict[str, Any] = {
        "python_sha256": hashlib.sha256(
            pickle.dumps(random.getstate(), protocol=4)
        ).hexdigest(),
        "numpy_sha256": hashlib.sha256(
            pickle.dumps(np.random.get_state(), protocol=4)
        ).hexdigest(),
        "torch_cpu_sha256": hashlib.sha256(_tensor_bytes(torch.get_rng_state())).hexdigest(),
        "torch_cuda_sha256": [],
    }
    if include_cuda:
        source_digests["torch_cuda_sha256"] = [
            hashlib.sha256(_tensor_bytes(state)).hexdigest()
            for state in torch.cuda.get_rng_state_all()
        ]
    return {
        "schema_version": RNG_AUDIT_SCHEMA_VERSION,
        **source_digests,
        "combined_sha256": _json_sha256(source_digests),
    }


def reset_inference_rng(
    seed: int,
    *,
    include_cuda: bool | None = None,
) -> dict[str, Any]:
    """Reset Python/NumPy/Torch RNGs and return before/after state digests."""

    seed = validate_inference_noise_seed(seed)
    if include_cuda is None:
        include_cuda = torch.cuda.is_available()
    before = capture_global_rng_state(include_cuda=include_cuda)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if include_cuda:
        torch.cuda.manual_seed_all(seed)
    after = capture_global_rng_state(include_cuda=include_cuda)
    return {
        "schema_version": RNG_AUDIT_SCHEMA_VERSION,
        "protocol": INFERENCE_NOISE_PROTOCOL,
        "seed": seed,
        "before_reset": before,
        "after_reset": after,
    }


def build_health_payload(
    *,
    policy: str,
    instance_id: str,
    checkpoint_identity: Mapping[str, Any],
    rtc_enabled: bool,
    rtc_max_delay: int | None,
    action_chunk_size: int,
    action_exec_horizon: int,
    inference_noise_seed: int,
    latest_rng_audit: Mapping[str, Any] | None,
    train_seed: int | None = None,
    checkpoint_manifest_identity: Mapping[str, Any] | None = None,
    checkpoint_manifest_artifact: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the stable JSON contract returned by ``GET /health``."""

    inference_noise_seed = validate_inference_noise_seed(inference_noise_seed)
    identity = dict(checkpoint_identity)
    manifest_identity = (
        None
        if checkpoint_manifest_identity is None
        else dict(checkpoint_manifest_identity)
    )
    manifest_artifact = (
        None
        if checkpoint_manifest_artifact is None
        else dict(checkpoint_manifest_artifact)
    )
    model_artifact = identity["artifacts"]["model"]
    complete_artifact = identity["artifacts"]["complete"]
    source_run = (
        identity["run_dir"]
        if manifest_identity is None
        else manifest_identity["source_run"]
    )
    model_path = (
        str(Path(str(identity["checkpoint_dir"])) / "model.safetensors")
        if manifest_identity is None
        else manifest_identity["model_path"]
    )
    complete_path = (
        str(Path(str(identity["checkpoint_dir"])) / "COMPLETE")
        if manifest_identity is None
        else manifest_identity["complete_marker_path"]
    )
    return {
        "status": "ok",
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "policy": policy,
        "instance_id": instance_id,
        "run_sha256": identity["run_sha256"],
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "model_sha256": identity["model_sha256"],
        "model_bytes": model_artifact["bytes"],
        "model_path": model_path,
        "source_run": source_run,
        "run_dir": identity["run_dir"],
        "checkpoint_step": identity["checkpoint_step"],
        "step": identity["checkpoint_step"],
        "COMPLETE": identity["checkpoint_step"],
        "complete_marker_path": complete_path,
        "complete_marker_sha256": complete_artifact["sha256"],
        "checkpoint_id": (
            None if manifest_identity is None else manifest_identity["checkpoint_id"]
        ),
        "train_seed": (
            train_seed
            if manifest_identity is None
            else manifest_identity["train_seed"]
        ),
        "manifest_sha256": (
            None if manifest_artifact is None else manifest_artifact["sha256"]
        ),
        "checkpoint_manifest_sha256": (
            None if manifest_artifact is None else manifest_artifact["sha256"]
        ),
        "rtc": rtc_enabled,
        "rtc_max_delay": rtc_max_delay,
        "action_chunk_size": action_chunk_size,
        "action_exec_horizon": action_exec_horizon,
        "inference_noise_seed": inference_noise_seed,
        "inference_noise_protocol": INFERENCE_NOISE_PROTOCOL,
        "identity": identity,
        "checkpoint_manifest_identity": manifest_identity,
        "checkpoint_manifest_artifact": manifest_artifact,
        "latest_rng_audit": None if latest_rng_audit is None else dict(latest_rng_audit),
    }

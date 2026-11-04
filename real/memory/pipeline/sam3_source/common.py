"""Small, dependency-light artifact helpers for the SAM-assisted FS experiment."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

DATA_REPO = "USC-PSI-Lab/psi-data"
SAM_REPO = "facebook/sam3"
SAM_REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"
SAM_CODE_REVISION = "2345a4ad109ac29c569da749c91d84f10dc08c40"
TASKS = {
    "bht": {"task": "G1WholebodyBendHandoverTeleop-v0", "prompts": ["cracker box"]},
    "bpmp": {"task": "G1WholebodyBendPickMP-v0", "prompts": ["red box"]},
    "hand": {"task": "G1WholebodyHandoverTeleop-v0", "prompts": ["cracker box"]},
    "ttg": {"task": "G1WholebodyTabletopGraspMP-v0", "prompts": ["red can"]},
    "chair": {"task": "G1WholebodyPushOfficeChairTeleop-v0", "prompts": ["chair backrest"]},
    "door": {"task": "G1WholebodyCloseDoorTeleop-v0", "prompts": ["door"]},
    "hug": {"task": "G1WholebodyPickAndPlaceAndHugContainerTeleop-v0", "prompts": ["basket"]},
    "oven": {"task": "G1WholebodyOpenOvenTeleop-v0", "prompts": ["oven door"]},
    "trash": {"task": "G1WholebodyOpenTrashCanTeleop-v0", "prompts": ["green lid"]},
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.pending")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def freeze_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Immutable artifact differs: {path}")
    else:
        write_json(path, value)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def identity(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def verify_identity(record):
    p = Path(record["path"])
    if not p.is_file() or p.stat().st_size != record["bytes"] or sha256(p) != record["sha256"]:
        raise ValueError(f"Bound input has changed: {p}")


def event(name, **values):
    print(json.dumps({"event": name, **values}, sort_keys=True, allow_nan=False), flush=True)

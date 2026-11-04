#!/usr/bin/env python3
"""Download checkpoint bundles from the Hugging Face Hub and verify their sha256.

    python scripts/download.py --task LitLamp --out checkpoints
    python scripts/download.py --suite simple --task all --out checkpoints
    python scripts/download.py --method psi0 --suite real --task TrashBag --out checkpoints

A bundle lands at <out>/<folder in the model repository>, e.g. checkpoints/wholebodymem/memory/LitLamp.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO_ID = "ChenyuanC/MemorybeforeGoal"
METHODS = ("wholebodymem", "psi0")
SUITES = ("memory", "simple", "real")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", choices=METHODS, default="wholebodymem")
    parser.add_argument("--suite", choices=SUITES, default="memory")
    parser.add_argument("--task", required=True, help="a task folder of the suite (e.g. LitLamp, T1_XMovePick, TrashBag), or 'all'")
    parser.add_argument("--out", type=Path, default=Path("checkpoints"))
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--revision", default=None, help="a commit of the model repository (default: main)")
    args = parser.parse_args()
    from huggingface_hub import HfApi, snapshot_download

    prefix = f"wholebodymem/{args.suite}" if args.method == "wholebodymem" else f"baselines/{args.method}/{args.suite}"
    files = HfApi().list_repo_files(args.repo_id, revision=args.revision)
    available = sorted({f.split("/")[prefix.count("/") + 1] for f in files
                        if f.startswith(prefix + "/") and f.endswith("/BUNDLE.json")})
    if not available:
        parser.error(f"no checkpoints under {prefix}/")
    tasks = available if args.task.lower() == "all" else [t for t in available if t.lower() == args.task.lower()]
    if not tasks:
        parser.error(f"unknown task {args.task!r} under {prefix}/; available: {', '.join(available)}")
    snapshot_download(args.repo_id, revision=args.revision, local_dir=args.out,
                      allow_patterns=[f"{prefix}/{task}/**" for task in tasks])
    failed = []
    for task in tasks:
        bundle = args.out / prefix / task
        meta = json.loads((bundle / "BUNDLE.json").read_text())
        ok = sha256_file(bundle / "model.safetensors") == meta["model_sha256"]
        print(f"{prefix}/{task}: {bundle} step {meta['step']} sha256 {'ok' if ok else 'MISMATCH'}")
        if not ok:
            failed.append(task)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

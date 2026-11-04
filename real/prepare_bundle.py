#!/usr/bin/env python3
"""Build the deployment bundle of one real G1 task from the released checkpoint and write its deploy config.

    python real/prepare_bundle.py --task TrashBag

1. Downloads wholebodymem/real/<Task>/ from the Hugging Face model repository (or uses --checkpoint-dir) and checks the
   model sha256 against BUNDLE.json.
2. Lays out the bundle the deployment server expects (real/bundles/<Task>/): run/checkpoints/ckpt_5000/model.safetensors,
   run/run_config.json, assets/qwen/ (the checkpoint's vlm/), normalization.json, memory/ (real/memory) and source/
   (real/source, the psi code the checkpoint was trained with), plus bundle.json (cart78 contract and per-file sha256) and
   BUNDLE_READY.json. The model file is hard-linked when possible, otherwise copied (the bundle may contain no symlinks).
3. Verifies the result with the server's own verify_bundle and writes real/deploy/config/<Task>/{deploy,controller}.json.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

REAL = Path(__file__).resolve().parent
sys.path.insert(0, str(REAL / "deploy/server"))
from bundle import (FORMAT, SOURCE_COMMIT, canonical_bytes, json_sha, normalization_from_config, read_header,  # noqa: E402
                    sha256, verify_bundle, write_json)


def copy_tree(src: Path, dst: Path) -> None:
    for p in sorted(src.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
            out = dst / p.relative_to(src)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, out)


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> int:
    tasks = json.loads((REAL / "tasks.json").read_text())
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", required=True, choices=sorted(tasks["tasks"]))
    ap.add_argument("--checkpoint-dir", type=Path, help="an already downloaded wholebodymem/real/<Task> folder")
    ap.add_argument("--download-dir", type=Path, default=REAL / "downloads")
    ap.add_argument("--bundles", type=Path, default=REAL / "bundles")
    ap.add_argument("--revision", default=None, help="a commit of the model repository (default: main)")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    task = tasks["tasks"][a.task]
    step = int(task["step"])

    src = a.checkpoint_dir
    if src is None:
        from huggingface_hub import snapshot_download
        snapshot_download(tasks["model_repository"], revision=a.revision, local_dir=a.download_dir,
                          allow_patterns=[task["checkpoint"] + "/**"])
        src = a.download_dir / task["checkpoint"]
    meta = json.loads((src / "BUNDLE.json").read_text())
    if sha256(src / "model.safetensors") != meta["model_sha256"]:
        raise SystemExit(f"{src}/model.safetensors does not match the sha256 in BUNDLE.json")

    out = (a.bundles / a.task).resolve()
    if out.exists():
        raise SystemExit(f"{out} exists; remove it to rebuild")
    stage = out.with_name(out.name + ".incoming")
    if stage.exists():
        shutil.rmtree(stage)
    model_rel = f"run/checkpoints/ckpt_{step}/model.safetensors"
    link_or_copy(src / "model.safetensors", stage / model_rel)
    (stage / "run").mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "run_config.json", stage / "run/run_config.json")
    (stage / "run/argv.txt").write_text("training command line not included in the public release\n")
    copy_tree(src / "vlm", stage / "assets/qwen")
    cfg = json.loads((stage / "run/run_config.json").read_text())
    (stage / "normalization.json").write_bytes(canonical_bytes(normalization_from_config(cfg)))
    copy_tree(REAL / "memory" / a.task, stage / "memory")
    copy_tree(REAL / "memory/pipeline", stage / "memory/pipeline")
    copy_tree(REAL / "source", stage / "source")

    files = {}
    for p in sorted(stage.rglob("*")):
        if p.is_file():
            files[p.relative_to(stage).as_posix()] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    info = read_header(stage / model_rel)
    contract = dict(tasks["contract"], stats_sha256=files["normalization.json"]["sha256"],
                    memory_spec_sha256=files["memory/spec.json"]["sha256"])
    manifest = {
        "format": FORMAT, "source_commit": SOURCE_COMMIT,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "checkpoint": {"step": step, "sha256": files[model_rel]["sha256"], "bytes": files[model_rel]["bytes"],
                       "tensor_count": info["tensor_count"], "memory_tensors": info["memory_tensors"]},
        "contract": contract, "contract_sha256": json_sha(contract), "files": files,
        "notes": {"built_by": "real/prepare_bundle.py", "checkpoint": f"{tasks['model_repository']}/{task['checkpoint']}"},
    }
    write_json(stage / "bundle.json", manifest)
    write_json(stage / "BUNDLE_READY.json", {"bundle_sha256": sha256(stage / "bundle.json"),
                                             "checkpoint_sha256": manifest["checkpoint"]["sha256"]})
    verify_bundle(stage, meta["model_sha256"])
    stage.rename(out)
    print(f"bundle: {out} (checkpoint {manifest['checkpoint']['sha256'][:12]}, contract {manifest['contract_sha256'][:12]})")

    conf = REAL / "deploy/config" / a.task
    conf.mkdir(parents=True, exist_ok=True)
    if not (conf / "deploy.json").exists():
        d = json.loads((REAL / "deploy/config/deploy.example.json").read_text())
        d["bundle"] = str(out)
        d["server"]["port"] = a.port
        d["client"]["server_url"] = f"ws://127.0.0.1:{a.port}/ws"
        d["client"]["instruction"] = task["instruction"]
        d["client"]["log_dir"] = f"${{PACKAGE}}/logs/{a.task}/client"
        d["runtime_base"] = f"${{PACKAGE}}/runtime/{a.task}"
        d["controller"]["config"] = str(conf / "controller.json")
        (conf / "deploy.json").write_text(json.dumps(d, indent=2) + "\n")
        print(f"config: {conf / 'deploy.json'}")
    if not (conf / "controller.json").exists():
        shutil.copy2(REAL / "deploy/config/controller.example.json", conf / "controller.json")
        print(f"config: {conf / 'controller.json'} (review the controller paths before use)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

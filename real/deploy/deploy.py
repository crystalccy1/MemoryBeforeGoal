#!/usr/bin/env python3
"""Portable front door. No subcommand implicitly enables robot execution."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PATH_FLAGS = {"--bundle", "--run-dir", "--vlm-assets", "--psi-repo", "--output", "--dataset",
              "--bundle-manifest", "--log-dir", "--config", "--source", "--destination", "--tensorrt-root"}


def absolute_paths(argv):
    result = list(argv)
    for i, arg in enumerate(result):
        key, sep, value = arg.partition("=")
        if key in PATH_FLAGS:
            if sep:
                result[i] = key + "=" + os.path.abspath(os.path.expanduser(value))
            elif i + 1 < len(result):
                result[i + 1] = os.path.abspath(os.path.expanduser(result[i + 1]))
    return result


def expanded(value):
    if isinstance(value, str):
        return os.path.expanduser(value.replace("${PACKAGE}", str(ROOT)))
    if isinstance(value, list):
        return [expanded(x) for x in value]
    if isinstance(value, dict):
        return {k: expanded(v) for k, v in value.items()}
    return value


def get_config(path):
    if not path.is_file():
        raise SystemExit(f"Missing {path}. Run ./deploy.sh init, then review that file.")
    cfg = expanded(json.loads(path.read_text()))
    if cfg.get("schema_version") != 1:
        raise SystemExit("Unsupported deployment config schema")
    return cfg


def resolve_python(value):
    p = Path(value).expanduser()
    if "/" not in str(p):
        found = shutil.which(str(p))
        if found:
            return Path(found).absolute()
    if p.is_file() and os.access(p, os.X_OK):
        return p.absolute()
    raise SystemExit(f"Python is not executable: {value}. Run setup_env.sh or set CART78_SERVER_PYTHON/CART78_CLIENT_PYTHON.")


def run_private(target, argv, python, runtime_base):
    """Replace this launcher with an opaque, private-config Python process."""
    os.umask(0o077)
    base = Path(runtime_base).expanduser()
    base.mkdir(parents=True, exist_ok=True)
    run = base / ("r-" + secrets.token_hex(4))
    run.mkdir(mode=0o700)
    target = target.resolve()
    if not target.is_file():
        raise SystemExit(f"Missing entry point: {target}")
    py = resolve_python(python)
    # Preserve a venv's pyvenv.cfg through a directory alias rather than copying
    # its interpreter or changing HOME/PYTHONHOME.
    if py.parent.name == "bin" and (py.parent.parent / "pyvenv.cfg").is_file():
        (run / "e").symlink_to(py.parent.parent, target_is_directory=True)
        command_python = run / "e/bin" / py.name
    else:
        (run / "p").symlink_to(py)
        command_python = run / "p"
    (run / "c.json").write_text(json.dumps({"target": str(target), "argv": absolute_paths(argv)}, indent=2))
    (run / "r.py").write_text(
        "import json,os,runpy,sys\nfrom pathlib import Path\n"
        "r=Path(__file__).resolve().parent\nos.umask(0o077)\n"
        "c=json.loads((r/'c.json').read_text())\n"
        "os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')\n"
        "os.environ.setdefault('WANDB_MODE','disabled')\n"
        "os.environ['CART78_RUNTIME_DIR']=str(r)\n"
        "sys.path.insert(0,str(Path(c['target']).parent))\n"
        "sys.path.insert(0,str(Path(c['target']).parent.parent))\n"
        "sys.argv=[c['target'],*c['argv']]\n"
        "os.chdir(r)\nrunpy.run_path(c['target'],run_name='__main__')\n"
    )
    print(f"Private runtime: {run}", flush=True)
    os.execv(str(command_python), [str(command_python), str(run / "r.py")])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/deploy.json")
    parser.add_argument("command", choices=["init", "verify", "server", "client",
                                           "replay", "mock", "controller-prepare", "controller-build",
                                           "controller", "test"])
    args, extra = parser.parse_known_args()
    if args.command == "init":
        init_parser = argparse.ArgumentParser(prog="deploy.sh init")
        init_parser.add_argument("--bundle", type=Path)
        init_args = init_parser.parse_args(extra)
        if args.config.exists():
            raise SystemExit(f"Refusing to overwrite {args.config}")
        args.config.parent.mkdir(parents=True, exist_ok=True)
        cfg = json.loads((ROOT / "config/deploy.example.json").read_text())
        cfg["controller"]["config"] = str((args.config.parent / "controller.json").absolute())
        if init_args.bundle:
            cfg["bundle"] = str(init_args.bundle.expanduser().absolute())
        ctl = args.config.parent / "controller.json"
        # Check both files before publishing either new configuration.
        ctl_cfg = expanded(json.loads((ROOT / "config/controller.example.json").read_text()))
        if ctl.exists():
            existing = json.loads(ctl.read_text())
            existing = existing.get("controller", existing)
            if Path(existing.get("source_root", "")).expanduser().absolute() != Path(ctl_cfg["source_root"]).absolute():
                raise SystemExit(f"Existing {ctl} targets a different controller source; use a new config directory or reconcile it first")
        else:
            ctl.write_text(json.dumps(ctl_cfg, indent=2) + "\n")
        args.config.write_text(json.dumps(cfg, indent=2) + "\n")
        print(f"Created {args.config} and {ctl}. Review machine-specific paths before starting hardware.")
        return
    cfg = get_config(args.config)
    py_server = os.environ.get("CART78_SERVER_PYTHON", cfg["python"]["server"])
    py_client = os.environ.get("CART78_CLIENT_PYTHON", cfg["python"]["client"])
    run_base = os.environ.get("CART78_RUN_BASE", cfg["runtime_base"])
    bundle_path = cfg["bundle"]
    command = args.command
    if command == "verify":
        run_private(ROOT / "server/serve.py", ["--bundle", bundle_path, "--verify-only"] + extra, py_server, run_base)
    elif command == "server":
        c = cfg["server"]
        argv = ["--bundle", bundle_path, "--host", c["host"], "--port", str(c["port"]),
                "--device", c["device"], "--inference-steps", str(c["inference_steps"]),
                "--attention-backend", c["attention_backend"]] + extra
        if "--mock" in extra:
            argv = ["--host", c["host"], "--port", str(c["port"]), *extra]
        run_private(ROOT / "server/serve.py", argv, py_server, run_base)
    elif command in {"client", "replay", "mock"}:
        c = cfg["client"]; mode = "live" if command == "client" else command
        argv = [mode, "--server-url", c["server_url"], "--bundle-manifest", str(Path(bundle_path) / "bundle.json"),
                "--instruction", c["instruction"], "--log-dir", c["log_dir"]]
        if mode == "live":
            argv += ["--state-endpoint", c["state_endpoint"], "--camera-endpoint", c["camera_endpoint"],
                     "--pose-bind", c["pose_bind"]]
        if mode == "mock":
            argv = [mode, "--server-url", c["server_url"], "--expected-checkpoint-sha", "1"*64, "--expected-stats-sha", "2"*64, "--log-dir", c["log_dir"]]
        argv += extra
        run_private(ROOT / "client/run_client.py", argv, py_client, run_base)
    elif command == "controller-prepare":
        c = cfg["controller"]
        run_private(ROOT / "controller/prepare.py", ["--source", c["original_sonic_root"],
                    "--destination", c["isolated_source"]] + extra, sys.executable, run_base)
    elif command == "controller-build":
        c = cfg["controller"]
        run_private(ROOT / "controller/build.py", ["--source", c["isolated_source"],
                    "--tensorrt-root", c["tensorrt_root"], "--jobs", "2"] + extra, sys.executable, run_base)
    elif command == "controller":
        run_private(ROOT / "controller/run.py", ["--config", cfg["controller"]["config"]] + extra,
                    sys.executable, run_base)
    elif command == "test":
        argv = [str(ROOT / "tests/run_all.py")] + extra
        raise SystemExit(subprocess.call([str(resolve_python(py_client)), *argv], cwd=ROOT))


if __name__ == "__main__":
    main()

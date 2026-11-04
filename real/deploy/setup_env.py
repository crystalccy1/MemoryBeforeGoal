#!/usr/bin/env python3
"""Create a dedicated environment; never modify an existing unmarked environment."""
from __future__ import annotations
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def call(args, **kwargs):
    print("Running:", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), check=True, **kwargs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("role", choices=["server", "client"])
    p.add_argument("--venv", type=Path)
    p.add_argument("--python", help="Path to a Python 3.11 interpreter; otherwise use python3.11 or uv")
    p.add_argument("--with-tests", action="store_true", help="Install CPU test dependencies in this dedicated environment")
    p.add_argument("--cuda-index", choices=["cu128", "cu126"], default="cu128",
                   help="cu128 is required for RTX5090; cu126 is only for supported older GPUs")
    a = p.parse_args()
    if a.role == "server" and sys.platform != "linux":
        raise SystemExit("GPU server setup targets Linux; use the client environment for CPU tests on other systems")
    os.umask(0o077)
    env_dir = (a.venv or ROOT / (".venv-" + a.role)).expanduser().absolute()
    marker = env_dir / ".cart78-managed-env.json"
    if env_dir.exists():
        if not marker.is_file():
            raise SystemExit(f"Refusing to modify unmanaged environment: {env_dir}. Use a new --venv path.")
        old = json.loads(marker.read_text())
        if old.get("schema") != 1 or old.get("role") != a.role:
            raise SystemExit("Existing environment marker does not match this role")
    else:
        interpreter = a.python or shutil.which("python3.11")
        if interpreter:
            version = subprocess.check_output([interpreter, "-c", "import sys;print('%d.%d'%sys.version_info[:2])"], text=True).strip()
            if version != "3.11":
                raise SystemExit(f"Expected Python3.11, got {version}: {interpreter}")
            call([interpreter, "-m", "venv", env_dir])
        elif shutil.which("uv"):
            isolated = os.environ.copy()
            isolated["UV_PYTHON_INSTALL_DIR"] = str(ROOT / ".python")
            isolated["UV_CACHE_DIR"] = str(ROOT / ".cache/uv")
            call([shutil.which("uv"), "venv", "--seed", "--python", "3.11", env_dir], env=isolated)
        else:
            raise SystemExit("Python3.11 is required. Install Python3.11 or uv, then retry (or pass --python /path/python3.11).")
        marker.write_text(json.dumps({"schema": 1, "role": a.role, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}, indent=2))
    py = env_dir / "bin/python"
    if a.role == "server":
        call([py, "-m", "pip", "install", "torch==2.7.0", "torchvision==0.22.0",
              "--index-url", "https://download.pytorch.org/whl/" + a.cuda_index])
        call([py, "-m", "pip", "install", "-r", ROOT / "server/requirements-inference.txt"])
        call([py, "-m", "pip", "install", "-r", ROOT / "server/requirements-sam3.txt"])
    else:
        call([py, "-m", "pip", "install", "-r", ROOT / "requirements-client.txt"])
    if a.with_tests:
        call([py, "-m", "pip", "install", "-r", ROOT / "requirements-test.txt"])
    call([py, "-m", "pip", "check"])
    print(f"Ready: {py}")
    if a.role == "server":
        print("Next: ./deploy.sh server --import-check (after exporting a bundle). SDPA is the default; FlashAttention is not installed.")


if __name__ == "__main__":
    main()

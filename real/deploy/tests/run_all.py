#!/usr/bin/env python3
"""Run nonempty CPU/localhost tests. Never runs a hardware controller or CUDA model."""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--list", action="store_true")
    p.add_argument("--replay-dataset", type=Path)
    p.add_argument("--output", type=Path, default=ROOT / "evidence/cpu_suite.json")
    a = p.parse_args()
    plan = [
        ("server_contract_and_websocket", [str(ROOT / "server/test_server.py")]),
        ("client_adapters_and_contract", [str(ROOT / "client/test_client.py")]),
        ("client_localhost_fault_and_lease", [str(ROOT / "client/test_integration.py")]),
        ("actual_package_server_client", [str(ROOT / "client/test_package_server.py")]),
        ("root_launcher", [str(ROOT / "tests/test_launcher.py")]),
    ]
    if a.replay_dataset:
        plan[2][1].extend(["--replay-dataset", str(a.replay_dataset.resolve())])
    if a.list:
        print(json.dumps({"scope": "CPU/localhost only; no robot or CUDA", "python_suites": plan,
                          "cpp_suite": str(ROOT / "controller/tests/test_token_lease.cpp")}, indent=2))
        return
    for _, args in plan:
        if not Path(args[0]).is_file():
            raise SystemExit(f"Missing mandatory test: {args[0]}")
    compiler = shutil.which("c++") or shutil.which("g++")
    if not compiler:
        raise SystemExit("C++17 compiler required for the controller lease test")
    report = {"started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "status": "running", "robot_access": False, "cuda_used": False, "tests": []}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    logs = a.output.parent / "cpu_test_logs"
    logs.mkdir(exist_ok=True)
    try:
        for name, args in plan:
            print(f"Checking {name}", flush=True)
            start = time.monotonic()
            proc = subprocess.Popen([sys.executable, *args], cwd=ROOT, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, start_new_session=True)
            try:
                output, _ = proc.communicate(timeout=150)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    output, _ = proc.communicate(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    output, _ = proc.communicate()
                raise RuntimeError(f"Test timed out: {name}")
            (logs / (name + ".log")).write_text(output)
            report["tests"].append({"name": name, "exit": proc.returncode,
                                    "seconds": time.monotonic() - start})
            if proc.returncode:
                raise RuntimeError(f"{name} failed: {output[-2500:]}")
        with tempfile.TemporaryDirectory(prefix="cart78-guard-") as td:
            binary = Path(td) / "check"
            source = ROOT / "controller/tests/test_token_lease.cpp"
            command = [compiler, "-std=c++17", "-O2", "-fno-finite-math-only", "-pthread",
                       "-I", str(ROOT / "controller/include"), str(source), "-o", str(binary)]
            build = subprocess.run(command, capture_output=True, text=True, timeout=60)
            if build.returncode:
                raise RuntimeError("Controller guard compile failed: " + build.stderr[-2000:])
            check = subprocess.run([str(binary)], capture_output=True, text=True, timeout=15)
            (logs / "controller_guard.log").write_text(check.stdout + check.stderr)
            report["tests"].append({"name": "controller_guard_cpp", "exit": check.returncode})
            if check.returncode:
                raise RuntimeError("Controller guard test failed")
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        raise
    finally:
        report["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Actual deploy.sh integration on localhost; never accesses robot endpoints."""
from __future__ import annotations

import hashlib
import ast
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
# The client environment (a venv made by setup_env.py client --with-tests); run_all.py runs this file with it.
PYTHON = Path(os.environ.get("CART78_TEST_PYTHON", sys.executable)).absolute()
FRONT = ROOT / "deploy.sh"


def invoke(argv, cwd, timeout=15, env=None):
    effective = {**os.environ, "CART78_LAUNCH_PYTHON": str(PYTHON), **(env or {})}
    return subprocess.run([str(FRONT), *map(str, argv)], cwd=cwd, env=effective,
                          capture_output=True, text=True, timeout=timeout)


def config_example():
    value = json.loads((ROOT / "config/deploy.example.json").read_text())
    value["python"] = {"server": str(PYTHON), "client": str(PYTHON)}
    return value


def expanded_path(value):
    return Path(os.path.expanduser(value.replace("${PACKAGE}", str(ROOT))))


def test_declared_client_dependencies():
    requirements = (ROOT / "requirements-client.txt").read_text()
    assert "websocket-client" in requirements, "setup_env client requirements lack import websocket's distribution websocket-client"
    result = subprocess.run([str(PYTHON), "-c", "import websocket,zmq,msgpack;print(websocket.__version__)"],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return {"websocket_client_version": result.stdout.strip()}


def test_init_paths_and_preservation():
    with tempfile.TemporaryDirectory(prefix="cart78-init-") as td:
        work = Path(td)
        config = work / "configs/deploy.json"
        result = invoke(["--config", "configs/deploy.json", "init", "--bundle", "bundle with spaces"], work)
        assert result.returncode == 0, result.stdout + result.stderr
        cfg = json.loads(config.read_text())
        controller = work / "configs/controller.json"
        ctl = json.loads(controller.read_text())
        assert expanded_path(cfg["controller"]["config"]).resolve() == controller.resolve(), (
            "init generated a sibling controller.json but deploy routes a different config", cfg["controller"]["config"])
        source = ctl["source_root"]
        assert "${PACKAGE}" not in source and Path(source).is_absolute(), source
        for key in ["decoder", "encoder", "observation_config", "reference_dir", "planner"]:
            assert ctl[key].startswith(source + "/"), (key, ctl[key])
        assert Path(cfg["bundle"]).resolve() == (work / "bundle with spaces").resolve(), (cfg["bundle"], str(work))
        original = config.read_bytes()
        again = invoke(["--config", config, "init"], work)
        assert again.returncode != 0 and config.read_bytes() == original
        return {"controller_paths_consistent": True, "relative_bundle_resolved": True,
                "existing_deploy_config_preserved": True}


def test_init_existing_controller_mismatch_rejected():
    with tempfile.TemporaryDirectory(prefix="cart78-init-existing-") as td:
        work = Path(td)
        config = work / "deploy.json"
        controller = work / "controller.json"
        before = b'{"source_root":"/keep/my/existing/controller","custom":true}\n'
        controller.write_bytes(before)
        result = invoke(["--config", config, "init"], work)
        assert result.returncode != 0, "init silently combines the example source with a different existing controller.json"
        assert controller.read_bytes() == before and not config.exists()
        return {"mismatched_existing_controller_refused": True, "existing_file_preserved": True}


def test_setup_refuses_unmanaged_environment():
    with tempfile.TemporaryDirectory(prefix="cart78-env-") as td:
        work = Path(td)
        env = work / "existing env"
        env.mkdir()
        sentinel = env / "preserve.txt"
        sentinel.write_text("must remain unchanged")
        result = subprocess.run([str(PYTHON), str(ROOT / "setup_env.py"), "client", "--venv", str(env)],
                                capture_output=True, text=True, timeout=15)
        assert result.returncode != 0 and "unmanaged" in result.stdout + result.stderr
        assert sentinel.read_text() == "must remain unchanged"
        assert not (env / ".cart78-managed-env.json").exists()
        marker = env / ".cart78-managed-env.json"
        marker.write_text(json.dumps({"schema": 1, "role": "server"}))
        result2 = subprocess.run([str(PYTHON), str(ROOT / "setup_env.py"), "client", "--venv", str(env)],
                                 capture_output=True, text=True, timeout=15)
        assert result2.returncode != 0 and "role" in result2.stdout + result2.stderr
        return {"unmanaged_and_wrong_role_envs_preserved": True, "packages_installed_by_test": False}


def test_missing_test_coordinator_is_failure():
    # Isolated copy exercises the real front-door branch without recursively
    # invoking the package's live run_all.py from this test itself.
    with tempfile.TemporaryDirectory(prefix="cart78-no-tests-") as td:
        work = Path(td)
        shutil.copy2(ROOT / "deploy.py", work / "deploy.py")
        config = config_example()
        config["runtime_base"] = str(work / "runtime")
        (work / "config.json").write_text(json.dumps(config))
        result = subprocess.run([str(PYTHON), str(work / "deploy.py"), "--config", str(work / "config.json"), "test"],
                                cwd=work, capture_output=True, text=True, timeout=15)
        assert result.returncode != 0, "missing test runner incorrectly returned success"
        return {"missing_runner_returncode": result.returncode, "no_recursive_full_test_call": True}


def test_root_test_lists_nonempty_real_suites():
    with tempfile.TemporaryDirectory(prefix="cart78-test-plan-") as td:
        work = Path(td)
        config = work / "deploy.json"
        config.write_text(json.dumps(config_example()))
        result = invoke(["--config", config, "test", "--list"], work, timeout=20)
        assert result.returncode == 0, result.stdout + result.stderr
        plan = json.loads(result.stdout)
        suites = plan.get("python_suites", [])
        assert suites, "root test coordinator exposed an empty Python test plan"
        names = [name for name, _ in suites]
        assert len(names) == len(set(names))
        checks = []
        for name, command in suites:
            source = Path(command[0])
            assert source.is_file() and source.stat().st_size > 100, (name, source)
            tree = ast.parse(source.read_text())
            assertion_sites = sum(isinstance(n, ast.Assert) or
                (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr.startswith("assert"))
                for n in ast.walk(tree))
            assert assertion_sites > 0, f"{name} is an empty/no-assertion test script"
            checks.append({"suite": name, "assertion_sites": assertion_sites})
        cpp = Path(plan["cpp_suite"])
        assert cpp.is_file() and cpp.stat().st_size > 100
        return {"python_suite_count": len(suites), "python_sources_have_assertions": checks,
                "cpp_test_present": True, "route_used": "actual deploy.sh test --list; no recursive test execution"}


def test_real_private_launcher_mock_flow():
    with tempfile.TemporaryDirectory(prefix="cart78-launcher-") as td:
        work = Path(td)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]
        cfg = config_example()
        cfg["runtime_base"] = "private runs"
        cfg["bundle"] = "missing bundle is unused by mock"
        cfg["server"]["host"], cfg["server"]["port"] = "127.0.0.1", port
        cfg["client"]["server_url"] = f"ws://127.0.0.1:{port}/ws"
        cfg["client"]["log_dir"] = "logs from config"
        # These values must never be used by the mock entry point.
        cfg["client"]["state_endpoint"] = "invalid://do-not-connect"
        cfg["client"]["camera_endpoint"] = "invalid://do-not-connect"
        config = work / "deploy.json"
        config.write_text(json.dumps(cfg))
        effective = {**os.environ, "CART78_LAUNCH_PYTHON": str(PYTHON)}
        with (work / "server.log").open("w") as output:
            server = subprocess.Popen([str(FRONT), "--config", "deploy.json", "server", "--mock"],
                                      cwd=work, env=effective, stdout=output, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 35
                while True:
                    if server.poll() is not None:
                        raise AssertionError((work / "server.log").read_text())
                    try:
                        info = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/deployment-info", timeout=.3))
                        break
                    except Exception:
                        if time.monotonic() >= deadline:
                            raise AssertionError("root server did not start: " + (work / "server.log").read_text())
                        time.sleep(.05)
                assert info["mock"] is True and info["dry_run"] is True
                client = invoke(["--config", "deploy.json", "mock", "--duration-seconds", "2", "--log-dir", "logs from CLI"], work, timeout=40)
                assert client.returncode == 0, client.stdout + client.stderr
                summaries = list((work / "logs from CLI").glob("*/summary.json"))
                assert len(summaries) == 1
                summary = json.loads(summaries[0].read_text())
                assert summary["actions_recorded"] >= 30 and summary["pose_packets_published"] == 0
                assert summary["execute"] is False and summary["source"] == "mock"
                runtimes = sorted((work / "private runs").glob("r-*"))
                assert len(runtimes) == 2
                runtime_records = []
                for run in runtimes:
                    assert stat.S_IMODE(run.stat().st_mode) == 0o700
                    for name in ["c.json", "r.py"]:
                        assert stat.S_IMODE((run / name).stat().st_mode) & 0o077 == 0
                    assert (run / "e").is_symlink()
                    assert (run / "e").resolve() == PYTHON.parent.parent.resolve()
                    payload = json.loads((run / "c.json").read_text())
                    assert "--execute" not in payload["argv"]
                    if payload["target"].endswith("client/run_client.py"):
                        assert payload["argv"][0] == "mock"
                        log_values = [payload["argv"][i + 1] for i, value in enumerate(payload["argv"]) if value == "--log-dir"]
                        assert Path(log_values[-1]).resolve() == (work / "logs from CLI").resolve(), log_values
                        assert all(Path(value).is_absolute() for value in log_values)
                        assert "--pose-bind" not in payload["argv"]
                    probe = subprocess.run([str(run / "e/bin/python"), "-c",
                        "import json,sys,websocket;print(json.dumps({'prefix':sys.prefix,'base_prefix':sys.base_prefix,'websocket':websocket.__version__}))"],
                        cwd=work, capture_output=True, text=True, timeout=30)
                    assert probe.returncode == 0, probe.stderr
                    probe_data = json.loads(probe.stdout)
                    assert Path(probe_data["prefix"]).resolve() == PYTHON.parent.parent.resolve()
                    runtime_records.append({"target": Path(payload["target"]).name, "mode": "0700", "private_files": True,
                                            "venv_alias_preserved": True, "venv_import_websocket": probe_data["websocket"]})
                return {"root_deploy_sh_used": True, "port": port, "actions_recorded": summary["actions_recorded"],
                        "pose_packets_published": 0, "default_execute": False,
                        "relative_cli_paths_preserved_after_private_chdir": True, "runtime_records": runtime_records}
            finally:
                server.terminate()
                try: server.wait(timeout=4)
                except subprocess.TimeoutExpired: server.kill(); server.wait()


def main():
    if not (PYTHON.parent.parent / "pyvenv.cfg").is_file():
        raise SystemExit(f"Run this with the client venv (setup_env.py client --with-tests); {PYTHON} is not in a venv")
    tests = [test_declared_client_dependencies, test_init_paths_and_preservation,
             test_init_existing_controller_mismatch_rejected, test_setup_refuses_unmanaged_environment,
             test_missing_test_coordinator_is_failure, test_root_test_lists_nonempty_real_suites,
             test_real_private_launcher_mock_flow]
    assert tests, "Empty launcher test plan is forbidden"
    results = []
    for test in tests:
        start = time.monotonic()
        try:
            evidence = test()
            result = {"test": test.__name__, "passed": True, "evidence": evidence}
        except Exception as exc:
            result = {"test": test.__name__, "passed": False, "error": str(exc), "traceback": traceback.format_exc()}
        result["duration_seconds"] = time.monotonic() - start
        results.append(result)
        print(json.dumps(result), flush=True)
    report = {"status": "passed" if all(x["passed"] for x in results) else "failed",
              "executed_count": len(results), "passed_count": sum(x["passed"] for x in results),
              "scope": "actual root deploy.sh frontend, localhost mock server/client, temporary configs only; no GPU/robot",
              "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                 for name in ["deploy.py", "deploy.sh", "setup_env.py", "requirements-client.txt"]},
              "results": results}
    destination = ROOT / "evidence/launcher_test.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["status"] == "passed" and report["executed_count"] > 0 else 1)


if __name__ == "__main__":
    main()

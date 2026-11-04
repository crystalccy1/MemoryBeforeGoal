"""Localhost HTTP+WebSocket+ZMQ integration; never contacts a real endpoint."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parent


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_server(port, process):
    for _ in range(600):
        if process.poll() is not None:
            raise RuntimeError("local fake server exited")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/deployment-info", timeout=.2) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(.05)
    raise RuntimeError("local fake server not ready")


def one(name, *, source="mock", behavior="normal", execute=False, dataset=None):
    with tempfile.TemporaryDirectory(prefix="client-it-", dir=ROOT) as td:
        tmp = Path(td)
        ports = [free_port() for _ in range(4)]
        while len(set(ports)) != 4:
            ports = [free_port() for _ in range(4)]
        server_log = (tmp / "server.log").open("w")
        server = subprocess.Popen([sys.executable, str(ROOT / "tests/fake_server.py"), "--port", str(ports[0]),
                                   "--behavior", behavior] + (["--execute-fixture"] if execute else []),
                                  stdout=server_log, stderr=subprocess.STDOUT)
        robot = None
        try:
            wait_server(ports[0], server)
            if execute:
                robot = subprocess.Popen([sys.executable, str(ROOT / "tests/fake_robot.py"),
                    "--state-port", str(ports[1]), "--camera-port", str(ports[2]), "--pose-port", str(ports[3]),
                    "--output", str(tmp / "poses.jsonl")], stdout=server_log, stderr=subprocess.STDOUT)
            command = [sys.executable, str(ROOT / "run_client.py"), source,
                       "--server-url", f"ws://127.0.0.1:{ports[0]}/ws",
                       "--expected-checkpoint-sha", "a" * 64, "--expected-stats-sha", "b" * 64,
                       "--expected-contract-sha", "c" * 64, "--duration-seconds", "2.0",
                       "--log-dir", str(tmp / "client")]
            if execute:
                command += ["--execute", "--state-endpoint", f"tcp://127.0.0.1:{ports[1]}",
                            "--camera-endpoint", f"tcp://127.0.0.1:{ports[2]}",
                            "--pose-bind", f"tcp://127.0.0.1:{ports[3]}"]
            if source == "replay":
                command += ["--dataset", str(dataset), "--episode", "2"]
            if behavior == "stale":
                command += ["--action-stream-timeout-ms", "400"]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=45)
            time.sleep(.1)
            summaries = list((tmp / "client").glob("*/summary.json"))
            if not summaries:
                raise AssertionError(completed.stdout + completed.stderr + (tmp / "server.log").read_text())
            summary = json.loads(summaries[0].read_text())
            events = [json.loads(l) for l in summaries[0].with_name("events.jsonl").read_text().splitlines()]
            poses_path = tmp / "poses.jsonl"
            poses = [json.loads(l) for l in poses_path.read_text().splitlines()] if poses_path.exists() else []
            enabled = [x for x in poses if x["fields"]["control_enable"] == [1]]
            stopped = [x for x in poses if x["fields"]["control_stop"] == [1]]
            if behavior == "normal":
                assert completed.returncode == 0, summary
                assert summary["actions_recorded"] >= 5, summary
            else:
                assert completed.returncode == 1, summary
            if execute:
                if behavior != "bad_identity":
                    assert stopped, "no stop lease packet received after an active lease"
                    assert len(enabled) >= 2, (summary, poses)
                else:
                    assert not enabled, "wrong-model action was published"
                if stopped:
                    first_stop = next(i for i, x in enumerate(poses) if x["fields"]["control_stop"] == [1])
                    assert all(x["fields"]["control_stop"] == [1] for x in poses[first_stop:]), "old actions replayed after stop"
                sequences = [x["fields"]["action_seq"][0] for x in enabled]
                assert all(b > a for a, b in zip(sequences, sequences[1:])), "outgoing lease sequence replay"
            else:
                assert summary["pose_packets_published"] == 0
            return {"test": name, "passed": True, "returncode": completed.returncode,
                    "reason": summary["reason"], "observations": summary["observations_sent"],
                    "actions_recorded": summary["actions_recorded"], "enabled_packets_received": len(enabled),
                    "stop_packets_received": len(stopped), "no_enable_after_stop": True,
                    "pending_actions_cleared": summary["pending_actions_cleared"],
                    "controller_fixture_armed": False,
                    "fault_events": [x for x in events if x["event"] == "fault"]}
        finally:
            for process in [robot, server]:
                if process is not None:
                    process.terminate()
                    try: process.wait(timeout=3)
                    except subprocess.TimeoutExpired: process.kill(); process.wait()
            server_log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-dataset", type=Path)
    args = parser.parse_args()
    results = []
    for name, kw in [
        ("mock_dry_run_no_pose", {}),
        ("loopback_execute_disarmed_controller_accepts_buffer", {"source": "live", "execute": True}),
        ("loopback_action_stall_sends_stop", {"source": "live", "execute": True, "behavior": "stall"}),
        ("loopback_expired_action_rejected", {"source": "live", "execute": True, "behavior": "stale"}),
        ("loopback_wrong_model_sha_no_enable", {"source": "live", "execute": True, "behavior": "bad_identity"}),
    ]:
        result = one(name, **kw); results.append(result); print(json.dumps(result), flush=True)
    if args.replay_dataset:
        result = one("actual_local_episode_replay_dry_run", source="replay", dataset=args.replay_dataset)
        results.append(result); print(json.dumps(result), flush=True)
    report = {"status": "passed", "scope": "CPU, localhost-only fake HTTP/WS/robot publishers; no real robot or GPU", "results": results}
    (ROOT / "integration_test_results.json").write_text(json.dumps(report, indent=2) + "\n")

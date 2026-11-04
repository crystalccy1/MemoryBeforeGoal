"""Actual package server.MockBackend <-> run_client mock source on localhost."""
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
with tempfile.TemporaryDirectory(prefix="package-server-it-", dir=ROOT) as td:
    temp = Path(td)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]
    with (temp / "server.log").open("w") as output:
        server = subprocess.Popen([sys.executable, str(ROOT.parent / "server/serve.py"), "--mock", "--port", str(port)],
                                  stdout=output, stderr=subprocess.STDOUT)
        try:
            for _ in range(600):
                try:
                    info = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/deployment-info", timeout=.3))
                    break
                except Exception:
                    if server.poll() is not None: raise RuntimeError((temp / "server.log").read_text())
                    time.sleep(.05)
            else: raise RuntimeError("package mock server did not start: " + (temp / "server.log").read_text())
            command = [sys.executable, str(ROOT / "run_client.py"), "mock", "--server-url", f"ws://127.0.0.1:{port}/ws",
                       "--expected-checkpoint-sha", info["checkpoint_sha256"],
                       "--expected-stats-sha", info["stats_sha256"], "--expected-contract-sha", info["contract_sha256"],
                       "--duration-seconds", "3", "--log-dir", str(temp / "client")]
            result = subprocess.run(command, capture_output=True, text=True, timeout=20)
            paths = list((temp / "client").glob("*/summary.json"))
            if not paths: raise AssertionError(result.stdout + result.stderr)
            summary = json.loads(paths[0].read_text())
            events = [json.loads(x) for x in paths[0].with_name("events.jsonl").read_text().splitlines()]
            actions = [x for x in events if x["event"] == "action"]
            if result.returncode != 0: raise AssertionError(result.stdout + result.stderr + (temp / "server.log").read_text())
            assert len(actions) >= 45 and summary["pose_packets_published"] == 0, summary
            assert len({x["chunk_id"] for x in actions}) >= 2
            assert all(b["action_seq"] > a["action_seq"] for a, b in zip(actions, actions[1:]))
            assert all(b["source_observation_seq"] >= a["source_observation_seq"] for a, b in zip(actions, actions[1:]))
            report = {"status": "passed", "scope": "actual package server.MockBackend with actual client, localhost only, no robot/GPU",
                      "returncode": result.returncode, "summary": summary, "action_count": len(actions),
                      "chunks": sorted({x["chunk_id"] for x in actions}),
                      "max_source_age_ms": max(x["source_age_ms"] for x in actions),
                      "max_rtt_upper_ms": max(x["rtt_upper_ms"] for x in actions),
                      "clipped_hand_values": sum(x["hand_clipped_count"] for x in actions),
                      "identity": {k: info[k] for k in ["checkpoint_sha256", "stats_sha256", "contract_sha256"]}}
            (ROOT / "package_server_integration_results.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))
        finally:
            server.terminate()
            try: server.wait(timeout=3)
            except subprocess.TimeoutExpired: server.kill(); server.wait()

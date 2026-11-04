"""Loopback-only protocol fixture. No model, GPU or robot access."""
import argparse
import asyncio
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi import FastAPI, WebSocket
import uvicorn
from core import CONTROLLER_HASHES, FIXED_CONTRACT

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--behavior", choices=["normal", "stall", "stale", "bad_identity"], default="normal")
parser.add_argument("--execute-fixture", action="store_true",
                    help="Test-only fake handshake, usable only on this hardcoded loopback server")
args = parser.parse_args()
app = FastAPI()
INFO = {**FIXED_CONTRACT, **CONTROLLER_HASHES, "checkpoint_sha256": "a" * 64,
        "stats_sha256": "b" * 64, "contract_sha256": "c" * 64, "inference_steps": 8,
        "ready": True, "latency_ok": True, "dry_run": not args.execute_fixture,
        "mock": not args.execute_fixture, "fault": None}


@app.get("/deployment-info")
def info():
    return INFO


@app.websocket("/ws")
async def websocket(ws: WebSocket):
    await ws.accept()
    hello = await ws.receive_json()
    session = hello["session_id"]
    await ws.send_json({"type": "hello_ack", "protocol": "cart78.v1", "session_id": session, "deployment": INFO})
    sequence = 0
    try:
        while True:
            observation = await ws.receive_json()
            received_ns = time.monotonic_ns()
            if args.behavior == "stall" and sequence >= 12:
                continue
            if args.behavior == "stale" and sequence >= 12:
                await asyncio.sleep(.16)
            now = time.time_ns()
            message = {"type": "action", "protocol": "cart78.v1", "session_id": session,
                "action_seq": sequence, "source_observation_seq": observation["observation_seq"],
                "source_client_sent_monotonic_ns": observation["client_sent_monotonic_ns"],
                "action": [0.0] * 78, "checkpoint_sha256": "a" * 64,
                "stats_sha256": "b" * 64, "contract_sha256": "c" * 64,
                "chunk_id": sequence // 30, "chunk_index": sequence % 30,
                "generation_timestamp_ns": now, "issued_timestamp_ns": now, "ttl_ms": 100,
                "dry_run": hello["mode"] == "dry_run", "latency_ms": 1., "latency_ok": True,
                "server_elapsed_since_source_observation_received_ms": (time.monotonic_ns() - received_ns) / 1e6}
            if args.behavior == "stale" and sequence >= 12:
                message["server_elapsed_since_source_observation_received_ms"] = 0
            if args.behavior == "bad_identity":
                message["checkpoint_sha256"] = "f" * 64
            await ws.send_json(message)
            sequence += 1
    except Exception:
        pass


uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="error")

"""Start the online SAM3 tracker (``tracking/service.py``) as a child process and call it over an AF_UNIX socket."""
from __future__ import annotations

import atexit
import hashlib
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time

import numpy as np

FRAME_SHAPE = (360, 640, 3)
MIN_N_DECL = 16
SERVICE = Path(__file__).resolve().parent / "tracking" / "service.py"
TRACKER_RPC_SCHEMA = "wholebodymem-tracker-rpc/v1"


class TrackerClient:
    """Spawn the SAM3 tracker child and call it over an authenticated AF_UNIX socket.

    The child inherits CUDA_VISIBLE_DEVICES, so it runs on the policy's GPU unless the caller changes that. It listens only
    after SAM3 is loaded; until then the client polls for the socket (``startup_timeout_s``). Requests are strictly
    serialised: one send/recv pair at a time.
    """

    SOCKET_NAME = "wholebodymem-tracker.sock"
    MAX_UNIX_PATH_BYTES = 107  # Linux sun_path holds 108 bytes including the terminating NUL
    CALL_TIMEOUT_S = 120.0
    TRACK_BASE_TIMEOUT_S = 300.0
    TRACK_PER_FRAME_TIMEOUT_S = 60.0

    def __init__(self, *, python: Path | None, sam_repo: Path, checkpoint: Path, prompt: str, namespace: str,
                 n_decl: int, socket_dir: Path, trim_past_keep: int = 16, check_weights_sha256: bool = True,
                 startup_timeout_s: float = 900.0) -> None:
        self.process = None
        self._conn = None
        self._failure: str | None = None
        python = Path(python or sys.executable)
        sam_repo, checkpoint, socket_dir = (Path(p) for p in (sam_repo, checkpoint, socket_dir))
        for path in (python, SERVICE, checkpoint):
            if not path.is_file():
                raise FileNotFoundError(f"SAM3 tracker input is missing: {path}")
        for path in (sam_repo, socket_dir):
            if not path.is_dir():
                raise NotADirectoryError(f"SAM3 tracker directory is missing: {path}")
        if isinstance(n_decl, bool) or int(n_decl) != n_decl or n_decl < MIN_N_DECL:
            raise ValueError(f"SAM3 N_decl must be an integer >= {MIN_N_DECL}")
        self.address = str(socket_dir / self.SOCKET_NAME)
        if len(os.fsencode(self.address)) > self.MAX_UNIX_PATH_BYTES:
            raise ValueError(f"SAM3 tracker socket path exceeds {self.MAX_UNIX_PATH_BYTES} bytes: {self.address}")
        if os.path.lexists(self.address):
            raise FileExistsError(f"Refusing to reuse an existing tracker socket: {self.address}")
        self.prompt, self.namespace, self.n_decl = prompt, namespace, int(n_decl)
        self.identity: dict | None = None
        self.last_timing: dict | None = None
        self._authkey = secrets.token_bytes(32)
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)   # the service imports its siblings from its own directory
        env.update({"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "GIT_OPTIONAL_LOCKS": "0",
                    "PYTHONNOUSERSITE": "1", "HF_HUB_OFFLINE": "1"})
        self.argv = [str(python), str(SERVICE), "--socket", self.address, "--sam-repo", str(sam_repo),
                     "--checkpoint", str(checkpoint), "--prompt", prompt, "--namespace", namespace,
                     "--n-decl", str(self.n_decl), "--trim-past-keep", str(int(trim_past_keep))]
        if not check_weights_sha256:
            self.argv.append("--skip-weights-sha256")
        self.process = subprocess.Popen(self.argv, stdin=subprocess.PIPE, env=env, close_fds=True)
        self.pid = self.process.pid
        atexit.register(self.close)
        try:
            self.process.stdin.write(self._authkey.hex().encode("ascii") + b"\n")  # authkey on stdin
            self.process.stdin.close()
            self._conn = self._connect(float(startup_timeout_s))
        except BaseException:
            self.close()
            raise

    def _connect(self, timeout_s: float):
        from multiprocessing.connection import Client
        deadline = time.monotonic() + timeout_s
        while True:
            if self.process.poll() is not None:
                raise RuntimeError(f"SAM3 tracker exited with {self.process.returncode} before listening")
            if os.path.exists(self.address):
                try:
                    return Client(self.address, family="AF_UNIX", authkey=self._authkey)
                except (FileNotFoundError, ConnectionRefusedError):
                    pass
            if time.monotonic() > deadline:
                raise TimeoutError(f"SAM3 tracker did not listen on {self.address} within {timeout_s:.0f} s")
            time.sleep(0.5)

    def _call(self, op: str, timeout_s: float, **fields) -> dict:
        if self._failure is not None or self._conn is None:
            raise RuntimeError(f"SAM3 tracker client is unusable: {self._failure or 'not connected'}")
        try:
            if self.process.poll() is not None:
                raise RuntimeError(f"SAM3 tracker exited with {self.process.returncode}")
            self._conn.send({"op": op, **fields})
            deadline = time.monotonic() + timeout_s
            while not self._conn.poll(1.0):
                if self.process.poll() is not None:
                    raise RuntimeError(f"SAM3 tracker exited with {self.process.returncode} during {op}")
                if time.monotonic() > deadline:
                    raise TimeoutError(f"SAM3 tracker {op} exceeded {timeout_s:.0f} s")
            reply = self._conn.recv()
        except BaseException as exc:  # the stream may be out of sync: never use it again
            self._failure = f"{op}: {type(exc).__name__}: {exc}"
            raise
        if not isinstance(reply, dict) or reply.get("ok") is not True:
            error = reply.get("error") if isinstance(reply, dict) else f"malformed reply {type(reply).__name__}"
            detail = reply.get("traceback", "") if isinstance(reply, dict) else ""
            raise RuntimeError(f"SAM3 tracker {op} failed: {error}\n{detail}")
        return reply

    def hello(self) -> dict:
        identity = self._call("hello", self.CALL_TIMEOUT_S).get("identity")
        expected = {"schema": TRACKER_RPC_SCHEMA, "pid": self.pid, "prompt": self.prompt, "namespace": self.namespace,
                    "n_decl": self.n_decl, "frame_shape": list(FRAME_SHAPE), "apply_temporal_disambiguation": False}
        if not isinstance(identity, dict):
            raise RuntimeError("SAM3 tracker returned no identity")
        drift = {key: (identity.get(key), value) for key, value in expected.items() if identity.get(key) != value}
        backends = identity.get("backends") or {}
        if (backends.get("deterministic_algorithms") is not True or backends.get("tf32_matmul") is not False
                or backends.get("tf32_cudnn") is not False or backends.get("cublas_workspace_config") != ":4096:8"):
            drift["backends"] = (backends, "deterministic, TF32 off, CUBLAS_WORKSPACE_CONFIG=:4096:8")
        if drift:
            raise RuntimeError(f"SAM3 tracker identity differs from the expected settings (got, expected): {drift}")
        self.identity = identity
        return identity

    def reset(self, session_id: str, episode_index: int) -> None:
        reply = self._call("reset", self.CALL_TIMEOUT_S, session_id=session_id, episode_index=int(episode_index))
        if (reply.get("session_id"), reply.get("episode_index"), reply.get("next_index")) != (
                session_id, int(episode_index), 0):
            raise RuntimeError("SAM3 tracker reset acknowledged a different session")

    def track(self, session_id: str, episode_index: int, start_index: int, frames: np.ndarray) -> list[dict]:
        start_index, frames = int(start_index), np.ascontiguousarray(frames)
        if frames.dtype != np.uint8 or frames.ndim != 4 or tuple(frames.shape[1:]) != FRAME_SHAPE or not len(frames):
            raise ValueError(f"SAM3 tracker frames must be uint8 [N,{FRAME_SHAPE}]")
        digests = [hashlib.sha256(frame.tobytes()).hexdigest() for frame in frames]
        timeout = self.TRACK_BASE_TIMEOUT_S + self.TRACK_PER_FRAME_TIMEOUT_S * len(frames)
        reply = self._call("track", timeout, session_id=session_id, episode_index=int(episode_index),
                           start_index=start_index, frames=frames, sha256=digests)
        if (reply.get("replayed") or reply.get("start_index") != start_index
                or reply.get("next_index") != start_index + len(frames)):
            raise RuntimeError("SAM3 tracker reply is a replay or covers different frames")
        records, timing = reply.get("records"), reply.get("timing")
        if not isinstance(records, list) or len(records) != len(frames):
            raise RuntimeError("SAM3 tracker returned the wrong number of records")
        if not isinstance(timing, dict) or any(len(timing.get(k) or ()) != len(frames) for k in ("sam3_s", "bc_s")):
            raise RuntimeError("SAM3 tracker returned malformed timing")
        self.last_timing = {k: [float(v) for v in timing[k]] for k in ("sam3_s", "bc_s")}
        return records

    def close_session(self) -> None:
        self._call("close", self.CALL_TIMEOUT_S)

    def alive(self) -> bool:
        return self._failure is None and self.process is not None and self.process.poll() is None

    def close(self) -> None:
        """Idempotent shutdown: shutdown RPC, then terminate/kill the child if it lingers."""
        conn, self._conn = self._conn, None
        if self._failure is None:
            self._failure = "closed"
        if conn is not None:
            try:
                conn.send({"op": "shutdown"})
                if conn.poll(10.0):
                    conn.recv()
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass
        process = self.process
        if process is not None and process.poll() is None:
            for stop in (None, process.terminate, process.kill):
                if stop is not None:
                    stop()
                try:
                    process.wait(timeout=30 if stop is None else 10)
                    break
                except subprocess.TimeoutExpired:
                    continue

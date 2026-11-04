#!/usr/bin/env python3
"""cart78.v1 client. Dry-run by default; --execute is live-source only."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import secrets
import signal
import threading
import time
import urllib.request
from urllib.parse import urlsplit, urlunsplit
import uuid

try:
    from .core import (ActionGate, ContractError, DEFAULT_MAX_HAND_CLIP_RAD, ExpectedIdentity, NotReady, PROTOCOL,
                       pack_control_pose)
    from .sources import LiveSource, MockSource, ReplaySource
except ImportError:
    from core import (ActionGate, ContractError, DEFAULT_MAX_HAND_CLIP_RAD, ExpectedIdentity, NotReady, PROTOCOL,
                      pack_control_pose)
    from sources import LiveSource, MockSource, ReplaySource


def deployment_info_url(websocket_url):
    u = urlsplit(websocket_url)
    if u.scheme not in {"ws", "wss"} or not u.netloc:
        raise ContractError("server URL must be ws:// or wss://")
    return urlunsplit(("https" if u.scheme == "wss" else "http", u.netloc, "/deployment-info", "", ""))


class PosePublisher:
    def __init__(self, bind, instance, session):
        import zmq
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 1)
        self.socket.setsockopt(zmq.LINGER, 0)
        # No connect-to-existing publisher or automatic takeover.
        self.socket.bind(bind)
        self.instance, self.session, self.sequence = instance, session, 0
        self.zmq = zmq

    def send(self, action):
        self.sequence += 1
        packet = pack_control_pose(action, self.instance, self.session, self.sequence)
        self.socket.send(packet, flags=self.zmq.NOBLOCK)
        return self.sequence

    def stop(self):
        self.sequence += 1
        packet = pack_control_pose(None, self.instance, self.session, self.sequence, stop=True)
        # Best effort; the independent native controller lease remains the backstop.
        for _ in range(3):
            try:
                self.socket.send(packet, flags=self.zmq.NOBLOCK)
            except self.zmq.ZMQError:
                break
            time.sleep(.02)
        self.socket.close(linger=0)
        self.context.term()


class Receiver:
    def __init__(self, websocket, gate):
        self.websocket, self.gate = websocket, gate
        self.lock = threading.Lock()
        self.pending = None
        self.fault = None
        self.profiles = []
        self.last_action_ns = None
        self.received_count = 0
        self.replaced_pending = 0
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self._run, name="cart78-receiver", daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        from websocket import WebSocketTimeoutException
        try:
            while not self.stopping.is_set():
                try:
                    raw = self.websocket.recv()
                except WebSocketTimeoutException:
                    continue
                if not raw:
                    raise ContractError("server disconnected")
                if not isinstance(raw, str) or len(raw) > 1024 * 1024:
                    raise ContractError("unexpected WS frame type/size")
                message = json.loads(raw)
                if message.get("protocol") != PROTOCOL or message.get("session_id") != self.gate.session_id:
                    raise ContractError("server envelope/session mismatch")
                if message.get("type") == "profile":
                    if self.gate.execute:
                        raise ContractError("server profiling/revoked readiness during execute")
                    with self.lock:
                        self.profiles.append(message)
                        self.profiles = self.profiles[-10:]
                    continue
                if message.get("type") == "fault":
                    raise ContractError("server fault: " + str(message.get("reason", message.get("fault", "unspecified"))))
                if message.get("type") == "error":
                    raise ContractError(f"server error {message.get('code', 'unknown')}: {message.get('message', '')}")
                accepted = self.gate.validate(message, time.monotonic_ns())
                with self.lock:
                    self.replaced_pending += int(self.pending is not None)
                    self.pending = accepted
                    self.last_action_ns = accepted["received_ns"]
                    self.received_count += 1
        except Exception as exc:
            if not self.stopping.is_set():
                with self.lock:
                    self.fault = f"{type(exc).__name__}: {exc}"
                    self.pending = None

    def take(self):
        with self.lock:
            value, self.pending = self.pending, None
            return value

    def status(self):
        with self.lock:
            profiles, self.profiles = self.profiles, []
            return self.fault, self.last_action_ns, profiles

    def stop(self):
        self.stopping.set()
        with self.lock:
            self.pending = None
        try:
            self.websocket.close()
        finally:
            if self.thread.ident is not None:
                self.thread.join(timeout=2)
            self.gate.clear()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", choices=["live", "replay", "mock"])
    p.add_argument("--server-url", required=True)
    p.add_argument("--bundle-manifest", type=Path)
    p.add_argument("--expected-checkpoint-sha")
    p.add_argument("--expected-stats-sha")
    p.add_argument("--expected-contract-sha")
    p.add_argument("--execute", action="store_true", help="Only live mode; publish guarded SONIC poses after complete identity validation")
    p.add_argument("--instruction", default="Move the dining cart.")
    p.add_argument("--hand-limit-policy", choices=["reject", "clip"], default="reject",
                   help="clip projects correctly reordered hands onto the inspected Dex3 driver bounds; every projection is logged")
    p.add_argument("--max-hand-clip-rad", type=float, default=DEFAULT_MAX_HAND_CLIP_RAD,
                   help="clip mode only: fault (stop instead of projecting) when any hand value lies further than "
                        "this outside its Dex3 limit. Default %(default)s rad")
    p.add_argument("--state-endpoint", default="tcp://127.0.0.1:5557")
    p.add_argument("--camera-endpoint", default="tcp://192.168.123.164:5555")
    p.add_argument("--pose-bind", default="tcp://*:5556")
    p.add_argument("--camera-width", type=int, default=640)
    p.add_argument("--camera-height", type=int, default=360)
    p.add_argument("--camera-clock-offset-ms", type=float, default=0,
                   help="Measured client_wall minus camera_wall offset; default requires synchronized wall clocks")
    p.add_argument("--max-state-age-ms", type=float, default=250)
    p.add_argument("--max-camera-age-ms", type=float, default=500)
    p.add_argument("--max-source-observation-age-ms", type=float, default=1000)
    p.add_argument("--action-stream-timeout-ms", type=float, default=200)
    p.add_argument("--startup-timeout-seconds", type=float, default=30)
    p.add_argument("--duration-seconds", type=float, default=0, help="0 means until interrupted/end of replay")
    p.add_argument("--dataset", type=Path)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--log-dir", type=Path)
    return p


def run(args):
    if args.execute and args.source != "live":
        raise ContractError("--execute is forbidden for mock and replay sources")
    if args.source == "replay" and args.dataset is None:
        raise ContractError("replay requires --dataset")
    for field in ["max_state_age_ms", "max_camera_age_ms", "max_source_observation_age_ms", "action_stream_timeout_ms", "startup_timeout_seconds"]:
        if not 0 < getattr(args, field) < 60000:
            raise ContractError(f"invalid {field}")
    if not 0 < args.max_hand_clip_rad <= 1:
        raise ContractError("--max-hand-clip-rad must be in (0, 1] rad")
    identity = ExpectedIdentity.from_bundle(args.bundle_manifest) if args.bundle_manifest else ExpectedIdentity(
        args.expected_checkpoint_sha, args.expected_stats_sha, args.expected_contract_sha)
    for key, value in [("image_width", args.camera_width), ("image_height", args.camera_height)]:
        if key in identity.contract and identity.contract[key] != value:
            raise ContractError(f"camera {key} differs from the bundle")
    session = str(uuid.uuid4())
    control_session = secrets.randbelow(2**63 - 1) + 1
    log_dir = (args.log_dir or Path.cwd() / "client_logs") / session
    log_dir.mkdir(parents=True, exist_ok=False)
    logfile = (log_dir / "events.jsonl").open("x", buffering=1)
    def event(kind, **fields):
        row = {"event": kind, "client_monotonic_ns": time.monotonic_ns(), "wall_timestamp_ns": time.time_ns(), **fields}
        logfile.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
        if kind != "action":
            print(json.dumps(row, separators=(",", ":")), flush=True)
    source = receiver = publisher = websocket = None
    shutdown = threading.Event()
    previous_handlers = {}
    for signum in [signal.SIGINT, signal.SIGTERM]:
        previous_handlers[signum] = signal.signal(signum, lambda _sig, _frame: shutdown.set())
    started = time.monotonic()
    accepted_count = published_count = observation_seq = 0
    hand_clipped_values = token_clipped_values = 0
    max_hand_clip_delta_rad = 0.0
    reason, exit_code = "duration_or_end", 0
    try:
        event("start", protocol=PROTOCOL, session_id=session, control_session=control_session,
              source=args.source, execute=args.execute, expected=identity.hello_expected(),
              hand_limit_policy=args.hand_limit_policy,
              max_hand_clip_rad=args.max_hand_clip_rad if args.hand_limit_policy == "clip" else None)
        request = urllib.request.Request(deployment_info_url(args.server_url), headers={"Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=5) as response:
            raw = response.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024:
                raise ContractError("deployment-info too large")
            deployment = json.loads(raw)
        identity.check_deployment(deployment, args.execute)
        # Source imports, file/codec setup, and initial live subscriptions are
        # startup work. Finish them before hello starts the server's unchanged
        # 250 ms observation watchdog. No pose publisher exists at this point.
        if args.source == "live":
            source = LiveSource(args.state_endpoint, args.camera_endpoint, execute=args.execute,
                                width=args.camera_width, height=args.camera_height,
                                max_state_age_ms=args.max_state_age_ms, max_camera_age_ms=args.max_camera_age_ms,
                                camera_clock_offset_ms=args.camera_clock_offset_ms)
        elif args.source == "replay":
            source = ReplaySource(args.dataset, args.episode)
        else:
            source = MockSource(args.camera_width, args.camera_height)
        if args.source == "live":
            while not shutdown.is_set():
                if time.monotonic() - started > args.startup_timeout_seconds:
                    raise ContractError("live observation source not ready before startup deadline")
                source.poll()
                try:
                    source.freshness()
                except NotReady:
                    time.sleep(.005)
                    continue
                break
        if shutdown.is_set():
            raise ContractError("startup interrupted before WebSocket handshake")
        event("observation_source_prepared", source=args.source)
        import websocket as ws_module
        websocket = ws_module.create_connection(args.server_url, timeout=5, enable_multithread=True)
        websocket.send(json.dumps({"type": "hello", "protocol": PROTOCOL, "session_id": session,
                                   "mode": "execute" if args.execute else "dry_run", "expected": identity.hello_expected()}))
        ack = json.loads(websocket.recv())
        if ack.get("type") != "hello_ack" or ack.get("protocol") != PROTOCOL or ack.get("session_id") != session:
            raise ContractError("invalid hello acknowledgment")
        identity.check_deployment(ack["deployment"], args.execute)
        event("handshake_verified", deployment=ack["deployment"])
        websocket.settimeout(.2)
        gate = ActionGate(identity, session, args.execute, args.max_source_observation_age_ms, args.hand_limit_policy,
                          args.max_hand_clip_rad if args.hand_limit_policy == "clip" else None)
        receiver = Receiver(websocket, gate)
        receiver.start()
        next_observation = time.monotonic()
        observations_started = False
        active_started = None
        while not shutdown.is_set():
            now = time.monotonic()
            if args.duration_seconds > 0 and active_started is not None and now - active_started >= args.duration_seconds:
                break
            fault, last_action_ns, profiles = receiver.status()
            for profile in profiles:
                event("profile", server=profile)
            if fault:
                raise ContractError(fault)
            if last_action_ns is not None and (time.monotonic_ns() - last_action_ns) / 1e6 > args.action_stream_timeout_ms:
                raise ContractError("action stream stalled; stop lease, do not replay the last action")
            if last_action_ns is None and now - started > args.startup_timeout_seconds:
                raise ContractError("no valid action before startup deadline")
            source.poll()
            try:
                source.freshness()
            except NotReady:
                if observations_started:
                    raise
                time.sleep(.005)
                continue
            if args.execute and publisher is None:
                publisher = PosePublisher(args.pose_bind, source.guard_instance, control_session)
                event("guard_verified", controller_instance=source.guard_instance,
                      controller_timeout_ms=source.guard_timeout_ms, pose_bind=args.pose_bind)
                event("manual_arm_required", note="Fresh leases only buffer commands while disarmed; the operator must use the controller's Enter action. Client never sends start/arm.")
            if now >= next_observation:
                try:
                    state, jpeg, diagnostic = source.snapshot()
                except EOFError:
                    reason = "replay_complete"
                    break
                sent_ns = time.monotonic_ns()
                gate.observe(observation_seq, sent_ns)
                websocket.send(json.dumps({"type": "observation", "protocol": PROTOCOL, "session_id": session,
                    "observation_seq": observation_seq, "client_sent_monotonic_ns": sent_ns,
                    "state_model43": state.tolist(), "image_jpeg_b64": base64.b64encode(jpeg).decode(),
                    "instruction": args.instruction}, allow_nan=False))
                if observation_seq == 0:
                    active_started = time.monotonic()
                    event("observation_source_verified", diagnostic=diagnostic,
                          first_image_sha256=hashlib.sha256(jpeg).hexdigest(), state_dim=len(state))
                observation_seq += 1
                observations_started = True
                next_observation = now + 1 / 30
            # The server owns the 30 Hz action clock. A second independently
            # paced client clock would drop rows through gradual phase drift.
            action = receiver.take()
            if action is not None:
                source.freshness()
                if time.monotonic_ns() >= action["deadline_ns"]:
                    raise ContractError("action expired before sender; refusing buffered action")
                control_seq = publisher.send(action["wire_action"]) if publisher else None
                accepted_count += 1
                published_count += int(publisher is not None)
                hand_clipped_values += action["hand_clipped_count"]
                token_clipped_values += action["token_clipped_dimensions"]
                max_hand_clip_delta_rad = max(max_hand_clip_delta_rad, action["hand_max_delta_rad"])
                message = action["message"]
                event("action", action_seq=message["action_seq"], source_observation_seq=message["source_observation_seq"],
                      chunk_id=message["chunk_id"], chunk_index=message["chunk_index"],
                      source_age_ms=action["source_age_ms"], rtt_upper_ms=action["rtt_upper_ms"],
                      token_clipped_dimensions=action["token_clipped_dimensions"], control_action_seq=control_seq,
                      hand_limit_policy=action["hand_limit_policy"], hand_clipped_count=action["hand_clipped_count"],
                      hand_max_delta_rad=action["hand_max_delta_rad"],
                      published=publisher is not None, model_action=action["model_action"].tolist(),
                      wire_action=action["wire_action"].tolist())
            time.sleep(.002)
        if shutdown.is_set():
            reason = "signal_stop"
    except Exception as exc:
        reason, exit_code = f"{type(exc).__name__}: {exc}", 1
        event("fault", reason=reason)
    finally:
        # Stop publication before closing source sockets, clear all old actions,
        # and do not reconnect or restart the lease automatically.
        if publisher is not None:
            publisher.stop()
            event("stop_lease_sent", controller_instance=publisher.instance, control_session=control_session)
        if receiver is not None:
            receiver.stop()
        elif websocket is not None:
            websocket.close()
        if source is not None:
            source.clear()
            source.close()
        if exit_code == 0 and accepted_count == 0:
            exit_code, reason = 2, "no_valid_actions_observed"
        summary = {"exit_code": exit_code, "reason": reason, "source": args.source,
                   "execute": args.execute, "session_id": session, "observations_sent": observation_seq,
                   "actions_recorded": accepted_count, "pose_packets_published": published_count,
                   "validated_actions_received": receiver.received_count if receiver else 0,
                   "superseded_pending_actions": receiver.replaced_pending if receiver else 0,
                   "hand_limit_policy": args.hand_limit_policy, "hand_clipped_values": hand_clipped_values,
                   "hand_clipped_fraction": hand_clipped_values / (14 * accepted_count) if accepted_count else 0.0,
                   "max_hand_clip_delta_rad": max_hand_clip_delta_rad, "token_clipped_values": token_clipped_values,
                   "pending_actions_cleared": True, "automatic_reconnect": False,
                   "log_dir": str(log_dir.resolve())}
        event("finished", **summary)
        (log_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        logfile.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    return exit_code


if __name__ == "__main__":
    args = parser().parse_args()
    try:
        raise SystemExit(run(args))
    except ContractError as exc:
        raise SystemExit(str(exc))

"""Loopback state/camera publishers and pose recorder; no robot SDK or motors."""
import argparse
import json
import signal
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import msgpack
import numpy as np
import zmq
from core import CONTROLLER_HASHES
from sources import rgb_to_jpeg

p = argparse.ArgumentParser()
p.add_argument("--state-port", type=int, required=True)
p.add_argument("--camera-port", type=int, required=True)
p.add_argument("--pose-port", type=int, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
ctx = zmq.Context()
state = ctx.socket(zmq.PUB); state.bind(f"tcp://127.0.0.1:{a.state_port}")
camera = ctx.socket(zmq.PUB); camera.bind(f"tcp://127.0.0.1:{a.camera_port}")
poses = ctx.socket(zmq.SUB); poses.setsockopt(zmq.SUBSCRIBE, b"pose"); poses.connect(f"tcp://127.0.0.1:{a.pose_port}")
running = True
def stop(*_):
    global running
    running = False
signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
config = {**CONTROLLER_HASHES, "controller_guard_protocol": "cart78.token_lease.v1",
          "controller_guard_active": True, "controller_guard_instance": "123456789",
          "controller_guard_timeout_ms": 250, "controller_guard_armed": False}
rgb = np.zeros((360, 640, 3), np.uint8); rgb[:] = [180, 40, 20]
jpeg = rgb_to_jpeg(rgb)
index = 0
last = 0.
with a.output.open("w", buffering=1) as log:
    while running:
        now = time.monotonic()
        if now - last >= .02:
            packet = {"control_loop_type": "cpp", "index": index, "body_q_measured": [0.] * 29,
                      "body_q": [0.] * 29, "left_hand_q": [0.] * 7, "right_hand_q": [0.] * 7,
                      "base_quat": [1., 0., 0., 0.], "state_timestamp_monotonic": now}
            state.send(b"g1_debug" + msgpack.packb(packet, use_bin_type=True))
            state.send(b"robot_config" + msgpack.packb(config, use_bin_type=True))
            camera.send(msgpack.packb({"timestamps": {"ego_view": time.time()}, "images": {"ego_view": jpeg}}, use_bin_type=True))
            index += 1
            last = now
        if poses.poll(timeout=2):
            message = poses.recv()
            header = json.loads(message[4:1284].rstrip(b"\0"))
            offset = 1284
            decoded = {}
            for field in header["fields"]:
                dtype = {"u8": "u1", "i64": "<i8", "f32": "<f4"}[field["dtype"]]
                n = int(np.prod(field["shape"]))
                value = np.frombuffer(message, dtype=dtype, count=n, offset=offset).reshape(field["shape"])
                offset += value.nbytes
                decoded[field["name"]] = value.tolist()
            assert offset == len(message)
            log.write(json.dumps({"received_ns": time.monotonic_ns(), "header": header, "fields": decoded}) + "\n")
for sock in [state, camera, poses]: sock.close(linger=0)
ctx.term()

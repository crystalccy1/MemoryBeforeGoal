"""Observed collector camera protocol and observation sources; no robot imports."""
from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path
import time
import numpy as np
from PIL import Image

try:
    from .core import ContractError, NotReady, finite_vector, integer, raw_recording_state_to_model, wire_state_to_model, validate_guard
except ImportError:
    from core import ContractError, NotReady, finite_vector, integer, raw_recording_state_to_model, wire_state_to_model, validate_guard


def rgb_to_jpeg(rgb):
    stream = io.BytesIO()
    Image.fromarray(rgb).save(stream, format="JPEG", quality=95)
    return stream.getvalue()


def decode_camera_message(data, expected_shape=(360, 640, 3)):
    """Match actual ImageMessageSchema: bytes RGB, legacy str OpenCV array order.

    The collector's legacy str branch uses cv2.imdecode without BGR->RGB;
    reproduce its returned array exactly, then send that training array as RGB.
    Raw JPEG bytes instead get the collector's BGR->RGB conversion.
    """
    if not isinstance(data, dict) or "ego_view" not in data.get("images", {}):
        raise ContractError("composed camera message lacks images.ego_view")
    stamp = data.get("timestamps", {}).get("ego_view")
    if isinstance(stamp, bool) or not isinstance(stamp, (float, int)) or not math.isfinite(stamp) or stamp <= 0:
        raise ContractError("invalid ego_view capture wall timestamp")
    value = data["images"]["ego_view"]
    legacy = isinstance(value, str)
    if legacy:
        try:
            encoded = base64.b64decode(value, validate=True)
        except (ValueError, TypeError) as exc:
            raise ContractError("invalid camera base64 JPEG") from exc
    elif isinstance(value, (bytes, bytearray)):
        encoded = bytes(value)
    else:
        encoded = None
    if encoded is not None:
        if not 0 < len(encoded) <= 8 * 1024 * 1024:
            raise ContractError("camera JPEG size outside bound")
        try:
            with Image.open(io.BytesIO(encoded)) as image:
                if image.format != "JPEG" or image.size != (expected_shape[1], expected_shape[0]):
                    raise ContractError("camera JPEG dimensions/format differ from the training contract")
                rgb = np.array(image.convert("RGB"))
        except (OSError, ValueError) as exc:
            raise ContractError("camera JPEG decode failed") from exc
        if legacy:
            rgb = rgb[..., ::-1].copy()
        encoding = "legacy_base64_jpeg_collector_array" if legacy else "raw_jpeg_rgb"
    elif isinstance(value, dict) and (value.get(b"nd") is True or value.get("nd") is True):
        # Minimal msgpack_numpy uint8 decoder. Never unpickle object arrays.
        get = lambda k: value[k.encode()] if k.encode() in value else value.get(k)
        try:
            dtype = np.dtype(get("type"))
            shape = tuple(get("shape"))
            raw = get("data")
            if dtype != np.uint8 or shape != expected_shape or len(raw) != int(np.prod(shape)):
                raise ContractError("camera ndarray metadata mismatch")
            rgb = np.frombuffer(raw, dtype=np.uint8).reshape(shape).copy()
        except (ValueError, TypeError, KeyError) as exc:
            raise ContractError("invalid msgpack camera ndarray") from exc
        encoding = "msgpack_numpy_uint8"
    elif isinstance(value, np.ndarray):
        rgb, encoding = value, "ndarray"
    else:
        raise ContractError("unsupported composed camera image encoding")
    if rgb.dtype != np.uint8 or rgb.shape != expected_shape:
        raise ContractError("camera array shape/dtype mismatch")
    return np.ascontiguousarray(rgb), float(stamp), encoding


class LiveSource:
    def __init__(self, state_endpoint, camera_endpoint, *, execute=False, width=640, height=360,
                 max_state_age_ms=250, max_camera_age_ms=500, max_config_age_ms=6000,
                 camera_clock_offset_ms=0):
        import zmq
        self.zmq = zmq
        self.context = zmq.Context()
        self.state_socket = self._sub(state_endpoint, b"g1_debug")
        self.config_socket = self._sub(state_endpoint, b"robot_config")
        self.camera_socket = self._sub(camera_endpoint, b"")
        self.execute = execute
        self.shape = (height, width, 3)
        self.max_state_age_ms = max_state_age_ms
        self.max_camera_age_ms = max_camera_age_ms
        self.max_config_age_ms = max_config_age_ms
        self.camera_clock_offset_ms = camera_clock_offset_ms
        self.state = self.camera = self.config = None
        self.state_index = -1
        self.camera_stamp = 0.0
        self.state_ns = self.camera_ns = self.config_ns = 0
        self.guard_instance = None
        self.guard_timeout_ms = None

    def _sub(self, endpoint, topic):
        sock = self.context.socket(self.zmq.SUB)
        sock.setsockopt(self.zmq.SUBSCRIBE, topic)
        sock.setsockopt(self.zmq.CONFLATE, 1)
        sock.setsockopt(self.zmq.RCVHWM, 1)
        sock.setsockopt(self.zmq.LINGER, 0)
        sock.connect(endpoint)
        return sock

    def _receive(self, sock, topic):
        import msgpack
        try:
            raw = sock.recv(self.zmq.NOBLOCK)
        except self.zmq.Again:
            return None
        if len(raw) > 12 * 1024 * 1024 or not raw.startswith(topic):
            raise ContractError("ZMQ payload size/topic mismatch")
        try:
            return msgpack.unpackb(raw[len(topic):], raw=False, strict_map_key=False)
        except Exception as exc:
            raise ContractError("invalid ZMQ msgpack payload") from exc

    def poll(self):
        config = self._receive(self.config_socket, b"robot_config")
        if config is not None:
            if not isinstance(config, dict):
                raise ContractError("robot_config must be a map")
            if self.execute:
                instance, lease = validate_guard(config, self.guard_instance)
                self.guard_instance, self.guard_timeout_ms = instance, lease
            self.config, self.config_ns = config, time.monotonic_ns()
        state = self._receive(self.state_socket, b"g1_debug")
        if state is not None:
            if not isinstance(state, dict) or state.get("control_loop_type") != "cpp":
                raise ContractError("g1_debug is not the expected C++ state map")
            index = integer(state.get("index"), "state index")
            if index < self.state_index:
                raise ContractError("state sequence restarted; explicit client restart required")
            if index > self.state_index:
                model_state = wire_state_to_model(state)
                if "body_q" in state:
                    q = finite_vector(state["body_q"], 29, "wire body_q")
                    if not np.allclose(q, model_state[:29], atol=1e-5, rtol=0):
                        raise ContractError("wire body_q and body_q_measured disagree with the inspected serializer")
                if "base_quat" in state:
                    quat = finite_vector(state["base_quat"], 4, "base_quat")
                    if not .9 <= np.linalg.norm(quat) <= 1.1:
                        raise ContractError("invalid robot quaternion")
                self.state, self.state_ns, self.state_index = model_state, time.monotonic_ns(), index
        camera = self._receive(self.camera_socket, b"")
        if camera is not None:
            rgb, stamp, encoding = decode_camera_message(camera, self.shape)
            if stamp < self.camera_stamp:
                raise ContractError("camera capture timestamp regressed")
            if stamp > self.camera_stamp:
                self.camera = (rgb_to_jpeg(rgb), encoding)
                self.camera_ns, self.camera_stamp = time.monotonic_ns(), stamp

    def freshness(self):
        now = time.monotonic_ns()
        if self.state is None or self.camera is None:
            raise NotReady("waiting for a new robot state and composed ego-view camera frame")
        state_age = (now - self.state_ns) / 1e6
        camera_age = (now - self.camera_ns) / 1e6
        wall_age = (time.time() - self.camera_stamp) * 1000 - self.camera_clock_offset_ms
        if state_age > self.max_state_age_ms:
            raise ContractError(f"robot state stopped advancing: {state_age:.1f} ms")
        if camera_age > self.max_camera_age_ms or not -50 <= wall_age <= self.max_camera_age_ms:
            raise ContractError(f"camera stale/clock mismatch: arrival={camera_age:.1f} ms capture={wall_age:.1f} ms")
        if self.execute:
            if self.config is None or self.guard_instance is None:
                raise NotReady("waiting for patched controller identity")
            if (now - self.config_ns) / 1e6 > self.max_config_age_ms:
                raise ContractError("robot_config identity heartbeat stale")
            validate_guard(self.config, self.guard_instance)
        return {"state_age_ms": state_age, "camera_arrival_age_ms": camera_age, "camera_capture_age_ms": wall_age,
                "state_index": self.state_index, "camera_encoding": self.camera[1]}

    def snapshot(self):
        diagnostics = self.freshness()
        return self.state.copy(), self.camera[0], diagnostics

    def clear(self):
        self.state = self.camera = None
        self.state_ns = self.camera_ns = 0

    def close(self):
        self.clear()
        for socket in [self.state_socket, self.config_socket, self.camera_socket]:
            socket.close(linger=0)
        self.context.term()


class MockSource:
    """No ZMQ or robot access. One fresh synthetic observation per main-loop tick."""
    def __init__(self, width=640, height=360):
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[:, :width // 2] = [180, 40, 20]
        rgb[:, width // 2:] = [20, 80, 180]
        self.jpeg = rgb_to_jpeg(rgb)
        self.state = np.zeros(43, dtype=np.float32)
        self.index = 0
        self.guard_instance = None

    def poll(self):
        pass

    def freshness(self):
        return {"synthetic": True, "state_index": self.index}

    def snapshot(self):
        self.index += 1
        return self.state.copy(), self.jpeg, self.freshness()

    def clear(self):
        pass

    def close(self):
        pass


class ReplaySource:
    """Decode one local episode. Never subscribes to or publishes robot traffic."""
    def __init__(self, dataset, episode=0):
        import pyarrow.parquet as pq
        import cv2
        self.cv2 = cv2
        root = Path(dataset)
        info = json.loads((root / "meta/info.json").read_text())
        if info["fps"] != 30:
            raise ContractError("replay expects the verified 30 Hz dataset")
        chunk = episode // info.get("chunks_size", 1000)
        parquet = root / info["data_path"].format(episode_chunk=chunk, episode_index=episode)
        table = pq.read_table(parquet)
        self.prepared = "states" in table.column_names
        column = "states" if self.prepared else "observation.state"
        self.rows = table[column].to_pylist()
        video_key = next((k for k, v in info["features"].items() if v.get("dtype") == "video"), None)
        if video_key is None:
            raise ContractError("replay metadata has no video")
        video = root / info["video_path"].format(episode_chunk=chunk, episode_index=episode, video_key=video_key)
        self.cap = cv2.VideoCapture(str(video))
        if not self.cap.isOpened():
            raise ContractError(f"cannot decode replay video: {video}")
        if int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) != len(self.rows):
            raise ContractError("replay video/Parquet frame count mismatch")
        self.index = 0
        self.guard_instance = None

    def poll(self):
        pass

    def freshness(self):
        return {"replay": True, "state_index": self.index}

    def snapshot(self):
        if self.index >= len(self.rows):
            raise EOFError("replay episode completed")
        ok, image = self.cap.read()
        if not ok:
            raise ContractError("replay video ended early")
        raw = self.rows[self.index]
        state = finite_vector(raw, 43, "prepared state") if self.prepared else raw_recording_state_to_model(raw)
        diagnostic = self.freshness()
        self.index += 1
        return state, rgb_to_jpeg(image[..., ::-1]), diagnostic

    def clear(self):
        pass

    def close(self):
        self.cap.release()

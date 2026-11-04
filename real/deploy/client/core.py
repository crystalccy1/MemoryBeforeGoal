"""Pure CPU contracts/adapters. Importing this module opens no sockets."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import time
import numpy as np

PROTOCOL = "cart78.v1"
GUARD_PROTOCOL = "cart78.token_lease.v1"
ACTUATED_TO_MODEL = np.array([3, 4, 5, 6, 0, 1, 2])
MODEL_TO_ACTUATED = np.array([4, 5, 6, 0, 1, 2, 3])
HEADER_SIZE = 1280
FIXED_CONTRACT = {
    "state_dim": 43, "action_dim": 78, "horizon": 30, "action_tick_hz": 30,
    "state_order": "model_urdf", "action_order": "token64+left_model7+right_model7",
    "actions_normalized": False, "max_delay": 8,
}
CONTROLLER_HASHES = {
    "controller_encoder_sha256": "013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3",
    "controller_decoder_sha256": "c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed",
    "controller_observation_config_sha256": "466d05947c78af6c76388adfb86e3a2a77b2a1d921a64883ed3d085ebf58de1b",
}
# Stop threshold for --hand-limit-policy clip (rad beyond a Dex3 limit); ported from the trashbin client
# (deploy_trashbin_5k), whose runner passes 0.3 at the user's request (2026-09-25).
DEFAULT_MAX_HAND_CLIP_RAD = 0.12
LEFT_LO = np.array([-1.05, -.724, 0, -1.57, -1.75, -1.57, -1.75])
LEFT_HI = np.array([1.05, 1.05, 1.75, 0, 0, 0, 0])
RIGHT_LO = np.array([-1.05, -1.05, -1.75, 0, 0, 0, 0])
RIGHT_HI = np.array([1.05, .742, 0, 1.57, 1.75, 1.57, 1.75])


class ContractError(RuntimeError):
    pass


class NotReady(ContractError):
    pass


def integer(value, name, minimum=0, maximum=2**63 - 1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ContractError(f"{name} must be an integer")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ContractError(f"{name} outside [{minimum}, {maximum}]")
    return value


def digest(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ContractError(f"{name} must be lowercase SHA256")
    return value


def finite_vector(value, size, name):
    try:
        if np.asarray(value).dtype.kind not in "iuf":
            raise ContractError(f"{name} must contain numeric values, not strings/objects/bools")
        array = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ContractError(f"invalid {name}") from exc
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ContractError(f"{name} must be finite ({size},), got {array.shape}")
    return array.astype(np.float32)


def wire_state_to_model(wire):
    body = finite_vector(wire["body_q_measured"], 29, "body_q_measured")
    left = finite_vector(wire["left_hand_q"], 7, "left_hand_q")
    right = finite_vector(wire["right_hand_q"], 7, "right_hand_q")
    # The actual C++ publisher already restores absolute MuJoCo body order.
    return np.concatenate([body, left[ACTUATED_TO_MODEL], right[ACTUATED_TO_MODEL]])


def raw_recording_state_to_model(raw):
    raw = finite_vector(raw, 43, "recording observation.state")
    # Recording state is URDF order, NOT the wire's actuated hand order.
    return np.concatenate([raw[:22], raw[29:36], raw[22:29], raw[36:43]])


def fsq_quantize(token):
    token = finite_vector(token, 64, "motion token")
    clipped = np.clip(token, -.625, .625)
    return np.clip(np.round(clipped / .0625) * .0625, -.625, .625).astype(np.float32)


def model_action_to_wire(action, enforce_hand_limits=True, hand_limit_policy="reject", return_metadata=False,
                         max_clip_rad=None):
    """Model action -> wire action. In clip mode, max_clip_rad (if given) is a hard stop: a hand value
    further than max_clip_rad outside its Dex3 limit raises ContractError instead of being projected."""
    action = finite_vector(action, 78, "denormalized model action")
    left = action[64:71][MODEL_TO_ACTUATED]
    right = action[71:78][MODEL_TO_ACTUATED]
    if hand_limit_policy not in {"reject", "clip"}:
        raise ContractError("hand_limit_policy must be reject or clip")
    original = np.concatenate([left, right])
    bad_by_side = {}
    if enforce_hand_limits:
        output = []
        for name, q, low, high in [("left", left, LEFT_LO, LEFT_HI), ("right", right, RIGHT_LO, RIGHT_HI)]:
            low, high = low.astype(np.float32), high.astype(np.float32)
            bad = np.flatnonzero((q < low) | (q > high))
            bad_by_side[name] = bad.tolist()
            if len(bad) and hand_limit_policy == "reject":
                raise ContractError(f"{name} hand exceeds fixed Dex3 driver limits at {bad.tolist()}")
            output.append(np.clip(q, low, high) if hand_limit_policy == "clip" else q)
        left, right = output
    projected = np.concatenate([left, right])
    max_delta = float(np.max(np.abs(projected - original)))
    if max_clip_rad is not None and hand_limit_policy == "clip" and max_delta > max_clip_rad:
        raise ContractError(f"hand clip {max_delta:.4f} rad exceeds --max-hand-clip-rad {max_clip_rad:g} "
                            f"(outside actuated indices {bad_by_side}); stopping instead of projecting")
    metadata = {"hand_limit_policy": hand_limit_policy,
                "hand_clipped_count": int(np.sum(projected != original)),
                "hand_max_delta_rad": max_delta,
                "hand_outside_actuated_indices": bad_by_side}
    wire = np.concatenate([fsq_quantize(action[:64]), left, right]).astype(np.float32)
    return (wire, metadata) if return_metadata else wire


def pack_control_pose(action, controller_instance, client_session, action_seq, stop=False):
    """Actual SONIC v4 framing plus the package controller's lease fields."""
    instance = integer(controller_instance, "controller_instance", 1)
    session = integer(client_session, "client_session", 1)
    sequence = integer(action_seq, "action_seq")
    fields = [
        ("control_enable", np.array([not stop], dtype=np.uint8), "u8"),
        ("control_stop", np.array([stop], dtype=np.uint8), "u8"),
        ("controller_instance", np.array([instance], dtype="<i8"), "i64"),
        ("client_session", np.array([session], dtype="<i8"), "i64"),
        ("action_seq", np.array([sequence], dtype="<i8"), "i64"),
    ]
    if not stop:
        action = finite_vector(action, 78, "wire action")
        fields += [("token_state", action[:64].astype("<f4").reshape(1, 64), "f32"),
                   ("left_hand_joints", action[64:71].astype("<f4").reshape(1, 7), "f32"),
                   ("right_hand_joints", action[71:78].astype("<f4").reshape(1, 7), "f32")]
    header = {"v": 4, "endian": "le", "count": 1,
              "fields": [{"name": name, "dtype": dtype, "shape": list(value.shape)} for name, value, dtype in fields]}
    encoded = json.dumps(header, separators=(",", ":")).encode()
    if len(encoded) > HEADER_SIZE:
        raise ContractError("v4 header overflow")
    return b"pose" + encoded.ljust(HEADER_SIZE, b"\0") + b"".join(value.tobytes() for _, value, _ in fields)


@dataclass
class ExpectedIdentity:
    checkpoint_sha256: str
    stats_sha256: str
    contract_sha256: str | None = None
    contract: dict | None = None

    def __post_init__(self):
        digest(self.checkpoint_sha256, "checkpoint_sha256")
        digest(self.stats_sha256, "stats_sha256")
        if self.contract_sha256 is not None:
            digest(self.contract_sha256, "contract_sha256")
        self.contract = {**FIXED_CONTRACT, **(self.contract or {})}
        for key, expected in FIXED_CONTRACT.items():
            if self.contract.get(key) != expected:
                raise ContractError(f"bundle {key} incompatible with this client")
        for key, value in CONTROLLER_HASHES.items():
            if key in self.contract and self.contract[key] != value:
                raise ContractError(f"bundle {key} is a different controller")

    @classmethod
    def from_bundle(cls, filename):
        path = Path(filename)
        bundle = json.loads(path.read_text())
        if bundle.get("format") not in ("psi0-cart78-deployment-v1", "psi0-objmem-cart78-v1"):
            raise ContractError("unsupported bundle manifest format")
        canonical = (json.dumps(bundle["contract"], sort_keys=True, separators=(",", ":"),
                                ensure_ascii=False, allow_nan=False) + "\n").encode()
        if hashlib.sha256(canonical).hexdigest() != bundle.get("contract_sha256"):
            raise ContractError("bundle contract content does not match contract_sha256")
        normalization = path.parent / "normalization.json"
        if normalization.is_file() and hashlib.sha256(normalization.read_bytes()).hexdigest() != bundle["contract"]["stats_sha256"]:
            raise ContractError("bundle normalization.json content does not match stats_sha256")
        return cls(bundle["checkpoint"]["sha256"], bundle["contract"]["stats_sha256"],
                   bundle.get("contract_sha256"), bundle["contract"])

    def hello_expected(self):
        result = {"checkpoint_sha256": self.checkpoint_sha256, "stats_sha256": self.stats_sha256}
        if self.contract_sha256:
            result["contract_sha256"] = self.contract_sha256
        return result

    def check_deployment(self, info, execute):
        for key, expected in {**FIXED_CONTRACT, **self.hello_expected()}.items():
            if info.get(key) != expected:
                raise ContractError(f"server identity/contract mismatch: {key}")
        if type(info.get("actions_normalized")) is not bool:
            raise ContractError("actions_normalized must be a JSON boolean")
        actual_contract_sha = digest(info.get("contract_sha256"), "server contract_sha256")
        if self.contract_sha256 is None:
            self.contract_sha256 = actual_contract_sha  # Bind subsequent frames to this handshake.
        if execute and (info.get("ready") is not True or info.get("latency_ok") is not True
                        or info.get("dry_run") is not False or info.get("mock") is True or info.get("fault")):
            raise ContractError("execute requires a ready non-mock non-dry-run server within its latency budget")
        for key in ["controller_encoder_sha256", "controller_decoder_sha256"]:
            if key in info and info[key] != CONTROLLER_HASHES[key]:
                raise ContractError(f"server {key} mismatch")


def validate_guard(config, expected_instance=None):
    if config.get("controller_guard_protocol") != GUARD_PROTOCOL or config.get("controller_guard_active") is not True:
        raise ContractError("controller does not advertise the active cart78 token lease guard")
    instance_raw = config.get("controller_guard_instance")
    if not isinstance(instance_raw, str) or re.fullmatch(r"[1-9][0-9]*", instance_raw) is None:
        raise ContractError("controller guard instance must be a positive int64 decimal string")
    instance = integer(int(instance_raw), "controller_guard_instance", 1)
    if expected_instance is not None and instance != expected_instance:
        raise ContractError("controller instance changed; client must be restarted and operator re-arm")
    lease = integer(config.get("controller_guard_timeout_ms"), "controller_guard_timeout_ms", 100, 1000)
    for key, expected in CONTROLLER_HASHES.items():
        if config.get(key) != expected:
            raise ContractError(f"controller asset SHA mismatch: {key}")
    return instance, lease


class ActionGate:
    def __init__(self, identity, session_id, execute=False, max_source_age_ms=1000.0, hand_limit_policy="reject",
                 max_hand_clip_rad=None):
        self.identity = identity
        self.session_id = session_id
        self.execute = execute
        self.max_source_age_ms = max_source_age_ms
        self.hand_limit_policy = hand_limit_policy
        self.max_hand_clip_rad = max_hand_clip_rad
        self.ledger = OrderedDict()
        self.last_observation_seq = -1
        self.last_action_seq = -1
        self.last_source_seq = -1
        self.last_chunk_id = -1
        self.last_chunk_index = -1

    def observe(self, seq, sent_ns):
        seq = integer(seq, "observation_seq")
        sent_ns = integer(sent_ns, "client_sent_monotonic_ns", 1)
        if seq <= self.last_observation_seq:
            raise ContractError("observation sequence must advance")
        self.last_observation_seq = seq
        self.ledger[seq] = sent_ns
        while len(self.ledger) > 256:
            self.ledger.popitem(last=False)

    def clear(self):
        self.ledger.clear()

    def validate(self, message, received_ns=None):
        received_ns = time.monotonic_ns() if received_ns is None else received_ns
        if message.get("type") != "action" or message.get("protocol") != PROTOCOL or message.get("session_id") != self.session_id:
            raise ContractError("action envelope/session mismatch")
        for key, expected in self.identity.hello_expected().items():
            if message.get(key) != expected:
                raise ContractError(f"action model identity mismatch: {key}")
        sequence = integer(message.get("action_seq"), "action_seq")
        source_seq = integer(message.get("source_observation_seq"), "source_observation_seq")
        if sequence <= self.last_action_seq or source_seq < self.last_source_seq:
            raise ContractError("replayed/backwards action or observation sequence")
        sent = self.ledger.get(source_seq)
        echoed = integer(message.get("source_client_sent_monotonic_ns"), "source_client_sent_monotonic_ns", 1)
        if sent is None or echoed != sent:
            raise ContractError("action does not echo a known source observation")
        age_ms = (received_ns - sent) / 1e6
        if not 0 <= age_ms <= self.max_source_age_ms:
            raise ContractError(f"source observation too old: {age_ms:.1f} ms")
        elapsed = message.get("server_elapsed_since_source_observation_received_ms")
        if not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool) or not math.isfinite(elapsed) or elapsed < 0:
            raise ContractError("invalid server elapsed duration")
        rtt_upper_ms = age_ms - elapsed
        if rtt_upper_ms < -5:
            raise ContractError("inconsistent echoed timing; server elapsed exceeds total client duration")
        rtt_upper_ms = max(0, rtt_upper_ms)
        ttl_ms = integer(message.get("ttl_ms"), "ttl_ms", 1, 1000)
        if rtt_upper_ms >= ttl_ms:
            raise ContractError(f"action transit exceeds TTL: {rtt_upper_ms:.1f} >= {ttl_ms} ms")
        for key in ["generation_timestamp_ns", "issued_timestamp_ns"]:
            integer(message.get(key), key, 1)
        latency_ms = message.get("latency_ms")
        if isinstance(latency_ms, bool) or not isinstance(latency_ms, (float, int)) or not math.isfinite(latency_ms) or latency_ms < 0:
            raise ContractError("invalid model latency measurement")
        chunk_id = integer(message.get("chunk_id"), "chunk_id")
        chunk_index = integer(message.get("chunk_index"), "chunk_index", 0, 29)
        if chunk_id < self.last_chunk_id or (chunk_id == self.last_chunk_id and chunk_index <= self.last_chunk_index):
            raise ContractError("replayed/backwards chunk position")
        if self.execute and (message.get("dry_run") is not False or message.get("latency_ok") is not True):
            raise ContractError("server revoked execute readiness")
        action = finite_vector(message.get("action"), 78, "action")
        wire, hand_metrics = model_action_to_wire(action, hand_limit_policy=self.hand_limit_policy, return_metadata=True,
                                                  max_clip_rad=self.max_hand_clip_rad)
        self.last_action_seq, self.last_source_seq = sequence, source_seq
        self.last_chunk_id, self.last_chunk_index = chunk_id, chunk_index
        return {"message": message, "model_action": action, "wire_action": wire,
                "received_ns": received_ns,
                "deadline_ns": received_ns + int((ttl_ms - rtt_upper_ms) * 1e6),
                "source_age_ms": age_ms, "rtt_upper_ms": rtt_upper_ms,
                "token_clipped_dimensions": int(np.sum(np.abs(action[:64]) > .625)), **hand_metrics}

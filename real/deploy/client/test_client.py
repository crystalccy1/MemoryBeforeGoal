"""Meaningful CPU contract tests. No sockets or robot connections in this file."""
import base64
import io
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image

from core import (ACTUATED_TO_MODEL, MODEL_TO_ACTUATED, CONTROLLER_HASHES, FIXED_CONTRACT,
                  ActionGate, ContractError, ExpectedIdentity, fsq_quantize, model_action_to_wire,
                  pack_control_pose, raw_recording_state_to_model, validate_guard, wire_state_to_model)
from sources import decode_camera_message


def identity():
    return ExpectedIdentity("a" * 64, "b" * 64, "c" * 64)


def action_message(**changes):
    value = {"type": "action", "protocol": "cart78.v1", "session_id": "test-session",
             "action_seq": 0, "source_observation_seq": 0, "source_client_sent_monotonic_ns": 1_000_000_000,
             "checkpoint_sha256": "a" * 64, "stats_sha256": "b" * 64, "contract_sha256": "c" * 64,
             "action": [0.0] * 78, "chunk_id": 0, "chunk_index": 0, "ttl_ms": 100,
             "generation_timestamp_ns": 9_000_000_000, "issued_timestamp_ns": 9_050_000_000,
             "server_elapsed_since_source_observation_received_ms": 95,
             "dry_run": True, "latency_ok": True, "latency_ms": 5.0}
    value.update(changes)
    return value


class ClientContracts(unittest.TestCase):
    def test_both_hand_permutations_are_inverse_and_body_unchanged(self):
        np.testing.assert_array_equal(np.arange(7)[ACTUATED_TO_MODEL][MODEL_TO_ACTUATED], np.arange(7))
        wire = {"body_q_measured": np.arange(29), "left_hand_q": np.arange(7) + 30,
                "right_hand_q": np.arange(7) + 40}
        state = wire_state_to_model(wire)
        np.testing.assert_array_equal(state[:29], wire["body_q_measured"])
        np.testing.assert_array_equal(state[29:36], [33, 34, 35, 36, 30, 31, 32])
        action = np.r_[np.zeros(64), state[29:43]]
        native = model_action_to_wire(action, enforce_hand_limits=False)
        np.testing.assert_array_equal(native[64:71], wire["left_hand_q"])
        np.testing.assert_array_equal(native[71:78], wire["right_hand_q"])

    def test_replay_uses_urdf_layout_not_live_actuated_layout(self):
        raw = np.arange(43)
        actual = raw_recording_state_to_model(raw)
        np.testing.assert_array_equal(actual, np.r_[raw[:22], raw[29:36], raw[22:29], raw[36:43]])

    def test_fsq_and_bad_vectors(self):
        q = fsq_quantize(np.linspace(-1, 1, 64))
        self.assertTrue(np.all(q >= -.625) and np.all(q <= .625))
        np.testing.assert_array_equal(q * 16, np.round(q * 16))
        for malformed in [np.zeros(77), np.r_[np.nan, np.zeros(77)], np.r_[np.inf, np.zeros(77)], ["0"] * 78, [False] * 78]:
            with self.assertRaises(ContractError):
                model_action_to_wire(malformed)
        bad = np.zeros(78)
        bad[64] = 1  # left index0 must be <= 0 after reordering
        with self.assertRaises(ContractError):
            model_action_to_wire(bad)

    def test_v4_lease_packet_and_explicit_stop(self):
        message = pack_control_pose(np.zeros(78), 12, 34, 56)
        header = json.loads(message[4:1284].rstrip(b"\0"))
        self.assertEqual((header["v"], header["count"], header["endian"]), (4, 1, "le"))
        offset = 1284
        decoded = {}
        for item in header["fields"]:
            dtype = {"u8": "u1", "i64": "<i8", "f32": "<f4"}[item["dtype"]]
            n = int(np.prod(item["shape"]))
            value = np.frombuffer(message, dtype=dtype, count=n, offset=offset).reshape(item["shape"])
            offset += value.nbytes
            decoded[item["name"]] = value
        self.assertEqual(offset, len(message))
        self.assertEqual(decoded["controller_instance"].item(), 12)
        self.assertEqual(decoded["control_enable"].item(), 1)
        self.assertEqual(decoded["token_state"].shape, (1, 64))
        stop = pack_control_pose(None, 12, 34, 57, stop=True)
        fields = json.loads(stop[4:1284].rstrip(b"\0"))["fields"]
        self.assertEqual(len(fields), 5)
        self.assertEqual(stop[1284:1286], bytes([0, 1]))

    def test_explicit_hand_clip_uses_actuated_limits_after_permutation(self):
        action = np.zeros(78)
        action[64] = .1  # model left index0 -> native slot3, upper bound0
        action[68] = 1.5  # model left thumb0 -> native slot0, upper bound1.05
        action[77] = .1  # model right thumb2 -> native slot2, upper bound0
        wire, metrics = model_action_to_wire(action, hand_limit_policy="clip", return_metadata=True)
        self.assertEqual(wire[64 + 3], 0)
        self.assertAlmostEqual(float(wire[64]), 1.05, places=6)
        self.assertEqual(wire[71 + 2], 0)
        self.assertEqual(metrics["hand_clipped_count"], 3)
        self.assertAlmostEqual(metrics["hand_max_delta_rad"], .45, places=6)
        with self.assertRaises(ContractError):
            model_action_to_wire(action, hand_limit_policy="reject")
        action[0] = np.nan
        with self.assertRaises(ContractError):
            model_action_to_wire(action, hand_limit_policy="clip")

    def test_camera_bytes_and_legacy_base64_match_collector_color_semantics(self):
        rgb = np.zeros((16, 20, 3), np.uint8); rgb[:] = [200, 30, 10]
        for legacy in [False, True]:
            encoded = io.BytesIO()
            # OpenCV legacy encoding treats the original array as BGR.
            Image.fromarray(rgb[..., ::-1] if legacy else rgb).save(encoded, format="JPEG", quality=95)
            payload = base64.b64encode(encoded.getvalue()).decode() if legacy else encoded.getvalue()
            decoded, stamp, kind = decode_camera_message({"timestamps": {"ego_view": 123.}, "images": {"ego_view": payload}}, (16, 20, 3))
            self.assertLess(np.abs(decoded.astype(float) - rgb).mean(), 2)
        with self.assertRaises(ContractError):
            decode_camera_message({"timestamps": {"ego_view": 123.}, "images": {"ego_view": encoded.getvalue()}})

    def test_identity_and_controller_hashes_gate_execution(self):
        info = {**FIXED_CONTRACT, **identity().hello_expected(), "ready": False, "latency_ok": False, "dry_run": True}
        identity().check_deployment(info, execute=False)
        with self.assertRaises(ContractError):
            identity().check_deployment(info, execute=True)
        wrong = {**info, "stats_sha256": "d" * 64}
        with self.assertRaises(ContractError):
            identity().check_deployment(wrong, execute=False)
        config = {**CONTROLLER_HASHES, "controller_guard_protocol": "cart78.token_lease.v1",
                  "controller_guard_active": True, "controller_guard_instance": "123", "controller_guard_timeout_ms": 250}
        self.assertEqual(validate_guard(config), (123, 250))
        with self.assertRaises(ContractError):
            validate_guard(config, expected_instance=124)
        with self.assertRaises(ContractError):
            validate_guard({**config, "controller_encoder_sha256": "f" * 64})

    def test_bundle_contract_hash_and_normalization_hash(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            normalization = b'{"action_min":[0]}\n'
            (root / "normalization.json").write_bytes(normalization)
            contract = {**FIXED_CONTRACT, **CONTROLLER_HASHES,
                        "stats_sha256": hashlib.sha256(normalization).hexdigest()}
            canonical = (json.dumps(contract, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode()
            manifest = {"format": "psi0-cart78-deployment-v1", "checkpoint": {"sha256": "a" * 64},
                        "contract": contract, "contract_sha256": hashlib.sha256(canonical).hexdigest()}
            path = root / "bundle.json"; path.write_text(json.dumps(manifest))
            expected = ExpectedIdentity.from_bundle(path)
            self.assertEqual(expected.stats_sha256, hashlib.sha256(normalization).hexdigest())
            (root / "normalization.json").write_bytes(b'changed')
            with self.assertRaises(ContractError): ExpectedIdentity.from_bundle(path)
            manifest["contract"]["horizon"] = 29; path.write_text(json.dumps(manifest))
            with self.assertRaises(ContractError): ExpectedIdentity.from_bundle(path)

    def test_echo_timing_uses_client_clock_and_rejects_replay_staleness(self):
        gate = ActionGate(identity(), "test-session")
        gate.observe(0, 1_000_000_000)
        result = gate.validate(action_message(), received_ns=1_100_000_000)
        self.assertEqual(result["rtt_upper_ms"], 5)
        self.assertEqual(result["deadline_ns"], 1_195_000_000)
        # Same observation is valid for the next action in the same chunk.
        gate.validate(action_message(action_seq=1, chunk_index=1,
                                     server_elapsed_since_source_observation_received_ms=125), received_ns=1_130_000_000)
        with self.assertRaises(ContractError):
            gate.validate(action_message(action_seq=1, chunk_index=1), received_ns=1_135_000_000)
        for edits, now in [({"source_client_sent_monotonic_ns": 999}, 1_100_000_000),
                           ({"server_elapsed_since_source_observation_received_ms": 0}, 1_110_000_000),
                           ({"checkpoint_sha256": "f" * 64}, 1_100_000_000),
                           ({}, 2_100_000_000),
                           ({"server_elapsed_since_source_observation_received_ms": 200}, 1_100_000_000)]:
            new = ActionGate(identity(), "test-session"); new.observe(0, 1_000_000_000)
            with self.assertRaises(ContractError):
                new.validate(action_message(**edits), received_ns=now)
        gate.clear()
        self.assertEqual(len(gate.ledger), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

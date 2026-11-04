# cart78.token_lease.v1

The controller subscribes to `pose` messages from the client (ZMQ, native v4 format: topic bytes, a 1280-byte padded JSON
header, ordered binary fields).

| Field | dtype | shape | Meaning |
|---|---|---|---|
| control_enable | u8 | [1] | 0/1 |
| control_stop | u8 | [1] | 1 stops |
| controller_instance | i64 | [1] | the instance published in `robot_config` |
| client_session | i64 | [1] | random id of the client session |
| action_seq | i64 | [1] | strictly increasing within a session |
| token_state | f32 | [1,64] | SONIC token, in [-0.625, 0.625] on the 1/16 grid |
| left_hand_joints | f32 | [1,7] | hand targets, actuator order |
| right_hand_joints | f32 | [1,7] | hand targets, actuator order |

Each new valid message renews a lease (250 ms by default). When the lease expires, or on a stop, an invalid message or a
session change, the controller drops the external token and returns to its paused reference frame; streaming resumes only
after it is armed again on the controller.

The controller publishes its state and identity in `robot_config`: `controller_guard_protocol`,
`controller_guard_active`, `controller_guard_instance`, `controller_guard_timeout_ms` and the SHA256 of the SONIC
encoder, decoder and observation config. The client checks these before it sends actions.

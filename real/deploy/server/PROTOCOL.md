# cart78.v1

`GET /deployment-info` returns the checkpoint, statistics and contract SHA256, the dimensions (`state_dim=43`,
`action_dim=78`, `horizon=30`, `action_tick_hz=30`), the controller asset SHA256, and the readiness fields `ready`,
`latency_ok`, `dry_run`, `mock` and `fault`.

A WebSocket session on `/ws` starts with a hello and its acknowledgement:

```json
{"type":"hello","protocol":"cart78.v1","session_id":"<new UUID>","mode":"dry_run","expected":{"checkpoint_sha256":"...","stats_sha256":"...","contract_sha256":"..."}}
```

The client then sends observations:

```json
{"type":"observation","protocol":"cart78.v1","session_id":"...","observation_seq":0,"client_sent_monotonic_ns":123456789,"state_model43":[...43 values...],"image_jpeg_b64":"...","instruction":"..."}
```

and receives one action message per control tick:

```text
type="action", protocol, session_id, action_seq, source_observation_seq, source_client_sent_monotonic_ns,
action[78], actions_normalized=false, checkpoint_sha256, stats_sha256, contract_sha256,
chunk_id, chunk_index, generation_timestamp_ns, issued_timestamp_ns, ttl_ms, dry_run, latency_ms, latency_ok
```

`action_seq` and `observation_seq` strictly increase within a session; the instruction is fixed per session. Errors have
`type="error"`, `code` and `message`, and close the session.

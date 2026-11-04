# Client

`run_client.py` reads observations, sends them to the policy server, validates every returned action and, only with
`--execute`, passes it to the controller. Without `--execute` it is a dry run: it logs the actions and sends no command.

```bash
python -m pip install -r client/requirements.txt
python -m pip install -r client/requirements-replay.txt     # replay of recorded episodes

# synthetic observations (server only)
python client/run_client.py mock --server-url ws://127.0.0.1:8000/ws --bundle-manifest /path/to/bundle/bundle.json

# a recorded HuMeGo-Real episode
python client/run_client.py replay --dataset /path/to/HuMeGo-Real-TrashBag --episode 2 \
  --server-url ws://127.0.0.1:8000/ws --bundle-manifest /path/to/bundle/bundle.json

# live robot observations (add --execute to send actions to the controller)
python client/run_client.py live --server-url ws://127.0.0.1:8000/ws --bundle-manifest /path/to/bundle/bundle.json \
  --state-endpoint tcp://<robot host>:5557 --camera-endpoint tcp://<camera host>:5555 --instruction '<task instruction>'
```

The client checks that the server runs the checkpoint, statistics and contract of the bundle manifest. Each action is a
78-D vector (64-D SONIC token and 7 joint targets per hand); the client rejects non-finite or out-of-range values
(`--hand-limit-policy reject`, the default, or `clip` to project hand targets onto their limits) and stops when the action
stream stalls or an observation is too old. Each session writes `events.jsonl` and `summary.json` to `--log-dir`.

Tests (CPU and localhost only):

```bash
python -m pip install -r client/requirements-test.txt
(cd client && python test_client.py && python test_package_server.py)
```

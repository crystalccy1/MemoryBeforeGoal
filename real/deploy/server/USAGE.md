# Policy server

`serve.py` verifies a bundle made by `real/prepare_bundle.py`, loads the policy with the online SAM 3 memory and serves it
over WebSocket (`/ws`, protocol in [PROTOCOL.md](PROTOCOL.md)) with `/health` and `/deployment-info`. It talks only to the
client, never to the robot. Execution is refused unless the server is started with `--allow-execution`.

Environment: Python 3.11, PyTorch 2.7.0 with CUDA 12.8, `server/requirements-inference.txt` and
`server/requirements-sam3.txt` (`setup_env.py server` installs them).

```bash
python server/serve.py --bundle /path/to/bundles/TrashBag --verify-only      # check the bundle files
python server/serve.py --bundle /path/to/bundles/TrashBag --sam3-root /path/to/sam3-root \
    --inference-steps 6 --min-exec-horizon 12 --device cuda:0 --host 127.0.0.1 --port 8000
```

`/deployment-info` reports the checkpoint, the readiness and the measured latency. A policy call that does not return in
time produces no action.

Tests (CPU only):

```bash
python server/test_server.py
python -m server.serve --mock --port 8000
```

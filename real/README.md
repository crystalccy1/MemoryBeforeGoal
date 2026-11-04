# WholeBodyMem on a real Unitree G1

Deployment code of the real-robot experiments (paper Table V): a GPU policy server that runs WholeBodyMem with the online
SAM 3 object memory, a robot-side client, and a patch for the SONIC whole-body controller of GR00T-WholeBodyControl. The
checkpoints are `wholebodymem/real/<Task>` in [ChenyuanC/MemorybeforeGoal](https://huggingface.co/ChenyuanC/MemorybeforeGoal).

* **deploy/server/**: the policy server ([PROTOCOL.md](deploy/server/PROTOCOL.md), [USAGE.md](deploy/server/USAGE.md)).
* **deploy/client/**: reads the robot state and the head camera, talks to the server and passes the validated actions to
  the controller ([README.md](deploy/client/README.md)). Dry-run unless `--execute`.
* **deploy/controller/**: builds the SONIC controller with the patch ([README.md](deploy/controller/README.md),
  [INTERFACE.md](deploy/controller/INTERFACE.md)).
* **source/**: the Psi0 code the checkpoints were trained with. **memory/**: the SAM 3 memory pipeline and each task's
  memory spec. **gr00t/**: the GR00T-WholeBodyControl changes the controller patch applies on.
* **tasks.json**: the instruction and SAM 3 prompts of each task (TrashBag, RememberChip, CafeA, CafeB, CafeC, ChipCart).

## 1. Environments

```bash
python real/deploy/setup_env.py server --python /path/to/python3.11 --venv real/deploy/.venv-server
python real/deploy/setup_env.py client --python /path/to/python3.11 --venv real/deploy/.venv-client
real/deploy/.venv-client/bin/python -m pip install -r real/deploy/client/requirements-replay.txt
```

SAM 3 for the server, in one folder `<sam3-root>`. The SAM 3 weights are gated: request access on
[facebook/sam3](https://huggingface.co/facebook/sam3) first.

```bash
git clone https://github.com/facebookresearch/sam3 <sam3-root>/vendor/sam3-2345a4ad
git -C <sam3-root>/vendor/sam3-2345a4ad checkout 2345a4ad109ac29c569da749c91d84f10dc08c40
hf auth login
hf download facebook/sam3 sam3.pt --local-dir <sam3-root>/weights/sam3 \
    --revision 3c879f39826c281e95690f02c7821c4de09afae7
```

## 2. Bundle

```bash
real/deploy/.venv-server/bin/python real/prepare_bundle.py --task TrashBag      # -> real/bundles/TrashBag
```

## 3. Run

```bash
# policy server
real/deploy/.venv-server/bin/python real/deploy/server/serve.py --bundle real/bundles/TrashBag \
    --sam3-root <sam3-root> --inference-steps 6 --min-exec-horizon 12 --port 8000
# dry run: replay a recorded HuMeGo-Real episode through the server (no robot)
real/deploy/.venv-client/bin/python real/deploy/client/run_client.py replay --dataset /path/to/HuMeGo-Real-TrashBag \
    --episode 2 --server-url ws://127.0.0.1:8000/ws --bundle-manifest real/bundles/TrashBag/bundle.json \
    --instruction "Remember where the trash bags appeared, then pick the cups and place them there." \
    --log-dir logs/TrashBag --hand-limit-policy clip
```

On the robot, build the controller as in [controller/README.md](deploy/controller/README.md) and run the client in `live`
mode ([client/README.md](deploy/client/README.md)). `deploy.py` wraps these commands with a per-task config.

**Safety.** This code drives a humanoid robot. Keep a person at the emergency stop, start with the robot suspended or
supported, and validate every step on your own hardware.

## License

Our code is under the Apache License 2.0. `source/` is Psi0 code (Apache 2.0, see `deploy/licenses/Psi0-LICENSE.txt`);
`gr00t/` and `controller/cart78_controller.patch` modify GR00T-WholeBodyControl (code under Apache 2.0; its model weights
are under the NVIDIA Open Model License and are not included; see `deploy/licenses/SONIC-LICENSE.txt`). SAM 3 is not
included (Meta's SAM License). Details: [../NOTICE](../NOTICE).

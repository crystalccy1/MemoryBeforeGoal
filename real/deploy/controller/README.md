# Controller

The SONIC whole-body controller of GR00T-WholeBodyControl with a token lease added: it executes the policy's actions only
while new valid actions keep arriving and pauses at its reference frame otherwise. `prepare.py` copies
`gear_sonic_deploy` from a GR00T-WholeBodyControl checkout and patches the copy; `build.py` compiles it; `run.py` checks the
build and, with `--execute`, starts it.

## Source

The patch applies to `physical-superintelligence-lab/GR00T-WholeBodyControl` at commit `d4037699` with the changes in
[../../gr00t](../../gr00t):

```bash
git clone https://github.com/physical-superintelligence-lab/GR00T-WholeBodyControl && cd GR00T-WholeBodyControl
git checkout d40376994ff094787591ab76a691c50b8e131786
git apply /path/to/MemoryBeforeGoal/real/gr00t/GR00T-WBC_submodule.diff
cp -r /path/to/MemoryBeforeGoal/real/gr00t/untracked/. .
```

The SONIC model files are not included; they are the public `nvidia/GEAR-SONIC` release.

## Build and run

```bash
python3 controller/prepare.py --source /path/to/GR00T-WholeBodyControl --destination /path/to/new/controller-source
python3 controller/build.py --source /path/to/new/controller-source --tensorrt-root /path/to/TensorRT \
    --cuda-root /usr/local/cuda-12.8
python3 controller/run.py --config /path/to/controller.json --check
python3 controller/run.py --config /path/to/controller.json --execute
```

The config fields are in `config/controller.example.json`; the message format is in [INTERFACE.md](INTERFACE.md).

## Test

```bash
c++ -std=c++17 -O3 -ffast-math -fno-finite-math-only -pthread controller/tests/test_token_lease.cpp -o /tmp/cart78_token_test
/tmp/cart78_token_test
```

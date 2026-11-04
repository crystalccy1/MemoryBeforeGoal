<div align="center">

<h2>Memory before Goal: <br> Role of Pre-History Memory for Humanoid Loco-Manipulation</h2>

<p>
  <b>Chenyuan Chen</b> &nbsp;·&nbsp;
  <b>Ran Ding</b> &nbsp;·&nbsp;
  <b>Lechen Zhang</b> &nbsp;·&nbsp;
  <b>Zewen He</b> &nbsp;·&nbsp;
  <b>Yoshihiko Nakamura</b>
  <br>
  <sub>MBZUAI</sub>
</p>

<div align="center">
  <a href='https://arxiv.org/abs/2610.31166'><img src='https://img.shields.io/badge/arXiv-2610.31166-red?logo=arxiv&logoColor=darkred&labelColor=white'></a>  &ensp;
  <a href='https://memory-before-goal.github.io'><img src='https://img.shields.io/badge/project-page-green?logo=googlechrome&logoColor=green&labelColor=white'></a>  &ensp;
  <a href='https://huggingface.co/ChenyuanC/MemorybeforeGoal'><img src='https://img.shields.io/badge/HuggingFace-model-yellow?logo=huggingface&logoColor=yellow&labelColor=white'></a>  &ensp;
  <a href='https://huggingface.co/collections/ChenyuanC/humego-6ada5cbea74d4cc678c91c68'><img src='https://img.shields.io/badge/HuggingFace-dataset-yellow?logo=huggingface&logoColor=yellow&labelColor=white'></a>  &ensp;
  <a href='LICENSE'><img src='https://img.shields.io/badge/license-Apache--2.0-blue?labelColor=white'></a>  &ensp;
</div>

<br>

<img src="assets/teaser.webp" alt="A real Unitree G1 serves a cola to the seat where the cola stopped first: by the decision the three seats look the same; no memory guesses, uniform memory misses the brief cue, WholeBodyMem keeps the event frame and serves the correct seat." width="90%">

</div>

---

## 📢 News

- **2026-11-04** — Released the inference and training code for HuMeGo memory tasks and baseline tasks, and the deployment code for the real Unitree G1 ([`real/`](real/README.md)).
- **2026-11-03** — Released 51 checkpoints at [`ChenyuanC/MemorybeforeGoal`](https://huggingface.co/ChenyuanC/MemorybeforeGoal): WholeBodyMem and the Psi0 baseline of the paper's tables.
- **2026-10-23** — Released the [HuMeGo datasets](https://huggingface.co/collections/ChenyuanC/humego-6ada5cbea74d4cc678c91c68): six simulated tasks with their test scenes and six real-world tasks.

## Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Checkpoints](#checkpoints)
  - [Policy server in simulation](#policy-server)
- [Training](#training)
- [HuMeGo Benchmark](#humego)
  - [Simulation tasks](#humego-sim)
  - [Real-world tasks](#humego-real)
- [Real Deployment](#real-g1)
- [License](#license)
- [Citation](#citation)

<a id="installation"></a>
## 🔧 Installation

```bash
git clone https://github.com/crystalccy1/MemoryBeforeGoal.git && cd MemoryBeforeGoal
conda create -n wholebodymem python=3.10 -y && conda activate wholebodymem
pip install torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements/policy.txt && pip install -e ".[tracker,examples]"

# SAM 3 at a fixed code revision, with the original weights
git clone https://github.com/facebookresearch/sam3 third_party/sam3
git -C third_party/sam3 checkout 2345a4ad109ac29c569da749c91d84f10dc08c40
hf auth login      # the weights are gated, see below
hf download facebook/sam3 sam3.pt --local-dir weights/sam3 \
    --revision 3c879f39826c281e95690f02c7821c4de09afae7
```

> Inference needs an NVIDIA GPU (we used one 48 GB GPU). The SAM 3 weights are gated: request access on
> [facebook/sam3](https://huggingface.co/facebook/sam3) before downloading.

<a id="quick-start"></a>
## 🚀 Quick Start

<a id="checkpoints"></a>
### Checkpoints

| Checkpoint | Tasks | Paper |
|:---|:---|:---:|
| `wholebodymem/memory/<Task>` | BlinkPad, TouchPad, FlashPad, LitLamp, HideMug, ShellGame | Table III |
| `wholebodymem/simple/<Task>` | the 14 SIMPLE tasks, `T1_XMovePick` to `T14_BendPick` | Table I |
| `wholebodymem/real/<Task>` | TrashBag, RememberChip, CafeA, CafeB, CafeC, ChipCart (Unitree G1) | Table V |
| `baselines/psi0/{memory,simple,real}/<Task>` | Psi0 fine-tuned per task | Tables III, I, V |

All are folders of [`ChenyuanC/MemorybeforeGoal`](https://huggingface.co/ChenyuanC/MemorybeforeGoal);
`scripts/download.py` downloads one (`--method`, `--suite`, `--task`).

<a id="policy-server"></a>
### Policy server in simulation

```bash
# 1. download a checkpoint (or --suite simple --task T1_XMovePick)
python scripts/download.py --task LitLamp --out checkpoints

# 2. start the policy server with the online SAM 3 memory
wholebodymem-serve --task LitLamp --bundle checkpoints/wholebodymem/memory/LitLamp \
    --sam3-repo third_party/sam3 --sam3-checkpoint weights/sam3/sam3.pt --port 8000

# 3. in a second shell: replay a recorded episode through the server
python examples/replay_episode.py --task LitLamp --episode 0 --port 8000
```

<details>
<summary><b>Request protocol</b></summary>

<br>

- `POST /reset_rollout` at the start of an episode.
- `POST /act` with the current head-camera image, the robot state, the instruction and the frames executed since the
  previous request; it returns the next 24 actions.
- `GET /health` returns the configuration and the memory state.

The message format is in [`src/wholebodymem/server.py`](src/wholebodymem/server.py);
[`examples/replay_episode.py`](examples/replay_episode.py) is a complete client.

</details>

<a id="training"></a>
## 🏋️ Training

```bash
pip install -e ".[train]"
hf download USC-PSI-Lab/psi-model --local-dir checkpoints \
    --include "psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k/*" \
              "psi0/postpre.1by1.pad36.2601131206.ckpt.he30k/*"

# memory task (two stages)
torchrun --standalone --nproc_per_node 8 scripts/train.py configs/train/memory/LitLamp_stage1.json
torchrun --standalone --nproc_per_node 8 scripts/train.py configs/train/memory/LitLamp.json \
    model.warm_start_checkpoint=runs/memory/LitLamp-stage1/finetune_keyframe_native_object/<run>/checkpoints/ckpt_40000

# SIMPLE task
torchrun --standalone --nproc_per_node 4 scripts/train.py configs/train/simple/T1_XMovePick.json

# real robot, in the environment of real/source (Python 3.11, pip install -e real/source)
python real/source/scripts/train.py configs/train/real/TrashBag.json
```

<details>
<summary><b>Details</b></summary>

<br>

```text
configs/train/memory/<Task>_stage1.json   memory tasks, stage 1
configs/train/memory/<Task>.json          memory tasks, stage 2 = wholebodymem/memory/<Task>
configs/train/simple/<Task>.json          wholebodymem/simple/<Task>
configs/train/real/<Task>.json            wholebodymem/real/<Task>
configs/train/labels/<Task>.json          decision labels (FlashPad, HideMug, LitLamp, ShellGame)
```

- **Data**: the [HuMeGo datasets](https://huggingface.co/collections/ChenyuanC/humego-6ada5cbea74d4cc678c91c68) for the
  memory and real-robot tasks and [USC-PSI-Lab/psi-data](https://huggingface.co/datasets/USC-PSI-Lab/psi-data) for
  SIMPLE, under `data/<dataset>` (the name in the config's `data.train_repo_ids`), and each task's precomputed memory
  inputs under `memory/<suite>/<Task>/`.
- **Overrides**: any field with `key=value` after the config, e.g. `data.root_dir=/datasets`.
- **Checkpoints**: `<train.output_dir>/<train.name>/<run>/checkpoints/ckpt_<step>/`.

</details>

<a id="humego"></a>
## 📦 HuMeGo Benchmark

All datasets are in [LeRobot](https://github.com/huggingface/lerobot) v2.1 format on Hugging Face ([collection](https://huggingface.co/collections/ChenyuanC/humego-6ada5cbea74d4cc678c91c68)) and are released under CC BY 4.0.

<a id="humego-sim"></a>
### Simulation tasks

100 demonstrations per task for training and 90 test scenes per task (30 per randomization level L0 / L1 / L2).

| Task | Memory | Training | Test scenes |
|---|---|---|---|
| FlashPad | counting | [HuMeGo-Sim-FlashPad](https://huggingface.co/datasets/ChenyuanC/HuMeGo-Sim-FlashPad) | [HuMeGo-Sim-TestScenes/FlashPad](https://huggingface.co/datasets/ChenyuanC/HuMeGo-Sim-TestScenes/tree/main/FlashPad) |
| HideMug | permanence | [HuMeGo-Sim-HideMug](https://huggingface.co/datasets/ChenyuanC/HuMeGo-Sim-HideMug) | [HuMeGo-Sim-TestScenes/HideMug](https://huggingface.co/datasets/ChenyuanC/HuMeGo-Sim-TestScenes/tree/main/HideMug) |
| ShellGame | permanence | [HuMeGo-Sim-ShellGame](https://huggingface.co/datasets/ChenyuanC/HuMeGo-Sim-ShellGame) | [HuMeGo-Sim-TestScenes/ShellGame](https://huggingface.co/datasets/ChenyuanC/HuMeGo-Sim-TestScenes/tree/main/ShellGame) |
| LitLamp | reference | [HuMeGo-Sim-LitLamp](https://huggingface.co/datasets/ChenyuanC/HuMeGo-Sim-LitLamp) | [HuMeGo-Sim-TestScenes/LitLamp](https://huggingface.co/datasets/ChenyuanC/HuMeGo-Sim-TestScenes/tree/main/LitLamp) |
| BlinkPad | reference | [HuMeGo-Sim-BlinkPad](https://huggingface.co/datasets/ChenyuanC/HuMeGo-Sim-BlinkPad) | [HuMeGo-Sim-TestScenes/BlinkPad](https://huggingface.co/datasets/ChenyuanC/HuMeGo-Sim-TestScenes/tree/main/BlinkPad) |
| TouchPad | reference | [HuMeGo-Sim-TouchPad](https://huggingface.co/datasets/ChenyuanC/HuMeGo-Sim-TouchPad) | [HuMeGo-Sim-TestScenes/TouchPad](https://huggingface.co/datasets/ChenyuanC/HuMeGo-Sim-TestScenes/tree/main/TouchPad) |

<a id="humego-real"></a>
### Real-world tasks

15 demonstrations per task.

| Task | Memory | Dataset |
|---|---|---|
| Trash bag | where | [HuMeGo-Real-TrashBag](https://huggingface.co/datasets/ChenyuanC/HuMeGo-Real-TrashBag) |
| Remember-chip | what | [HuMeGo-Real-RememberChip](https://huggingface.co/datasets/ChenyuanC/HuMeGo-Real-RememberChip) |
| Café A: first come, first served | when | [HuMeGo-Real-CafeA](https://huggingface.co/datasets/ChenyuanC/HuMeGo-Real-CafeA) |
| Café B: which seat has the cola | what + where | [HuMeGo-Real-CafeB](https://huggingface.co/datasets/ChenyuanC/HuMeGo-Real-CafeB) |
| Café C: where did the cola stop first | what + where + when | [HuMeGo-Real-CafeC](https://huggingface.co/datasets/ChenyuanC/HuMeGo-Real-CafeC) |
| Chip-cart | progress | [HuMeGo-Real-ChipCart](https://huggingface.co/datasets/ChenyuanC/HuMeGo-Real-ChipCart) |

<a id="real-g1"></a>
## 🤖 Real Deployment

[`real/`](real/README.md) holds the deployment code: the policy server with the online SAM 3 memory, the robot-side
client and the patch for the SONIC whole-body controller.

```bash
python real/deploy/setup_env.py server --python /path/to/python3.11 --venv real/deploy/.venv-server
real/deploy/.venv-server/bin/python real/prepare_bundle.py --task TrashBag      # -> real/bundles/TrashBag
```

More in [real/README.md](real/README.md).

<a id="license"></a>
## 📜 License

| Part | License |
|---|---|
| Code | Apache License 2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)) |
| Checkpoints | Apache License 2.0 |
| HuMeGo datasets | CC BY 4.0 |
| SAM 3 code and weights (not included) | Meta's SAM License |
| SONIC model weights (not included) | NVIDIA Open Model License |

<a id="citation"></a>
## 📖 Citation

```bibtex
@misc{chen2026memorybeforegoal,
      title={Memory before Goal: Role of Pre-History Memory for Humanoid Loco-Manipulation},
      author={Chenyuan Chen and Ran Ding and Lechen Zhang and Zewen He and Yoshihiko Nakamura},
      year={2026},
      eprint={2610.31166},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2610.31166},
}
```

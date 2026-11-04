"""HTTP policy server for a WholeBodyMem checkpoint.

Endpoints (all JSON; arrays use the numpy encoding of ``psi.deploy.helpers``):
    POST /reset_rollout   {"session_id": str, "episode_index": int, "epoch": int (optional), "seed": int (optional)}
                          Starts a new episode: clears the real-time-chunking state and the episode memory and reseeds the
                          policy noise. Without "seed" the n-th reset of this server uses seed_base + n (the paper protocol).
    POST /act             ``RequestMessage`` (see ``examples/replay_episode.py``) -> ``ResponseMessage`` with
                          the next ``action_exec_horizon`` actions (denormalised, shape [24, 36] for the released
                          checkpoints).
    GET  /health          configuration, memory and tracker state.

Every /act uploads the head-camera frames executed since the previous request (``history.framesamp_raw_rgb``, uint8
[N, 360, 640, 3], with their episode frame indices) and ends at the current image. The first request of an episode carries
``history.reset = true`` and a single frame. Requests are served one at a time.

Decoding is Psi0 flow matching with 10 steps. With real-time chunking (``--rtc``, used for all paper results) every request
after the first conditions on the unexecuted tail of the previous chunk (inference delay = chunk size - execution horizon).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import random
import tempfile
import threading
import time
from typing import Any, Mapping

import numpy as np

LOG = logging.getLogger("wholebodymem.server")
NUM_INFERENCE_STEPS = 10
SEED_BASE = 292285   # the paper evaluation: episode n of a server uses seed SEED_BASE + n


def reset_inference_rng(seed: int, *, include_cuda: bool) -> None:
    """Reset the Python, NumPy and torch (CPU and CUDA) generators; the flow-matching noise is drawn from them."""
    import torch
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an integer in [0, 2**32 - 1]")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if include_cuda:
        torch.cuda.manual_seed_all(seed)


class PolicyServer:
    def __init__(self, *, bundle: Path, task: str, device: str = "cuda:0", rtc: bool = True,
                 action_exec_horizon: int | None = 24, seed_base: int = SEED_BASE, sam_repo: Path,
                 sam3_checkpoint: Path, tracker_python: Path | None = None, socket_dir: Path | None = None,
                 trim_past_keep: int = 16, check_sha256: bool = True, pack_precision: str = "fp32",
                 memory_log: str | None = None):
        import torch
        from psi.utils import seed_everything
        from wholebodymem.memory import make_memory_store
        from wholebodymem.policy import load_policy, load_run_config
        from wholebodymem.tasks import get_task

        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        self.device = torch.device(device)
        self.task = get_task(task)
        self.bundle = Path(bundle).expanduser().resolve(strict=True)
        bundle_meta = json.loads((self.bundle / "BUNDLE.json").read_text())
        if bundle_meta.get("task") != self.task.name:
            raise ValueError(f"bundle {self.bundle} is for task {bundle_meta.get('task')!r}, not {self.task.name!r}")
        self.bundle_meta = bundle_meta
        config = load_run_config(self.bundle)
        # As in the paper evaluation: seed with the training seed before building the model. The rollout noise seed is
        # set again at every episode reset.
        seed_everything(config.seed or 42)
        self.model = load_policy(self.bundle, config, device, check_sha256=check_sha256, pack_precision=pack_precision)
        self.config = config
        self.maxmin = config.data.transform.field
        self.model_transform = config.data.transform.model
        self.Da = config.model.action_dim
        self.Tp = config.model.action_chunk_size
        self.Ta = action_exec_horizon or config.model.action_exec_horizon
        if self.Ta > self.Tp:
            raise ValueError("action_exec_horizon is larger than the action chunk")
        self.enable_rtc = rtc
        self.rtc_max_delay = None
        if rtc:
            if not config.model.rtc:
                raise ValueError("this checkpoint was not trained for real-time chunking")
            self.rtc_max_delay = config.model.max_delay
            if self.Tp - self.Ta > self.rtc_max_delay:
                raise ValueError("action_exec_horizon is too small for the checkpoint's max_delay")
        self._socket_tmp = None
        if socket_dir is None:   # short path: AF_UNIX socket paths are limited to 107 bytes
            self._socket_tmp = tempfile.TemporaryDirectory(prefix="wbm-")
            socket_dir = Path(self._socket_tmp.name)
        self.memory = make_memory_store(self.model, config, self.device, self.task, sam_repo=sam_repo,
                                        sam3_checkpoint=sam3_checkpoint, socket_dir=socket_dir,
                                        tracker_python=tracker_python, trim_past_keep=trim_past_keep,
                                        check_weights_sha256=check_sha256, memory_log=memory_log)
        self.memory_view = config.model.framesamp_online_memory_view
        self.seed_base = int(seed_base)
        self.inference_noise_seed = self.seed_base
        self.completed_episode_resets = 0
        self._lock = threading.Lock()
        self._epochs: dict[str, int] = {}
        self._rtc_previous_actions: dict[str, np.ndarray] = {}
        self.requests = 0
        reset_inference_rng(self.inference_noise_seed, include_cuda=self.device.type == "cuda")
        LOG.info("WholeBodyMem %s ready: %s", self.task.name, json.dumps(
            {"bundle": str(self.bundle), "rtc": rtc, "Tp": self.Tp, "Ta": self.Ta, "Da": self.Da,
             "object_gates": self.model.object_gates()}))

    # -- endpoints ------------------------------------------------------------------------------
    def reset_rollout(self, payload: Mapping[str, Any]) -> dict:
        with self._lock:
            if not isinstance(payload, Mapping):
                raise TypeError("reset_rollout payload must be a mapping")
            session_id = payload.get("session_id")
            if not isinstance(session_id, str) or not session_id:
                raise ValueError("reset_rollout requires a non-empty session_id")
            if "episode_index" not in payload:
                raise ValueError("reset_rollout requires episode_index")
            epoch = payload.get("epoch")
            if epoch is not None:
                if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= self._epochs.get(session_id, -1):
                    raise ValueError("reset_rollout epoch must be an integer that increases per session")
            seed = payload.get("seed")
            seed = self.seed_base + self.completed_episode_resets if seed is None else seed
            self.memory.reset(session_id, payload["episode_index"])
            self._rtc_previous_actions.pop(session_id, None)
            if epoch is not None:
                self._epochs[session_id] = epoch
            self.inference_noise_seed = seed
            reset_inference_rng(seed, include_cuda=self.device.type == "cuda")
            ordinal = self.completed_episode_resets
            self.completed_episode_resets += 1
            LOG.info("episode reset: session=%s episode_index=%s ordinal=%d seed=%d",
                     session_id, payload["episode_index"], ordinal, seed)
            return {"status": "ok", "session_id": session_id, "epoch": epoch, "seed": seed,
                    "episode_ordinal": ordinal, "rtc_state_cleared": True}

    def act(self, payload: dict) -> dict:
        with self._lock:
            return self._act(payload)

    def _act(self, payload: dict) -> dict:
        import torch
        from PIL import Image
        from torchvision.transforms import v2
        from psi.deploy.helpers import RequestMessage, ResponseMessage
        from psi.utils import pad_to_len

        request = RequestMessage.deserialize(payload)
        images, instruction, history, state = request.image, request.instruction, request.history, request.state
        for name in ("session_id", "episode_index", "framesamp_raw_rgb", "framesamp_raw_frame_indices"):
            if name not in history:
                raise KeyError(f"request history is missing {name!r}")
        if self.memory_view not in images:
            raise KeyError(f"request images must contain the memory view {self.memory_view!r}")
        session_id = history["session_id"]
        reset = bool(history.get("reset", False))
        self.requests += 1
        if reset:   # a new episode: same noise seed as its reset_rollout, no previous chunk
            reset_inference_rng(self.inference_noise_seed, include_cuda=self.device.type == "cuda")
            self._rtc_previous_actions.pop(session_id, None)
        previous = self._rtc_previous_actions.get(session_id)
        try:
            memory = self.memory.append_and_assemble(
                session_id=session_id, episode_index=history["episode_index"], reset=reset,
                raw_frames=history["framesamp_raw_rgb"], raw_frame_indices=history["framesamp_raw_frame_indices"],
                current_rgb=images[self.memory_view], instruction=instruction)
            transform = v2.Compose([self.model_transform.resize(), self.model_transform.center_crop()])
            states = torch.from_numpy(np.asarray(state["states"]).copy())
            if self.maxmin.normalize_state:
                states = torch.from_numpy(self.maxmin.normalize_state_func(
                    pad_to_len(states.numpy(), self.maxmin.pad_state_dim, dim=1)[0])).to(self.device)
            observations = [[transform(Image.fromarray(img)) for img in images.values()]]
            batched_states = states.unsqueeze(0)
            if not self.enable_rtc or previous is None or reset:
                raw_actions = self.model.predict_action(
                    observations=observations, states=batched_states, instructions=[instruction],
                    num_inference_steps=NUM_INFERENCE_STEPS, traj2ds=None, **memory)
            else:
                prev_actions = np.concatenate([previous[None, self.Ta:, :],
                                               np.zeros((1, self.Ta, self.Da), dtype=np.float32)], axis=1)
                raw_actions = self.model.predict_action_with_training_rtc_flow(
                    observations=observations, states=batched_states, instructions=[instruction],
                    num_inference_steps=NUM_INFERENCE_STEPS, traj2ds=None,
                    prev_actions=torch.from_numpy(prev_actions).to(self.device),
                    inference_delay=self.Tp - self.Ta, max_delay=self.rtc_max_delay, **memory)
        except BaseException:
            # The memory store has poisoned itself if a frame was already consumed; keep the RTC state as it was.
            if previous is None:
                self._rtc_previous_actions.pop(session_id, None)
            else:
                self._rtc_previous_actions[session_id] = previous
            raise
        normalized = raw_actions.reshape(-1, self.Da).detach().float().cpu().numpy()   # (Tp, Da)
        self._rtc_previous_actions[session_id] = normalized.copy().astype(np.float32)
        actions = self.maxmin.denormalize(normalized)[:self.Ta]
        return ResponseMessage(actions, 0.0).serialize()

    def health(self) -> dict:
        return {
            "status": "ok", "task": self.task.name, "bundle": str(self.bundle),
            "model_sha256": self.bundle_meta.get("model_sha256"), "step": self.bundle_meta.get("step"),
            "device": str(self.device), "rtc": self.enable_rtc, "rtc_max_delay": self.rtc_max_delay,
            "action_chunk_size": self.Tp, "action_exec_horizon": self.Ta, "action_dim": self.Da,
            "num_inference_steps": NUM_INFERENCE_STEPS, "pack_precision": self.model.pack_precision,
            "object_gates": self.model.object_gates(), "seed_base": self.seed_base,
            "completed_episode_resets": self.completed_episode_resets,
            "active_noise_seed": self.inference_noise_seed, "requests": self.requests,
            "memory": self.memory.health(),
        }

    def close(self) -> None:
        self.memory.tracker.close()
        if self._socket_tmp is not None:
            self._socket_tmp.cleanup()

    # -- HTTP ----------------------------------------------------------------------------------
    def app(self):
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse

        app = FastAPI(title=f"WholeBodyMem {self.task.name}")

        def guarded(handler, payload):
            try:
                return JSONResponse(content=handler(payload))
            except Exception as exc:
                LOG.exception("request failed")
                return JSONResponse(status_code=500, content={"status": "error", "error": f"{type(exc).__name__}: {exc}"})

        @app.post("/act")
        def act(payload: dict):
            return guarded(self.act, payload)

        @app.post("/reset_rollout")
        def reset_rollout(payload: dict):
            return guarded(self.reset_rollout, payload)

        @app.get("/health")
        def health():
            return guarded(lambda _: self.health(), None)

        return app


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Serve a WholeBodyMem checkpoint over HTTP.")
    parser.add_argument("--task", required=True,
                        help="a HuMeGo memory task (BlinkPad ... ShellGame) or a SIMPLE task (T1_XMovePick ... T14_BendPick)")
    parser.add_argument("--bundle", type=Path, required=True, help="checkpoint bundle directory (scripts/download.py)")
    parser.add_argument("--sam3-checkpoint", type=Path, required=True, help="sam3.pt from facebook/sam3")
    parser.add_argument("--sam3-repo", type=Path, required=True, help="facebookresearch/sam3 checkout at 2345a4ad")
    parser.add_argument("--tracker-python", type=Path, default=None,
                        help="Python interpreter for the SAM3 tracker process (default: this one)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-rtc", action="store_true", help="plain decoding instead of real-time chunking")
    parser.add_argument("--action-exec-horizon", type=int, default=24)
    parser.add_argument("--seed-base", type=int, default=SEED_BASE)
    parser.add_argument("--trim-past-keep", type=int, default=16,
                        help="bound the tracker's GPU memory (0 = off; values >= 16 do not change the tracks)")
    parser.add_argument("--skip-sha256", action="store_true", help="skip the sha256 checks of the weights")
    parser.add_argument("--memory-log", default=None, help="append the selected memory frames of each request (jsonl)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Numerics settings of the paper evaluation (must be set before CUDA starts).
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("MKL_CBWR", "COMPATIBLE")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    server = PolicyServer(bundle=args.bundle, task=args.task, device=args.device, rtc=not args.no_rtc,
                          action_exec_horizon=args.action_exec_horizon, seed_base=args.seed_base,
                          sam_repo=args.sam3_repo, sam3_checkpoint=args.sam3_checkpoint,
                          tracker_python=args.tracker_python, trim_past_keep=args.trim_past_keep,
                          check_sha256=not args.skip_sha256, memory_log=args.memory_log)
    import uvicorn
    try:
        uvicorn.run(server.app(), host=args.host, port=args.port)
    finally:
        server.close()


if __name__ == "__main__":
    main()

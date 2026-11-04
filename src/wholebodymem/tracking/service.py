"""Online SAM3 tracker: one SAM3 text-prompt session per episode, advanced one frame at a time.

Runs as a child process of the policy server (``wholebodymem.tracker_client.TrackerClient`` starts it). It owns the frozen
SAM3 video predictor and, per episode, one ``EpisodeTracker`` that turns each executed RGB frame into exactly the per-frame
record the training tracks hold (``sam_adapter._encode_objects`` objects plus the background-compensation residuals).

Protocol: ``multiprocessing.connection`` over an AF_UNIX socket, authkey read as one hex line from stdin, pickled dict
requests handled one at a time in the main thread:
    hello | reset{session_id, episode_index} | track{session_id, episode_index, start_index,
    frames uint8[N,360,640,3] RGB, sha256[N]} | close | health | shutdown
Every reply is a dict with ``ok`` (bool) and ``op``; failures carry ``error`` and ``traceback``.

Why a separate process: SAM3 builds its predictor with a thread-local, permanent bf16 autocast and needs deterministic
kernels; keeping it in its own process keeps those settings away from the policy, and requests run on the thread that
built the predictor, as when the training tracks were made.

Numerics that must match the training tracks (checked at start and before every frame):
    SAM3 code at revision 2345a4ad (clean checkout), the facebook/sam3 ``sam3.pt`` weights, upstream CPU NMS and CPU
    connected components (the training tracks were made on ROCm, where ``sam_adapter`` selects them), TF32 off,
    deterministic algorithms (warn_only), cudnn deterministic, ``CUBLAS_WORKSPACE_CONFIG=:4096:8``, bf16 autocast on the
    tracking thread, temporal disambiguation off, detector limited to the current frame.
Frame preprocessing replays SAM3's own video loader one frame at a time (resize to 1008x1008, float16, mean/std 0.5), so
frame t is processed only after frames 0..t-1 and never sees a later frame. The declared video length N_decl (the task's
horizon + 64) is what SAM3 sees as num_frames; the detector sees t + 1 frames.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import operator
import os
from pathlib import Path
import signal
import site
import sys
import threading
import time
import traceback
import uuid

import numpy as np

import background
import sam_adapter

SCHEMA = "wholebodymem-tracker-rpc/v1"
SAM_CODE_REVISION = "2345a4ad109ac29c569da749c91d84f10dc08c40"
SAM3_WEIGHTS_SHA256 = "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
SAM3_WEIGHTS_BYTES = 3_450_062_241
FRAME_SHAPE = (360, 640, 3)          # head camera frames of the HuMeGo tasks
FPS = 50
AVAILABILITY_MODE = "synchronous_replay; no asynchronous latency simulation"   # record field of the training tracks
MIN_N_DECL = 16                      # SAM3 uses min(num_frames, 16) for its object-pointer positions
IMAGE_SIZE = 1008
IMAGE_MEAN = (0.5, 0.5, 0.5)
IMAGE_STD = (0.5, 0.5, 0.5)
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
ACCEPT_TIMEOUT_S = 900.0
TRIM_KEEP_KEYS = ("obj_ptr", "object_score_logits")
READY_PREFIX = "WHOLEBODYMEM_TRACKER_READY "
OPS = ("hello", "reset", "track", "close", "health", "shutdown")
PROMPT_SEP = "|"
# Library versions the training tracks and the paper evaluation used. Other versions run, with a warning; scipy matters most
# (it does SAM3's detector-to-track assignment).
TESTED_VERSIONS = {
    "torch": "2.7.0", "torchvision": "0.22.0", "numpy": "1.26.4", "pillow": "11.3.0",
    "opencv": "4.11.0", "pycocotools": "2.0.11", "scikit-image": "0.25.2", "timm": "1.0.17",
    "scipy": "1.15.3",
}

_PATCHED: dict = {}       # the exact CPU NMS / CC callables installed by create_predictor_cuda()


def _require(condition, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _warn(message: str) -> None:
    print(f"[wholebodymem tracker] WARNING: {message}", file=sys.stderr, flush=True)


def sha256_file(path, chunk: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def preflight_process() -> None:
    """Process-level preconditions that must hold before torch/SAM3 touch CUDA."""
    _require(threading.current_thread() is threading.main_thread(),
             "SAM3 must be built and driven from the main thread (thread-local permanent autocast)")
    _require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == CUBLAS_WORKSPACE_CONFIG,
             f"CUBLAS_WORKSPACE_CONFIG must be {CUBLAS_WORKSPACE_CONFIG}")
    _require(os.environ.get("USE_PERFLIB", "1") == "1", "the training tracks ran with SAM3 perflib enabled")
    if not sys.flags.no_user_site or site.ENABLE_USER_SITE:
        _warn("user site-packages are enabled; packages there can shadow the tested versions")
    sys.dont_write_bytecode = True


def weights_identity(checkpoint: Path, *, check_sha256: bool) -> dict:
    path = Path(checkpoint).resolve(strict=True)
    size = path.stat().st_size
    _require(size == SAM3_WEIGHTS_BYTES, f"sam3.pt has {size} bytes, expected {SAM3_WEIGHTS_BYTES}")
    if not check_sha256:
        return {"path": str(path), "bytes": size, "sha256": None}
    digest = sha256_file(path)
    _require(digest == SAM3_WEIGHTS_SHA256, f"sam3.pt sha256 {digest} != {SAM3_WEIGHTS_SHA256}")
    return {"path": str(path), "bytes": size, "sha256": digest}


def _verify_backends() -> dict:
    """Numerics switches that must match the training tracks; checked at build and before every step."""
    import torch
    nms = sys.modules["sam3.perflib.nms"]
    cc = sys.modules["sam3.perflib.connected_components"]
    perflib = sys.modules["sam3.perflib"]
    _require(threading.current_thread() is threading.main_thread(), "SAM3 step outside the main thread")
    _require(nms.generic_nms is nms.generic_nms_cpu and nms.generic_nms is _PATCHED.get("nms"),
             "NMS is not the upstream CPU implementation")
    _require(cc.connected_components is _PATCHED.get("cc"), "connected components is not the CPU wrapper")
    _require(perflib.is_enabled is True, "SAM3 perflib must be enabled")
    _require(torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.bfloat16,
             "SAM3's permanent bf16 autocast is not active on this thread")
    _require(not torch.is_grad_enabled(), "grad mode must be off")
    _require(torch.are_deterministic_algorithms_enabled()
             and torch.is_deterministic_algorithms_warn_only_enabled(),
             "deterministic algorithms must be on with warn_only=True")
    _require(torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark,
             "cudnn must be deterministic without benchmark")
    _require(not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32,
             "TF32 must stay off")
    return {
        "nms": "upstream_generic_nms_cpu",
        "connected_components": "upstream_connected_components_cpu+empty_batch_guard",
        "perflib_enabled": True,
        "autocast_cuda_dtype": "bfloat16",
        "grad_enabled": False,
        "deterministic_algorithms": True,
        "deterministic_warn_only": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "tf32_matmul": False,
        "tf32_cudnn": False,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def create_predictor_cuda(checkpoint: Path, sam_repo: Path):
    """``sam_adapter.create_predictor`` on CUDA with CPU NMS/CC, TF32 off and determinism asserted.

    Returns (predictor, backends). The caller owns the predictor and must call predictor.shutdown().
    """
    _require(threading.current_thread() is threading.main_thread(), "create_predictor_cuda needs the main thread")
    import importlib
    sam_repo = Path(sam_repo).resolve(strict=True)
    # 1. Same import as sam_adapter.create_predictor. sam3/__init__.py already imports sam3.model_builder (and runs its
    #    TF32 setup), exactly as when the training tracks were made, before the patch below.
    sys.path.insert(0, str(sam_repo))
    sam3 = importlib.import_module("sam3")
    _require(Path(sam3.__file__).resolve().is_relative_to(sam_repo),
             f"sam3 imported from {sam3.__file__}, expected {sam_repo}")
    # 2. The CPU post-processing that sam_adapter selects on ROCm, applied unconditionally: the training tracks were
    #    made on ROCm. The call sites resolve these at call time, so patching here is effective.
    importlib.import_module("skimage.measure")
    nms = importlib.import_module("sam3.perflib.nms")
    cc = importlib.import_module("sam3.perflib.connected_components")
    nms.generic_nms = nms.generic_nms_cpu
    cc.connected_components = sam_adapter._empty_batch_safe_cpu_cc(cc.connected_components_cpu)
    _PATCHED.update(nms=nms.generic_nms, cc=cc.connected_components)
    # 3. The training loader, unchanged (revision + clean-tree check, determinism, build, causal settings).
    predictor = sam_adapter.create_predictor(checkpoint, sam_repo)
    try:
        import torch
        # 4. TF32 off (sam3.model_builder turns it on for sm >= 8); determinism asserted.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        backends = _verify_backends()
        model = predictor.model
        _require(predictor.world_size == 1 and predictor.rank == 0, "tracker must run single-GPU rank 0")
        _require(model.image_size == IMAGE_SIZE and tuple(model.image_mean) == IMAGE_MEAN
                 and tuple(model.image_std) == IMAGE_STD, "unexpected SAM3 image preprocessing constants")
        _require(model.detector.forward_video_grounding_multigpu.__name__ == "current_frame_only",
                 "detector prefetch guard is not installed")
        _require(int(model.tracker.max_obj_ptrs_in_encoder) == MIN_N_DECL, "max_obj_ptrs_in_encoder drift")
        return predictor, backends
    except BaseException:
        predictor.shutdown()
        raise


def _validate_frame(rgb) -> np.ndarray:
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.shape != FRAME_SHAPE:
        raise ValueError(f"frame must be uint8 RGB {FRAME_SHAPE}, got "
                         f"{type(rgb).__name__} {getattr(rgb, 'dtype', None)} {getattr(rgb, 'shape', None)}")
    return np.ascontiguousarray(rgb)


class OnlineFrameBuffer:
    """Plain indexable stand-in for SAM3's AsyncImageFrameLoader, filled one frame at a time.

    Deliberately NOT a list/tuple/Mapping/dataclass/Tensor and without a ``to`` method, so SAM3's copy_data_to_device
    passes it through unchanged. ``len`` is N_decl.
    """

    def __init__(self, n_decl: int, *, image_size: int = IMAGE_SIZE,
                 img_mean=IMAGE_MEAN, img_std=IMAGE_STD):
        import torch
        _require(isinstance(n_decl, int) and not isinstance(n_decl, bool) and n_decl >= MIN_N_DECL,
                 f"N_decl must be an int >= {MIN_N_DECL}")
        self._n = n_decl
        self._image_size = int(image_size)
        # as sam3/model/io_utils.py
        self._img_mean = torch.tensor(img_mean, dtype=torch.float16)[:, None, None]
        self._img_std = torch.tensor(img_std, dtype=torch.float16)[:, None, None]
        self._frames: dict = {}
        self._next = 0
        self._released_below = 0

    def __len__(self) -> int:
        return self._n

    def put(self, t: int, rgb: np.ndarray) -> None:
        import torch
        import torchvision.transforms.functional as TF
        from PIL import Image
        _require(t == self._next and t < self._n, f"put({t}) but next frame is {self._next} of {self._n}")
        rgb = _validate_frame(rgb)
        # SAM3's loader opens lossless PNG frames with PIL and converts them to RGB: same pixels as Image.fromarray(rgb).
        img = Image.fromarray(rgb)
        _require(img.mode == "RGB", f"unexpected PIL mode {img.mode}")
        img = TF.resize(img, size=(self._image_size, self._image_size))
        img = TF.to_tensor(img)
        # offload_video_to_cpu=True keeps the frame on the CPU
        img = img.to(dtype=torch.float16)
        img -= self._img_mean
        img /= self._img_std
        self._frames[t] = img
        self._next = t + 1

    def __getitem__(self, index):
        i = operator.index(index)
        if not 0 <= i < self._n:
            raise IndexError(f"frame {i} outside declared range [0, {self._n})")
        if i >= self._next:
            raise RuntimeError(f"causality guard: frame {i} read before it was put (next={self._next})")
        if i < self._released_below:
            raise RuntimeError(f"frame {i} was released (retained from {self._released_below})")
        return self._frames[i]

    def release_before(self, t: int) -> None:
        """Drop frames < t (SAM3 reads only frame t per step)."""
        stop = max(self._released_below, min(int(t), self._next))
        for i in range(self._released_below, stop):
            self._frames.pop(i, None)
        self._released_below = stop


def start_online_session(predictor, rgb0: np.ndarray, n_decl: int):
    """SAM3's init_state + start_session with an OnlineFrameBuffer holding frame 0. Returns (sid, state)."""
    import torch
    rgb0 = _validate_frame(rgb0)
    model = predictor.model
    # init_state runs under torch.inference_mode, and so does start_session inside handle_request.
    with torch.inference_mode():
        buffer = OnlineFrameBuffer(n_decl, image_size=model.image_size,
                                   img_mean=model.image_mean, img_std=model.image_std)
        buffer.put(0, rgb0)        # SAM3's loader loads frame 0 at session start
        # sam3_video_inference.init_state with the buffer in place of the frame loader; its len is N_decl, and the
        # original height/width come from frame 0.
        state = {}
        state["image_size"] = model.image_size
        state["offload_state_to_cpu"] = True
        state["num_frames"] = len(buffer)
        state["orig_height"] = int(rgb0.shape[0])
        state["orig_width"] = int(rgb0.shape[1])
        state["constants"] = {}
        model._construct_initial_input_batch(state, buffer)
        state["tracker_inference_states"] = []
        state["tracker_metadata"] = {}
        state["feature_cache"] = {}
        state["cached_frame_outputs"] = {}
        state["action_history"] = []
        state["is_image_only"] = False
    _require(state["input_batch"].img_batch is buffer, "copy_data_to_device did not pass the buffer through")
    _require(state["num_frames"] == n_decl, "declared num_frames drift")
    # sam3_base_predictor.start_session
    session_id = str(uuid.uuid4())
    _require(session_id not in predictor._all_inference_states, "session id collision")
    now = time.time()
    predictor._all_inference_states[session_id] = {
        "state": state, "session_id": session_id, "start_time": now, "last_use_time": now,
    }
    return session_id, state


class EpisodeTracker:
    """One SAM3 session per prompt + one BackgroundCompensation for one episode, advanced by step(t, rgb)."""

    def __init__(self, predictor, prompt: str, namespace: str, episode_index: int, n_decl: int, *, trim_past_keep: int):
        _require(isinstance(episode_index, int) and not isinstance(episode_index, bool) and episode_index >= 0,
                 "episode_index must be a non-negative int")
        _require(isinstance(n_decl, int) and not isinstance(n_decl, bool) and n_decl >= MIN_N_DECL,
                 f"N_decl must be an int >= {MIN_N_DECL}")
        self.predictor = predictor
        self.prompt = prompt
        self.episode_index = episode_index
        self.n_decl = n_decl
        self.trim_past_keep = int(trim_past_keep)
        # one SAM3 session per prompt, object ids "<namespace>/<episode:06d>/<prompt_idx>:<id>"
        self.prompts = prompt.split(PROMPT_SEP)
        self.namespaces = [f"{namespace}/{episode_index:06d}/{i}" for i in range(len(self.prompts))]
        self.namespace = self.namespaces[0]
        # one BackgroundCompensation per episode, on the merged objects
        self.compensation = background.BackgroundCompensation()
        self.next_t = 0
        self.subs = [{"session_id": None, "buffer": None, "stream": None} for _ in self.prompts]
        self.started = None
        self.failed = False
        self.closed = False
        self.last_timing = None

    def step(self, t: int, rgb: np.ndarray) -> dict:
        import torch
        _require(not self.closed and not self.failed, "episode tracker is closed or failed; reset required")
        _require(isinstance(t, int) and not isinstance(t, bool) and t == self.next_t,
                 f"step({t}) but next frame is {self.next_t}")
        _require(t < self.n_decl, f"frame {t} exceeds declared N_decl={self.n_decl}")
        rgb = _validate_frame(rgb)
        self.failed = True                                   # cleared only when the step completes
        _verify_backends()
        t0 = time.perf_counter()
        if t == 0:
            self.started = time.monotonic()
        objects = []
        for sub, prompt, namespace in zip(self.subs, self.prompts, self.namespaces):
            if t == 0:
                sub["session_id"], state = start_online_session(self.predictor, rgb, self.n_decl)
                sub["buffer"] = state["input_batch"].img_batch
                # the request sequence of sam_adapter.track_prompt
                self.predictor.handle_request({
                    "type": "add_prompt", "session_id": sub["session_id"],
                    "frame_index": 0, "text": prompt,
                })
                # without max_frame_num_to_track (N_decl bounds it)
                sub["stream"] = self.predictor.handle_stream_request({
                    "type": "propagate_in_video", "session_id": sub["session_id"],
                    "start_frame_index": 0, "propagation_direction": "forward",
                    "evict_cached_frame_outputs": True,
                })
            else:
                sub["buffer"].put(t, rgb)
            # The permanent main-thread autocast is asserted above; being explicit is a no-op.
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                response = next(sub["stream"])
            _require(response["frame_index"] == t, f"SAM stream returned frame {response['frame_index']}, expected {t}")
            objects.extend(sam_adapter._encode_objects(response["outputs"], prompt, namespace))
        t1 = time.perf_counter()
        # the per-frame record of the training tracks
        record = {"frame_index": t, "available_frame": t,
                  "capture_time_seconds": t / FPS,
                  "availability_mode": AVAILABILITY_MODE,
                  "objects": []}
        record["objects"].extend(objects)
        record["tracking_elapsed_wall_seconds"] = time.monotonic() - self.started
        # BackgroundCompensation expects the BGR image
        record.update(self.compensation.update(np.ascontiguousarray(rgb[..., ::-1]), record["objects"]))
        t2 = time.perf_counter()
        for sub in self.subs:
            sub["buffer"].release_before(t - 1)
        if self.trim_past_keep > 0:
            for sub in self.subs:
                self._trim_past(t, sub["session_id"])
        self.next_t = t + 1
        self.last_timing = {"sam3_s": t1 - t0, "bc_s": t2 - t1}
        self.failed = False
        return record

    def _trim_past(self, t: int, session_id: str) -> None:
        """Bound the tracker's GPU memory on long episodes. Outputs of NON-conditioning frames older than t-K are reduced
        to {obj_ptr, object_score_logits} in every tracker state's output_dict and output_dict_per_obj (the per-object
        entries hold slices that keep the batched tensors alive). With memory selection off the tracker reads the mask
        memory of only the last num_maskmem-1 = 6 non-conditioning frames and object pointers of <= 16 frames, so K >= 16
        leaves every record unchanged (checked on a 1298-frame episode: identical records, peak memory 27.8 -> 4.1 GB)."""
        state = self.predictor._all_inference_states[session_id]["state"]
        for tracker_state in state.get("tracker_inference_states", []):
            for outputs in [tracker_state["output_dict"]] + list(tracker_state.get("output_dict_per_obj", {}).values()):
                past = outputs["non_cond_frame_outputs"]
                for frame in [f for f in past if f < t - self.trim_past_keep]:
                    out = past[frame]
                    if out is not None and any(k not in TRIM_KEEP_KEYS for k in out):
                        past[frame] = {k: out[k] for k in TRIM_KEEP_KEYS if k in out}

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        errors = []
        for sub in self.subs:
            try:
                if sub["stream"] is not None:
                    sub["stream"].close()
            except Exception as exc:
                errors.append(exc)
            finally:
                sub["stream"] = None
                sub["buffer"] = None
                if sub["session_id"] is not None:
                    session_id, sub["session_id"] = sub["session_id"], None
                    try:
                        self.predictor.handle_request({"type": "close_session", "session_id": session_id})
                    except Exception as exc:
                        errors.append(exc)
        if errors:
            raise errors[0]


def _stats(values) -> dict:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {"count": int(array.size), "mean": float(array.mean()), "p50": float(np.percentile(array, 50)),
            "p95": float(np.percentile(array, 95)), "max": float(array.max())}


def _session_id(request) -> str:
    value = request.get("session_id")
    _require(isinstance(value, str) and value.strip(), "session_id must be a non-empty string")
    return value


def _episode_index(request) -> int:
    value = request.get("episode_index")
    _require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
             "episode_index must be a non-negative int")
    return value


class TrackerService:
    """Request dispatcher; exactly one active episode; every failure closes and poisons it."""

    def __init__(self, predictor, identity: dict, *, prompt: str, namespace: str, n_decl: int, trim_past_keep: int):
        self.predictor = predictor
        self.identity = identity
        self.prompt = prompt
        self.namespace = namespace
        self.n_decl = n_decl
        self.trim_past_keep = trim_past_keep
        self.episode = None
        self.session_id = None
        self.episode_index = None
        self.poisoned = False
        self.last_track = None
        self.stopping = False
        self.started = time.time()
        self.episodes_started = 0
        self.frames_total = 0
        self.errors = 0
        self.last_error = None
        self.sam3_s: list = []
        self.bc_s: list = []

    def dispatch(self, request) -> dict:
        op = request.get("op") if isinstance(request, dict) else None
        try:
            _require(op in OPS, f"unknown op {op!r}")
            _require(threading.current_thread() is threading.main_thread(), "requests must run on the main thread")
            return {"ok": True, "op": op, **getattr(self, f"_op_{op}")(request)}
        except Exception as exc:
            self.errors += 1
            error = f"{type(exc).__name__}: {exc}"
            trace = traceback.format_exc()
            try:
                self._close_episode()
            except Exception as close_exc:
                error += f"; closing the episode also failed: {type(close_exc).__name__}: {close_exc}"
            self.poisoned = True
            self.last_error = error
            return {"ok": False, "op": op, "error": error, "traceback": trace, "poisoned": True}

    def _close_episode(self) -> None:
        episode, self.episode = self.episode, None
        self.last_track = None
        self.session_id = self.episode_index = None
        if episode is not None:
            episode.close()

    def _op_hello(self, request) -> dict:
        return {"identity": self.identity}

    def _op_reset(self, request) -> dict:
        session_id, episode_index = _session_id(request), _episode_index(request)
        self._close_episode()
        self.episode = EpisodeTracker(self.predictor, self.prompt, self.namespace, episode_index, self.n_decl,
                                      trim_past_keep=self.trim_past_keep)
        self.session_id, self.episode_index = session_id, episode_index
        self.poisoned = False
        self.episodes_started += 1
        return {"session_id": session_id, "episode_index": episode_index, "next_index": 0}

    def _op_track(self, request) -> dict:
        session_id, episode_index = _session_id(request), _episode_index(request)
        _require(not self.poisoned, f"tracker poisoned ({self.last_error}); reset required")
        _require(self.episode is not None and (session_id, episode_index) == (self.session_id, self.episode_index),
                 f"track for {session_id!r}/{episode_index} but active is {self.session_id!r}/{self.episode_index}")
        start = request.get("start_index")
        _require(isinstance(start, int) and not isinstance(start, bool) and start >= 0,
                 "start_index must be a non-negative int")
        frames, claimed = request.get("frames"), request.get("sha256")
        _require(isinstance(frames, np.ndarray) and frames.dtype == np.uint8 and frames.ndim == 4
                 and frames.shape[1:] == FRAME_SHAPE and len(frames) >= 1,
                 f"frames must be uint8 [N,{FRAME_SHAPE[0]},{FRAME_SHAPE[1]},3] RGB with N>=1")
        _require(isinstance(claimed, list) and len(claimed) == len(frames), "one sha256 per frame is required")
        actual = [hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest() for frame in frames]
        _require(actual == claimed, "frame bytes do not match their sha256")
        key = (session_id, episode_index, start, tuple(actual))
        if start != self.episode.next_t:
            if self.last_track is not None and self.last_track[0] == key:
                return {**self.last_track[1], "replayed": True}
            raise RuntimeError(f"start_index {start} != next frame {self.episode.next_t} and not a byte-identical replay")
        records, sam3_s, bc_s = [], [], []
        for offset, frame in enumerate(frames):
            records.append(self.episode.step(start + offset, frame))
            sam3_s.append(self.episode.last_timing["sam3_s"])
            bc_s.append(self.episode.last_timing["bc_s"])
        self.frames_total += len(records)
        self.sam3_s.extend(sam3_s)
        self.bc_s.extend(bc_s)
        reply = {"session_id": session_id, "episode_index": episode_index, "start_index": start,
                 "next_index": self.episode.next_t, "records": records,
                 "timing": {"sam3_s": sam3_s, "bc_s": bc_s}}
        self.last_track = (key, reply)
        return reply

    def _op_close(self, request) -> dict:
        self._close_episode()
        return {}

    def _op_health(self, request) -> dict:
        import torch
        return {"health": {
            "schema": SCHEMA, "pid": os.getpid(), "alive": True, "uptime_s": time.time() - self.started,
            "prompt": self.prompt, "namespace": self.namespace, "n_decl": self.n_decl,
            "session_id": self.session_id, "episode_index": self.episode_index,
            "next_index": None if self.episode is None else self.episode.next_t,
            "poisoned": self.poisoned, "episodes_started": self.episodes_started,
            "frames_total": self.frames_total, "errors": self.errors, "last_error": self.last_error,
            "timing": {"sam3_s": _stats(self.sam3_s), "bc_s": _stats(self.bc_s)},
            "gpu_memory_bytes": {"allocated": torch.cuda.memory_allocated(),
                                 "reserved": torch.cuda.memory_reserved(),
                                 "max_allocated": torch.cuda.max_memory_allocated(),
                                 "max_reserved": torch.cuda.max_memory_reserved()},
            "backends": _verify_backends(),
        }}

    def _op_shutdown(self, request) -> dict:
        self._close_episode()
        self.stopping = True
        return {}

    def shutdown_predictor(self) -> None:
        try:
            self._close_episode()
        finally:
            self.predictor.shutdown()


def _versions(strict: bool) -> dict:
    import importlib.metadata as metadata
    import cv2
    import PIL
    import skimage
    import timm
    import torch
    import torchvision
    found = {
        "torch": torch.__version__, "torchvision": torchvision.__version__, "numpy": np.__version__,
        "pillow": PIL.__version__, "opencv": cv2.__version__, "pycocotools": metadata.version("pycocotools"),
        "scikit-image": skimage.__version__, "timm": timm.__version__, "scipy": metadata.version("scipy"),
    }
    base = {name: str(value).split("+")[0] for name, value in found.items()}
    mismatched = {name: found[name] for name, value in TESTED_VERSIONS.items() if base[name] != value}
    if mismatched:
        message = f"library versions differ from the tested ones {TESTED_VERSIONS}: {mismatched}"
        _require(not strict, message)
        _warn(message)
    return {**{name: str(value) for name, value in found.items()},
            "torch_cuda": torch.version.cuda, "torch_hip": getattr(torch.version, "hip", None)}


def _identity(predictor, *, weights: dict, backends: dict, versions: dict, prompt: str, namespace: str,
              n_decl: int, trim_past_keep: int) -> dict:
    import torch
    provenance = json.loads(json.dumps(getattr(predictor, "_adapter_provenance", {})))
    _require(provenance.get("sam_code_revision") == SAM_CODE_REVISION, "SAM3 revision drift")
    _require(provenance.get("apply_temporal_disambiguation") is False, "temporal disambiguation must be off")
    return {
        "schema": SCHEMA,
        "pid": os.getpid(),
        "python": sys.version.split()[0],
        "sam3": {"revision": provenance["sam_code_revision"], "clean": True,   # create_predictor raises otherwise
                 "repo": provenance.get("sam_repo")},
        "weights": weights,
        "prompt": prompt, "namespace": namespace, "n_decl": n_decl, "fps": FPS, "frame_shape": list(FRAME_SHAPE),
        "namespace_format": "{namespace}/{episode:06d}/{prompt_idx}",
        "apply_temporal_disambiguation": False,
        "temporal_settings": provenance.get("temporal_settings"),
        "trim_past_keep": trim_past_keep,
        "versions": versions,
        "gpu": {"name": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0))},
        "backends": backends,
    }


def build_service(*, sam_repo: Path, checkpoint: Path, prompt: str, namespace: str, n_decl: int,
                  trim_past_keep: int = 16, check_weights_sha256: bool = True,
                  strict_versions: bool = False) -> TrackerService:
    """Verify the environment, load SAM3 once, and return the dispatcher."""
    preflight_process()
    _require(isinstance(n_decl, int) and n_decl >= MIN_N_DECL, f"N_decl must be an int >= {MIN_N_DECL}")
    _require(trim_past_keep == 0 or trim_past_keep >= MIN_N_DECL,
             f"trim_past_keep must be 0 (off) or >= {MIN_N_DECL}; smaller values change the tracks")
    versions = _versions(strict_versions)   # before any CUDA work
    _require(sam_adapter.SAM_CODE_REVISION == SAM_CODE_REVISION, "sam_adapter revision drift")
    weights = weights_identity(checkpoint, check_sha256=check_weights_sha256)
    predictor, backends = create_predictor_cuda(Path(weights["path"]), sam_repo)
    try:
        identity = _identity(predictor, weights=weights, backends=backends, versions=versions, prompt=prompt,
                             namespace=namespace, n_decl=n_decl, trim_past_keep=trim_past_keep)
        return TrackerService(predictor, identity, prompt=prompt, namespace=namespace, n_decl=n_decl,
                              trim_past_keep=trim_past_keep)
    except BaseException:
        predictor.shutdown()
        raise


def read_authkey(stream) -> bytes:
    """One ASCII hex line on stdin (>= 32 bytes of key), written by the parent right after spawn."""
    line = stream.readline(4096)
    key = bytes.fromhex(line.decode("ascii").strip())
    _require(len(key) >= 32, "authkey must be at least 32 bytes")
    return key


def _accept_with_timeout(listener, seconds: float):
    def expired(signum, frame):
        raise TimeoutError(f"no client connected within {seconds}s")
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return listener.accept()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def serve(conn, service: TrackerService) -> bool:
    """Main-thread request loop. Returns True on shutdown, False if the client went away."""
    while not service.stopping:
        try:
            request = conn.recv()
        except EOFError:
            return False
        conn.send(service.dispatch(request))
    return True


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--socket", type=Path, required=True, help="AF_UNIX path; created after SAM3 loads")
    parser.add_argument("--sam-repo", type=Path, required=True, help="facebookresearch/sam3 checkout at 2345a4ad")
    parser.add_argument("--checkpoint", type=Path, required=True, help="sam3.pt from facebook/sam3")
    parser.add_argument("--prompt", required=True, help="SAM3 text prompt; '|' separates several prompts")
    parser.add_argument("--namespace", required=True, help="object-id namespace of the task's training tracks")
    parser.add_argument("--n-decl", type=int, required=True)
    parser.add_argument("--trim-past-keep", type=int, default=16)
    parser.add_argument("--skip-weights-sha256", action="store_true", help="check only the size of sam3.pt")
    parser.add_argument("--strict-versions", action="store_true", help="fail instead of warn on untested versions")
    parser.add_argument("--accept-timeout", type=float, default=ACCEPT_TIMEOUT_S)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    os.umask(0o077)
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(143))
    args = parse_args(argv)
    authkey = read_authkey(sys.stdin.buffer)
    parent = os.getppid()
    socket_path = Path(os.path.abspath(args.socket))
    _require(socket_path.parent.is_dir() and not os.path.lexists(socket_path),
             f"socket directory must exist and {socket_path} must not")
    _require(len(os.fsencode(str(socket_path))) < 108, "socket path exceeds the AF_UNIX sun_path limit")
    service = build_service(sam_repo=args.sam_repo, checkpoint=args.checkpoint, prompt=args.prompt,
                            namespace=args.namespace, n_decl=args.n_decl, trim_past_keep=args.trim_past_keep,
                            check_weights_sha256=not args.skip_weights_sha256, strict_versions=args.strict_versions)
    clean = False
    try:
        _require(os.getppid() == parent, "parent exited while SAM3 was loading")
        from multiprocessing.connection import Listener
        with Listener(str(socket_path), family="AF_UNIX", authkey=authkey) as listener:
            print(READY_PREFIX + json.dumps({"pid": os.getpid(), "socket": str(socket_path), "schema": SCHEMA}),
                  flush=True)
            with _accept_with_timeout(listener, args.accept_timeout) as conn:
                clean = serve(conn, service)
    finally:
        service.shutdown_predictor()
    if not clean:
        print("wholebodymem tracker: client disconnected without shutdown", file=sys.stderr, flush=True)
    return 0 if clean else 2


if __name__ == "__main__":
    sys.exit(main())

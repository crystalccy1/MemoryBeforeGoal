"""Version-pinned, forward-only SAM 3 interface used to build the training tracks and by the online tracker.

Importing this module does not import torch or SAM 3. The adapter's fake-predictor
tests can therefore run without a GPU, a checkpoint, or SAM 3 installed.
"""
from __future__ import annotations

import importlib
from pathlib import Path
import subprocess
import sys


SAM_CODE_REVISION = "2345a4ad109ac29c569da749c91d84f10dc08c40"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def _empty_batch_safe_cpu_cc(cpu_impl):
    """Keep upstream CPU CC behavior, adding its missing zero-batch result.

    The video model calls hole filling even when no *new* objects are detected.
    Upstream CPU CC then stacks an empty list. Its output dtype is not uniformly
    int32: skimage labels boolean inputs as int32 but uint8 inputs as intp (int64
    on our 64-bit systems). Probe once on CPU per input dtype to match the actual
    installed implementation without changing any nonempty result.
    """
    output_dtypes = {}

    def connected_components(input_tensor):
        ndim = input_tensor.dim()
        valid_shape = ndim == 3 or (ndim == 4 and input_tensor.shape[1] == 1)
        if not valid_shape or input_tensor.shape[0] != 0:
            return cpu_impl(input_tensor)
        torch = importlib.import_module("torch")
        if input_tensor.dtype not in output_dtypes:
            probe = torch.zeros((1, 1, 1), dtype=input_tensor.dtype, device="cpu")
            labels, counts = cpu_impl(probe)
            output_dtypes[input_tensor.dtype] = (labels.dtype, counts.dtype)
        label_dtype, count_dtype = output_dtypes[input_tensor.dtype]
        return (torch.zeros_like(input_tensor, dtype=label_dtype),
                torch.zeros_like(input_tensor, dtype=count_dtype))

    return connected_components


def _disable_detector_prefetch(detector):
    """Limit detector-only lookahead without changing tracker sequence metadata.

    Even a one-GPU upstream detector eagerly computes the next chunk. Its output
    is not used on the current frame, but disabling the read makes the causal
    contract directly auditable and avoids charging next-frame work to this one.
    """
    if detector.world_size != 1:
        raise RuntimeError("The causal detector adapter requires exactly one GPU")
    original_forward = detector.forward_video_grounding_multigpu

    def current_frame_only(*, frame_idx, num_frames, track_in_reverse=False, **kwargs):
        if track_in_reverse:
            raise RuntimeError("Backward detector propagation is forbidden")
        if not 0 <= frame_idx < num_frames:
            raise ValueError("Detector frame index is outside the available sequence")
        return original_forward(
            frame_idx=frame_idx, num_frames=frame_idx + 1,
            track_in_reverse=False, **kwargs,
        )

    detector.forward_video_grounding_multigpu = current_frame_only


def create_predictor(checkpoint: Path, sam_repo: Path, rocm_cpu_postprocess=True):
    """Load the original SAM 3 checkpoint, frozen, on the first visible GPU.

    ROCm postprocessing explicitly selects the upstream CPU implementations for
    NMS and connected components. It does not change existing torch packages.
    The caller owns the returned predictor and must eventually call shutdown().
    """
    checkpoint = Path(checkpoint).resolve(strict=True)
    sam_repo = Path(sam_repo).resolve(strict=True)
    if not checkpoint.is_file() or not sam_repo.is_dir():
        raise ValueError("checkpoint must be a file and sam_repo a directory")
    revision = subprocess.run(
        ["git", "-C", str(sam_repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if revision != SAM_CODE_REVISION:
        raise RuntimeError(f"SAM 3 code revision mismatch: {revision} != {SAM_CODE_REVISION}")
    clean = subprocess.run(
        ["git", "-C", str(sam_repo), "diff", "--quiet", "HEAD", "--"],
        capture_output=True,
    )
    if clean.returncode != 0:
        raise RuntimeError("SAM 3 checkout has tracked edits or cannot be checked")

    # Prefer exactly the audited checkout, and reject a previously imported copy
    # from another installation instead of silently using the wrong code.
    sys.path.insert(0, str(sam_repo))
    sam3 = importlib.import_module("sam3")
    if not Path(sam3.__file__).resolve().is_relative_to(sam_repo):
        raise RuntimeError(f"SAM 3 already imported from a different checkout: {sam3.__file__}")
    torch = importlib.import_module("torch")
    torch.set_grad_enabled(False)
    if not torch.cuda.is_available():
        raise RuntimeError("SAM 3 video predictor requires an available CUDA/HIP GPU")
    # Default ROCm kernels were not bitwise reproducible: the same 64 frames tracked
    # twice gave mask IoU down to 0.981. Deterministic kernels make
    # repeated and truncated-prefix replays bit-identical, so the exact prefix check holds.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    is_rocm = bool(getattr(torch.version, "hip", None))
    cpu_postprocess = bool(is_rocm and rocm_cpu_postprocess)
    if cpu_postprocess:
        # These upstream functions transfer their working data to CPU and return
        # results on the input tensor's device. USE_PERFLIB=0 does not select them.
        importlib.import_module("skimage.measure")
        nms = importlib.import_module("sam3.perflib.nms")
        cc = importlib.import_module("sam3.perflib.connected_components")
        nms.generic_nms = nms.generic_nms_cpu
        cc.connected_components = _empty_batch_safe_cpu_cc(cc.connected_components_cpu)

    builder = importlib.import_module("sam3.model_builder")
    predictor = builder.build_sam3_video_predictor(
        checkpoint_path=str(checkpoint),
        gpus_to_use=[0],
        apply_temporal_disambiguation=False,
        async_loading_frames=True,
        compile=False,
    )
    try:
        predictor.model.eval()
        for parameter in predictor.model.parameters():
            parameter.requires_grad_(False)
        required = {
            "hotstart_delay": 0,
            "hotstart_unmatch_thresh": 0,
            "hotstart_dup_thresh": 0,
            "masklet_confirmation_enable": False,
            "recondition_every_nth_frame": 0,
            "compile_model": False,
        }
        for name, expected in required.items():
            actual = getattr(predictor.model, name, None)
            if actual != expected:
                raise RuntimeError(f"Noncausal/unpinned SAM 3 setting: {name}={actual!r}")
        if predictor.model.tracker.use_memory_selection:
            raise RuntimeError("Audited zero-delay variant must disable tracker memory selection")
        _disable_detector_prefetch(predictor.model.detector)
        predictor._adapter_provenance = {
            "adapter_schema": "sam3-causal-fs-v1",
            "sam_code_revision": revision,
            "sam_repo": str(sam_repo),
            "checkpoint": str(checkpoint),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "torch_version": str(torch.__version__),
            "hip_version": getattr(torch.version, "hip", None),
            "cuda_version": getattr(torch.version, "cuda", None),
            "rocm_cpu_postprocess_requested": bool(rocm_cpu_postprocess),
            "rocm_cpu_postprocess_enabled": cpu_postprocess,
            "nms_backend": "upstream_generic_nms_cpu" if cpu_postprocess else "upstream_dispatch",
            "connected_components_backend": "upstream_connected_components_cpu" if cpu_postprocess else "upstream_dispatch",
            "connected_components_empty_batch_guard": cpu_postprocess,
            "gpus_to_use": [0],
            "parameters_frozen": True,
            "model_eval": True,
            "grad_enabled": False,
            "deterministic_kernels": {"cudnn_benchmark": False, "cudnn_deterministic": True,
                                      "use_deterministic_algorithms": "warn_only"},
            "apply_temporal_disambiguation": False,
            "temporal_settings": required,
            "tracker_use_memory_selection": False,
            "async_loading_frames": True,
            "offload_video_to_cpu": True,
            "offload_state_to_cpu": True,
            "evict_cached_frame_outputs": True,
            "propagation_direction": "forward",
            "detector_prefetch_disabled": True,
            "detector_local_num_frames": "frame_index+1; tracker total num_frames unchanged",
            "future_host_decoding": "async loader may decode/cache future pixels; detector does not process them",
            "score_kind": "initial_detection_probability",
            "centroid_convention": "mean mask pixel centers divided by image width and height",
            "object_id_convention": "namespace:session_local_integer_id",
            "timing_semantics": "logical visual causality; runtime completion latency must be recorded separately",
            "prefix_caveat": {
                "total_num_frames_in_position_encoding": True,
                "max_obj_ptrs_in_encoder": int(predictor.model.tracker.max_obj_ptrs_in_encoder),
                "rule": "min(total_num_frames,max_obj_ptrs_in_encoder)",
                "warning": "Truncated total videos shorter than 16 frames can change positional normalization without using future pixels",
            },
        }
        return predictor
    except BaseException:
        predictor.shutdown()
        raise


def _numbered_frame_count(frames_dir: Path) -> int:
    frames = [p for p in frames_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    if not frames or any(not p.is_file() or not p.stem.isdecimal() for p in frames):
        raise ValueError("frames_dir must contain numerically named image files")
    numbers = sorted(int(p.stem) for p in frames)
    if numbers != list(range(len(numbers))):
        raise ValueError("Frame filenames must have unique contiguous zero-based indices")
    return len(frames)


def _encode_objects(outputs, prompt: str, namespace: str):
    import numpy as np
    from pycocotools import mask as mask_utils

    ids = np.asarray(outputs["out_obj_ids"])
    scores = np.asarray(outputs["out_probs"])
    boxes = np.asarray(outputs["out_boxes_xywh"])
    masks = np.asarray(outputs["out_binary_masks"])
    count = len(ids)
    if ids.ndim != 1 or scores.shape != (count,) or boxes.shape != (count, 4):
        raise ValueError("SAM output IDs, scores, and boxes have inconsistent shapes")
    if masks.ndim != 3 or masks.shape[0] != count or min(masks.shape[1:]) <= 0:
        raise ValueError("SAM masks must have shape [objects,height,width]")
    if masks.dtype != np.bool_:
        raise ValueError("SAM out_binary_masks must be boolean CPU arrays")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("SAM initial detection probabilities must be finite in [0,1]")
    if not np.isfinite(boxes).all() or np.any((boxes < -1e-6) | (boxes > 1 + 1e-6)):
        raise ValueError("SAM boxes must contain finite normalized xywh coordinates")
    if len(set(int(i) for i in ids)) != count:
        raise ValueError("SAM output repeats an object ID on one frame")

    objects = []
    for position, raw_id in enumerate(ids):
        if int(raw_id) != raw_id:
            raise ValueError("SAM object IDs must be integers")
        binary_mask = masks[position]
        rle = mask_utils.encode(np.asfortranarray(binary_mask, dtype=np.uint8))
        if isinstance(rle["counts"], bytes):
            rle["counts"] = rle["counts"].decode("ascii")
        if not isinstance(rle["counts"], str):
            raise ValueError("Expected COCO compressed RLE counts")
        rle["size"] = [int(v) for v in rle["size"]]
        ys, xs = np.nonzero(binary_mask)
        x, y, width, height = [float(v) for v in boxes[position]]
        if x + width > 1 + 1e-6 or y + height > 1 + 1e-6:
            raise ValueError("SAM normalized box extends outside the image")
        objects.append({
            "object_id": f"{namespace}:{int(raw_id)}",
            "prompt": prompt,
            "score": float(scores[position]),
            "score_kind": "initial_detection_probability",
            "relevance": 1.0,
            "bbox_xyxy": [max(0.0, x), max(0.0, y), min(1.0, x + width), min(1.0, y + height)],
            "mask_area_fraction": float(binary_mask.mean()),
            "centroid_xy": (
                [float((xs.mean() + 0.5) / binary_mask.shape[1]),
                 float((ys.mean() + 0.5) / binary_mask.shape[0])]
                if len(xs) else None
            ),
            "mask_rle": rle,
        })
    return objects


def track_prompt(predictor, frames_dir: Path, prompt: str, namespace: str,
                 max_frames: int | None = None):
    """Yield immutable-per-frame CPU records from one independent text session.

    A partial consumer must call generator.close(), e.g. via contextlib.closing,
    to release the session immediately. max_frames limits the output prefix;
    it does not change the total-frame metadata or avoid host video preloading.
    """
    frames_dir = Path(frames_dir).resolve(strict=True)
    if not frames_dir.is_dir() or not prompt.strip() or not namespace.strip():
        raise ValueError("A frame directory, nonempty prompt, and namespace are required")
    num_frames = _numbered_frame_count(frames_dir)
    if max_frames is not None and (isinstance(max_frames, bool) or
                                   not isinstance(max_frames, int) or max_frames <= 0):
        raise ValueError("max_frames must be a positive integer or None")
    expected_count = num_frames if max_frames is None else min(num_frames, max_frames)
    predictor._adapter_last_session_provenance = {
        "frames_dir": str(frames_dir), "prompt": prompt, "namespace": namespace,
        "total_num_frames": num_frames, "output_num_frames": expected_count,
        "prefix_position_normalization": min(num_frames, 16),
        "short_total_video": num_frames < 16,
        "host_memory": "all resized episode frames may be cached",
    }
    session_id = None
    stream = None
    try:
        session_id = predictor.handle_request({
            "type": "start_session", "resource_path": str(frames_dir),
            "offload_video_to_cpu": True, "offload_state_to_cpu": True,
        })["session_id"]
        predictor.handle_request({
            "type": "add_prompt", "session_id": session_id,
            "frame_index": 0, "text": prompt,
        })
        request = {
            "type": "propagate_in_video", "session_id": session_id,
            "start_frame_index": 0, "propagation_direction": "forward",
            "evict_cached_frame_outputs": True,
        }
        if max_frames is not None:
            # Upstream computes inclusive end = start + max_frame_num_to_track.
            request["max_frame_num_to_track"] = expected_count - 1
        stream = predictor.handle_stream_request(request)
        received = 0
        for response in stream:
            frame_index = response["frame_index"]
            if frame_index != received or received >= expected_count:
                raise RuntimeError(f"SAM stream is not the expected prefix: got {frame_index}, expected {received}")
            objects = _encode_objects(response["outputs"], prompt, namespace)
            received += 1
            yield {"frame_index": int(frame_index), "objects": objects}
        if received != expected_count:
            raise RuntimeError(f"SAM stream ended early: {received} of {expected_count} frames")
    finally:
        try:
            if stream is not None and hasattr(stream, "close"):
                stream.close()
        finally:
            if session_id is not None:
                predictor.handle_request({"type": "close_session", "session_id": session_id})

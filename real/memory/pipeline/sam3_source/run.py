"""Resumable, frozen SAM 3 forward tracking and one causal FS32 index row per frame."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import time

from common import canonical_sha, event, freeze_json, identity, read_jsonl, sha256, verify_identity, write_json
from selector import select_records


def validate_selection(ids, mask, frames, k=32):
    import numpy as np
    if ids.shape != (frames, k) or ids.dtype != np.int64 or mask.shape != ids.shape or mask.dtype != bool:
        raise ValueError("Selection tensor shape/dtype mismatch")
    counts = np.minimum(np.arange(frames) + 1, k)
    if not np.array_equal(mask, np.arange(k)[None] < counts[:, None]):
        raise ValueError("Selection must fill min(t+1,K) slots, with right padding")
    if (ids[~mask] != -1).any() or (ids[mask] < 0).any() or ((ids > np.arange(frames)[:, None]) & mask).any():
        raise ValueError("Future/negative selection index or invalid padding")
    if (((ids[:, 1:] - ids[:, :-1]) <= 0) & mask[:, 1:]).any():
        raise ValueError("Repeated or unsorted selection IDs")
    if not np.array_equal(ids[np.arange(frames), counts - 1], np.arange(frames)):
        raise ValueError("Every row must contain its current frame exactly once")


def decode_pngs(video, destination, frames):
    import cv2
    destination.mkdir(parents=True, exist_ok=False)
    cap = cv2.VideoCapture(str(video))
    try:
        for frame in range(frames):
            ok, image = cap.read()
            if not ok or not cv2.imwrite(str(destination / f"{frame:08d}.png"), image):
                raise ValueError(f"Video decode/write failed at frame {frame}")
    finally:
        cap.release()


class BackgroundCompensation:
    """Past-to-current affine image registration; residuals are visual proxies only."""
    def __init__(self):
        self.previous = None
        self.objects = {}

    def update(self, image, objects):
        import cv2
        import numpy as np
        from pycocotools import mask as mask_utils
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        old = self.previous
        transform, inlier_fraction = None, None
        if old is not None:
            background = np.full(old.shape, 255, np.uint8)
            for obj in self.objects.values():
                rle = dict(obj["mask_rle"])
                rle["counts"] = rle["counts"].encode("ascii")
                mask = mask_utils.decode(rle).astype(np.uint8)
                background[cv2.dilate(mask, np.ones((9, 9), np.uint8)) > 0] = 0
            points = cv2.goodFeaturesToTrack(old, 180, 0.01, 7, mask=background)
            if points is not None and len(points) >= 12:
                tracked, status, _ = cv2.calcOpticalFlowPyrLK(old, gray, points, None)
                if tracked is not None:
                    valid = status.reshape(-1).astype(bool)
                    if valid.sum() >= 12:
                        cv2.setRNGSeed(0)
                        candidate, inliers = cv2.estimateAffinePartial2D(
                            points[valid], tracked[valid], method=cv2.RANSAC, ransacReprojThreshold=2.0)
                        if candidate is not None and inliers is not None:
                            scale = float(np.linalg.norm(candidate[:, 0]))
                            inlier_fraction = float(inliers.mean())
                            if 0.75 <= scale <= 1.33 and inlier_fraction >= 0.6:
                                transform = candidate
        if transform is not None:
            height, width = gray.shape
            for obj in objects:
                previous = self.objects.get(obj["object_id"])
                if previous is None or previous.get("centroid_xy") is None or obj.get("centroid_xy") is None:
                    continue
                center = np.array([previous["centroid_xy"][0] * width,
                                   previous["centroid_xy"][1] * height, 1.0])
                predicted = transform @ center
                actual = np.array([obj["centroid_xy"][0] * width, obj["centroid_xy"][1] * height])
                obj["motion_residual"] = float(np.linalg.norm(actual - predicted) / np.hypot(width, height))
                rle_old, rle_new = dict(previous["mask_rle"]), dict(obj["mask_rle"])
                rle_old["counts"] = rle_old["counts"].encode("ascii")
                rle_new["counts"] = rle_new["counts"].encode("ascii")
                warp = cv2.warpAffine(mask_utils.decode(rle_old), transform, (width, height), flags=cv2.INTER_NEAREST) > 0
                current_mask = mask_utils.decode(rle_new) > 0
                union = np.count_nonzero(warp | current_mask)
                obj["mask_change_residual"] = float(1 - np.count_nonzero(warp & current_mask) / union) if union else 0.0
        self.previous = gray
        self.objects = {obj["object_id"]: obj for obj in objects}
        return {"background_alignment_valid": transform is not None,
                "background_inlier_fraction": inlier_fraction}


def verify_episode(folder, config_sha, expected_frames):
    import numpy as np
    from safetensors.numpy import load_file
    receipt = json.loads((folder / "COMPLETE.json").read_text())
    if receipt["config_sha256"] != config_sha or receipt["raw_frames"] != expected_frames:
        raise ValueError("Completed episode belongs to different inputs/config")
    for record in receipt["files"]:
        verify_identity(record)
    tensors = load_file(str(folder / "selection.safetensors"))
    validate_selection(tensors["selected_raw_frame_indices"], tensors["selected_frame_mask"], expected_frames)
    if receipt["tracking_frame_count"] != expected_frames:
        raise ValueError("Tracking did not cover every raw frame")
    return receipt


def process_episode(predictor, task, row, root, config, config_sha, limit=None):
    import cv2
    import numpy as np
    from safetensors.numpy import load_file, save_file
    from sam_adapter import track_prompt

    episode = row["episode_index"]
    frames = min(row["raw_frames"], limit) if limit else row["raw_frames"]
    folder = root / task["alias"] / "episodes" / f"{episode:06d}"
    if (folder / "COMPLETE.json").exists():
        return verify_episode(folder, config_sha, frames)
    folder.mkdir(parents=True, exist_ok=True)
    freeze_json(folder / "identity.json", {"config_sha256": config_sha, "task": task["task"],
                                          "episode_index": episode, "raw_frames": frames, "source": row})
    verify_identity(row["video"])
    temporary = folder / "decoded_frames"
    # Only this run's disposable decoded PNGs are ever removed.
    if temporary.exists():
        shutil.rmtree(temporary)
    started = time.monotonic()
    try:
        decode_pngs(row["video"]["path"], temporary, frames)
        records = [{"frame_index": t, "available_frame": t,
                    "capture_time_seconds": t / task["fps"],
                    "availability_mode": "synchronous_replay; no asynchronous latency simulation",
                    "objects": []} for t in range(frames)]
        prompt_statistics = []
        for prompt_idx, prompt in enumerate(task["prompts"]):
            observed = 0
            count = 0
            namespace = f"{task['alias']}/{episode:06d}/{prompt_idx}"
            for result in track_prompt(predictor, temporary, prompt, namespace, max_frames=frames):
                t = result["frame_index"]
                if t != count or t >= frames:
                    raise ValueError("Tracker returned noncontiguous/out-of-range frames")
                records[t]["objects"].extend(result["objects"])
                records[t]["tracking_elapsed_wall_seconds"] = time.monotonic() - started
                observed += bool(result["objects"])
                count += 1
                if count % 100 == 0:
                    event("tracking_progress", alias=task["alias"], episode=episode, frames=count, total=frames)
            if count != frames:
                raise ValueError(f"Incomplete tracking: {count}/{frames}")
            prompt_statistics.append({"prompt": prompt, "observed_frames": observed, "total_frames": frames})
        model_prefix_passed = False
        if limit and frames >= 32:
            # Same known prefix, physically truncated before the model sees a suffix.
            # Both lengths >=16 keep upstream object-pointer normalization identical.
            prefix_length = max(16, frames // 2)
            prefix_dir = folder / "prefix_check_frames"
            if prefix_dir.exists():
                shutil.rmtree(prefix_dir)
            prefix_dir.mkdir()
            try:
                for t in range(prefix_length):
                    os.link(temporary / f"{t:08d}.png", prefix_dir / f"{t:08d}.png")
                observed_prefix = [[] for _ in range(prefix_length)]
                for prompt_idx, prompt in enumerate(task["prompts"]):
                    namespace = f"{task['alias']}/{episode:06d}/{prompt_idx}"
                    for response in track_prompt(predictor, prefix_dir, prompt, namespace, max_frames=prefix_length):
                        observed_prefix[response["frame_index"]].extend(response["objects"])
                for t in range(prefix_length):
                    if observed_prefix[t] != records[t]["objects"]:
                        write_json(folder / "PREFIX_MISMATCH.json", {"frame": t, "prefix_length": prefix_length,
                                                                  "full": records[t]["objects"], "prefix": observed_prefix[t]})
                        raise ValueError(f"SAM model prefix output changed at frame {t}")
                model_prefix_passed = True
            finally:
                shutil.rmtree(prefix_dir)
        compensation = BackgroundCompensation()
        with (folder / "tracks.jsonl.pending").open("w") as stream:
            for t, record in enumerate(records):
                image = cv2.imread(str(temporary / f"{t:08d}.png"))
                record.update(compensation.update(image, record["objects"]))
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        os.replace(folder / "tracks.jsonl.pending", folder / "tracks.jsonl")
        result = select_records(records, max_frames=32, **config["selector"])
        ids, mask = result["selected_raw_frame_indices"], result["selected_frame_mask"]
        validate_selection(ids, mask, frames)
        # Independently replay prefixes; future evidence cannot change earlier selections.
        for length in sorted({1, min(16, frames), max(1, frames // 2), frames}):
            prefix = select_records(records[:length], max_frames=32, **config["selector"])
            if not np.array_equal(prefix["selected_raw_frame_indices"], ids[:length]):
                raise ValueError("Selector prefix replay mismatch")
        target = folder / "selection.safetensors"
        save_file({"selected_raw_frame_indices": ids, "selected_frame_mask": mask}, str(target) + ".pending",
                  metadata={"schema_version": "sam3-fs-selection/v1", "config_sha256": config_sha,
                            "episode_index": str(episode), "raw_frames": str(frames),
                            "source_video_sha256": row["video"]["sha256"]})
        os.replace(str(target) + ".pending", target)
        loaded = load_file(str(target))
        if not np.array_equal(loaded["selected_raw_frame_indices"], ids):
            raise ValueError("Serialized indices changed")
        with (folder / "selection_audit.jsonl").open("w") as stream:
            for selection in result["rows"]:
                stream.write(json.dumps(selection, sort_keys=True, allow_nan=False) + "\n")
        report = {"schema_version": "sam3-fs-episode/v1", "config_sha256": config_sha,
                  "task": task["task"], "alias": task["alias"], "episode_index": episode,
                  "raw_frames": frames, "tracking_frame_count": frames,
                  "seconds": time.monotonic() - started, "prompt_statistics": prompt_statistics,
                  "selector_summary": result["summary"], "selector_prefix_replay_passed": True,
                  "model_prefix_replay_passed": model_prefix_passed,
                  "model_causality_evidence": "source-audited single forward pass; no lookahead heuristics",
                  "quality_status": "needs_visual_review" if any(p["observed_frames"] for p in prompt_statistics) else "no_target_detected",
                  "files": [identity(folder / name) for name in ("identity.json", "tracks.jsonl", "selection.safetensors", "selection_audit.jsonl")]}
        write_json(folder / "COMPLETE.json", report)
        event("episode_complete", alias=task["alias"], episode=episode, frames=frames,
              seconds=round(report["seconds"], 2), quality=report["quality_status"])
        return report
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main(args):
    os.umask(0o077)
    root = args.run_root.resolve(strict=True)
    plan_path = root / "work/plan.json"
    plan = json.loads(plan_path.read_text())
    checkpoint = identity(args.checkpoint)
    config = {"schema_version": "sam3-fs-runtime/v1", "plan": identity(plan_path),
              "checkpoint": checkpoint, "sam_code_revision": plan["sam_code_revision"],
              "sam_model": plan["sam_model"], "max_memory_frames": 32, "include_current": True,
              "selector": plan["selector"], "tracking": plan["causality"],
              "motion": {"method": "background_affine_lk_ransac", "min_inlier_fraction": 0.6},
              "source_code": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
              "limit_frames": args.limit_frames,
              "availability_mode": "synchronous_replay_no_async_latency_simulation"}
    config_sha = canonical_sha(config)
    destination = root / ("qa/smoke" if args.limit_frames else "work/results")
    freeze_json(destination / "config.json", config)
    tasks = []
    for reference in plan["input_manifests"]:
        verify_identity(reference)
        task = json.loads(Path(reference["path"]).read_text())
        if args.task in ("all", task["alias"]):
            tasks.append(task)
    if not tasks:
        raise ValueError("No tasks match the requested scope")
    if args.episode is not None:
        missing = [t["alias"] for t in tasks if args.episode not in {r["episode_index"] for r in t["outputs"]}]
        if missing:
            raise ValueError(f"Requested episode does not exist in tasks: {missing}")
    from sam_adapter import create_predictor
    predictor = create_predictor(args.checkpoint, args.sam_repo)
    write_json(destination / "runtime_environment.json", getattr(predictor, "_adapter_provenance", {}))
    completed = []
    index, count = args.shard
    try:
        for task in tasks:
            rows = sorted(task["outputs"], key=lambda r: r["episode_index"])
            for position, row in enumerate(rows):
                if args.episode is not None and row["episode_index"] != args.episode:
                    continue
                if position % count != index:
                    continue
                completed.append(process_episode(predictor, task, row, destination, config, config_sha, args.limit_frames))
    finally:
        predictor.shutdown()
    expected = sum(len(task["outputs"]) for task in tasks)
    all_tasks = len(tasks) == len(plan["input_manifests"])
    full = (not args.limit_frames and args.episode is None and count == 1
            and all_tasks and len(completed) == expected)
    manifest = {"schema_version": "sam3-fs-selection-set/v1", "status": "complete" if full else "partial",
                "config_sha256": config_sha, "tasks": [t["task"] for t in tasks],
                "episodes": len(completed), "raw_frames": sum(r["raw_frames"] for r in completed),
                "no_target_episodes": sum(r["quality_status"] == "no_target_detected" for r in completed),
                "max_memory_frames": 32, "include_current": True, "training_rows_filtered": False,
                "contains_visual_features": False, "training_loader_registered": False,
                "outputs": completed}
    if full:
        write_json(destination / "manifest.json", manifest)
        write_json(destination / "INDICES_COMPLETE.json", {"manifest": identity(destination / "manifest.json"),
                                                           "note": "Index generation complete; perception QA and training integration are separate."})
    else:
        scope = f"{args.task}-episode-{args.episode if args.episode is not None else 'all'}"
        if count > 1:
            scope += f"-shard-{index}-of-{count}"
        write_json(destination / "scopes" / f"{scope}.json", manifest)
    event("run_complete", status=manifest["status"], episodes=len(completed), frames=manifest["raw_frames"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--sam-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", choices=["all", "bht", "bpmp", "hand", "ttg", "chair", "door", "hug", "oven", "trash"], default="all")
    parser.add_argument("--episode", type=int)
    parser.add_argument("--limit-frames", type=int)
    parser.add_argument("--shard", default="0/1", help="INDEX/COUNT over each task's sorted episodes")
    options = parser.parse_args()
    try:
        options.shard = tuple(int(v) for v in options.shard.split("/"))
        if len(options.shard) != 2 or not 0 <= options.shard[0] < options.shard[1]:
            raise ValueError
    except ValueError:
        parser.error("--shard must be INDEX/COUNT with 0 <= INDEX < COUNT")
    if options.limit_frames is not None and options.limit_frames < 1:
        parser.error("--limit-frames must be positive")
    main(options)

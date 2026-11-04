#!/usr/bin/env python3
"""Replay one recorded episode through a running WholeBodyMem server (open loop) and compare with the recorded actions.

This is the request protocol of the closed-loop evaluation, fed with recorded frames instead of a simulator: the first
request of an episode uploads frame 0 with ``reset = true``; after that the client executes ``action_exec_horizon`` (24)
actions and uploads the 24 frames it saw since the previous request. Here the frames and states come from the dataset, so
the policy never sees the consequences of its own actions; the error to the recorded actions is only a sanity check, not a
success rate.

    python examples/replay_episode.py --task LitLamp --episode 0 --port 8000

The episode comes from the task's HuMeGo dataset; for a SIMPLE task the task's zip in USC-PSI-Lab/psi-data is downloaded
and extracted to --download-dir.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid
import zipfile

import numpy as np
import requests

from psi.deploy.helpers import RequestMessage, ResponseMessage
from wholebodymem.tasks import SIMPLE_DATA_REVISION, get_task

MEMORY_VIEW = "rgb_head_stereo_left"   # the head camera; "observation.images.egocentric" in the datasets


def load_episode(root: Path, episode: int):
    import av
    import pyarrow.parquet as pq

    info = json.loads((root / "meta/info.json").read_text())
    chunk = episode // info["chunks_size"]
    table = pq.read_table(root / info["data_path"].format(episode_chunk=chunk, episode_index=episode))
    states = np.stack([np.asarray(v, dtype=np.float32) for v in table.column("states").to_pylist()])
    actions = np.stack([np.asarray(v, dtype=np.float32) for v in table.column("action").to_pylist()])
    tasks = {json.loads(line)["task_index"]: json.loads(line)["task"]
             for line in (root / "meta/tasks.jsonl").read_text().splitlines() if line.strip()}
    instruction = tasks[table.column("task_index")[0].as_py()]
    video = root / info["video_path"].format(episode_chunk=chunk, episode_index=episode,
                                             video_key="observation.images.egocentric")
    if not video.exists():   # some releases name the camera folder without the feature prefix
        video = root / f"videos/chunk-{chunk:03d}/egocentric/episode_{episode:06d}.mp4"
    with av.open(str(video)) as container:
        frames = np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])
    n = min(len(frames), len(states))
    return frames[:n], states[:n], actions[:n], instruction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", required=True)
    parser.add_argument("--dataset-dir", type=Path, default=None, help="local copy of the task's dataset (LeRobot layout)")
    parser.add_argument("--download-dir", type=Path, default=Path("data"),
                        help="where the zip of a SIMPLE task is downloaded and extracted")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--exec-horizon", type=int, default=24)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    task = get_task(args.task)
    root = args.dataset_dir
    if root is None and task.suite == "simple":
        from huggingface_hub import hf_hub_download
        archive = hf_hub_download(task.dataset, f"simple/{task.env_id}.zip", repo_type="dataset",
                                  revision=SIMPLE_DATA_REVISION, local_dir=args.download_dir)
        root = args.download_dir / task.env_id
        if not (root / "meta/info.json").is_file():
            with zipfile.ZipFile(archive) as z:
                z.extractall(args.download_dir)
    elif root is None:
        from huggingface_hub import snapshot_download
        root = Path(snapshot_download(task.dataset, repo_type="dataset", allow_patterns=[
            "meta/*", f"data/*/episode_{args.episode:06d}.parquet", f"videos/*/*/episode_{args.episode:06d}.mp4"]))
    frames, states, actions, instruction = load_episode(root, args.episode)
    if frames.shape[1:] != (360, 640, 3):
        raise SystemExit(f"expected 360x640 RGB frames, got {frames.shape[1:]}")
    url = f"http://{args.host}:{args.port}"
    session_id = f"replay-{uuid.uuid4().hex[:8]}"
    reply = requests.post(f"{url}/reset_rollout", json={"session_id": session_id, "episode_index": args.episode,
                                                        "epoch": 0}, timeout=120).json()
    print("reset:", reply)

    steps = len(frames) if args.max_steps is None else min(len(frames), args.max_steps)
    errors, previous = [], -1
    for t in range(0, steps, args.exec_horizon):
        message = RequestMessage(
            image={MEMORY_VIEW: frames[t]},
            instruction=instruction,
            history={"session_id": session_id, "episode_index": args.episode, "reset": t == 0,
                     "framesamp_raw_rgb": frames[previous + 1:t + 1],
                     "framesamp_raw_frame_indices": np.arange(previous + 1, t + 1, dtype=np.int64)},
            state={"states": states[t][None]},
            condition={}, gt_action=[], dataset_name="simple",
            timestamp=datetime.now(timezone.utc).isoformat())
        response = requests.post(f"{url}/act", json=message.serialize(), timeout=600)
        if response.status_code != 200:
            raise SystemExit(f"/act failed at frame {t}: {response.text}")
        predicted = np.asarray(ResponseMessage.deserialize(response.json()).action)
        recorded = actions[t:t + len(predicted)]
        error = float(np.abs(predicted[:len(recorded), :recorded.shape[1]] - recorded).mean())
        errors.append(error)
        print(f"frame {t:5d}: {len(predicted)} actions, mean |pred - recorded| = {error:.4f}")
        previous = t
    print(f"episode {args.episode}: {len(errors)} requests, mean error {np.mean(errors):.4f}")


if __name__ == "__main__":
    main()

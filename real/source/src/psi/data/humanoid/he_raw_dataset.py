from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
import re

class HERawDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root: str,
        num_past_frames: int = 0,
        action_chunk_size: int = 1,
        upsample_rate: int = 1,
        use_delta_actions: bool = True,
        robot_type: str = "both",
        episodes: Optional[List[int]] = None,
        force_rewrite_cache: bool = False,
        read_mp4: bool = False,
    ) -> None:
        self.data_root = Path(data_root)
        self.num_past_frames = num_past_frames
        self.action_chunk_size = action_chunk_size
        self.upsample_rate = upsample_rate
        self.use_delta_actions = use_delta_actions
        self.robot_type = robot_type
        self.episodes_filter = set(episodes) if episodes is not None else None
        self.read_mp4 = read_mp4

        with open(f"{data_root}/task_description_dict.json", "r") as f:
            self.task_description_dict = json.load(f)

        file_suffix = f"_{self.robot_type}" if self.robot_type != "both" else ""
        if force_rewrite_cache or not os.path.exists(self.data_root / f"episode_cache{file_suffix}.json"):
            self.episode_list, self.episodes_lens = self._load_episode_list()

            def to_rel_path(eps_list: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
                import os
                root_dir_with_slash = str(self.data_root) + os.sep
                for ep in eps_list:
                    ep["episode_path"] = str(ep["episode_path"]).replace(root_dir_with_slash, "")
                    ep["data_path"] = str(ep["data_path"]).replace(root_dir_with_slash, "")
                return eps_list

            cache_data = {
                "episode_list": to_rel_path(self.episode_list),
                "episodes_lens": self.episodes_lens,
            }
            with open(self.data_root / f"episode_cache{file_suffix}.json", "w") as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)

        else:
            with open(self.data_root / f"episode_cache{file_suffix}.json", "r") as f:
                cache_data = json.load(f)
            self.episode_list = cache_data["episode_list"]
            self.episodes_lens = cache_data["episodes_lens"]

        self.cumsum_lens = np.cumsum(self.episodes_lens)

    def _load_episode_list(self) -> Tuple[List[Dict[str, Any]], List[int]]:
        episode_list: List[Dict[str, Any]] = []
        episode_idx = 0
        for cat_dir in tqdm(sorted([p for p in self.data_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower())):
            for task_dir in tqdm(sorted([p for p in cat_dir.iterdir() if p.is_dir()], key=lambda p: p.name.lower()), leave=False):
                task_desc = self.task_description_dict.get(task_dir.name, "")
                ep_dirs = tqdm(sorted([p for p in task_dir.iterdir() if p.is_dir() and p.name.startswith("episode_")]), leave=False)
                for ep_dir in ep_dirs:
                    if self.episodes_filter is not None and episode_idx not in self.episodes_filter:
                        episode_idx += 1
                        continue
                    data_path = ep_dir / "data.json"
                    if not data_path.exists():
                        episode_idx += 1
                        continue
                    data_list = self._load_data_json(data_path)
                    if not data_list:
                        episode_idx += 1
                        continue
                    ep_robot_type = self._detect_robot_type(data_list)
                    if self.robot_type != "both" and ep_robot_type != self.robot_type:
                        episode_idx += 1
                        continue
                    episode_list.append(
                        {
                            "episode_index": episode_idx,
                            "episode_path": ep_dir,
                            "data_path": data_path,
                            "len_episode": len(data_list),
                            "instruction": task_desc,
                        }
                    )
                    episode_idx += 1
        episodes_lens = [e["len_episode"] for e in episode_list]
        return episode_list, episodes_lens

    @staticmethod
    def _detect_robot_type(data_list: List[Dict[str, Any]]) -> str:
        if not data_list:
            return "h1"
        for frame in data_list:
            st = frame.get("states", {})
            if isinstance(st, dict) and "robot_type" in st:
                try:
                    return str(st["robot_type"]).lower()
                except Exception:
                    return "h1"
            if "robot_type" in frame:
                try:
                    return str(frame["robot_type"]).lower()
                except Exception:
                    return "h1"
        return "h1"

    # @staticmethod
    # @lru_cache(maxsize=128)
    def _load_data_json(self, path: Path) -> List[Dict[str, Any]]:
        with open(os.path.join(self.data_root, path), "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected list in {path}")
        return data

    def _locate_frame(self, idx: int) -> Tuple[int, int]:
        episode_id = int(np.searchsorted(self.cumsum_lens, idx, side="right"))
        if episode_id == 0:
            frame_id = idx
        else:
            frame_id = idx - int(self.cumsum_lens[episode_id - 1])
        return episode_id, frame_id

    def __len__(self) -> int:
        return int(np.sum(self.episodes_lens))

    def _load_image(self, episode_path: Path, rel_path: str) -> torch.Tensor:
        img_path = self.data_root / episode_path / rel_path
        if self.read_mp4:
            match = re.search(r'frame_(\d+)', rel_path)
            assert match is not None
            frame_idx = int(match.group(1))
            ... # TODO

        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Missing image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return image # (3, H, W)

    def _flatten_action(self, action_dict: Dict[str, Any]) -> Dict[str, np.ndarray]:
        flat: Dict[str, np.ndarray] = {}

        def walk(prefix: str, node: Any) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    walk(f"{prefix}.{key}" if prefix else key, value)
                return
            arr = np.asarray(node, dtype=np.float32)
            flat[f"action.{prefix}"] = arr

        walk("", action_dict)
        return flat

    def _pad_index(self, idx: int, length: int) -> int:
        if idx < 0:
            return 0
        if idx >= length:
            return length - 1
        return idx

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        episode_id, frame_id = self._locate_frame(idx)
        ep_info = self.episode_list[episode_id]
        data_list = self._load_data_json(ep_info["data_path"])
        length = ep_info["len_episode"]

        obs_indices = [
            self._pad_index(frame_id - (self.num_past_frames + t) * self.upsample_rate, length)
            for t in range(self.num_past_frames + 1)
        ]
        action_len = self.action_chunk_size + (1 if self.use_delta_actions else 0)
        action_indices = [
            self._pad_index(frame_id + t * self.upsample_rate, length)
            for t in range(action_len)
        ]

        # Compute masks for padding (1 = real, 0 = padded)
        obs_mask = [
            int(0 <= frame_id - (self.num_past_frames + t) * self.upsample_rate < length)
            for t in range(self.num_past_frames + 1)
        ]
        action_mask = [
            int(0 <= frame_id + t * self.upsample_rate < length)
            for t in range(action_len)
        ]

        images = []
        hand_states = []
        arm_states = []

        for i in obs_indices:
            frame = data_list[i]
            images.append(self._load_image(ep_info["episode_path"], frame["image"]))
            states = frame.get("states", {}) or {}
            hand_states.append(np.asarray(states.get("hand_state", []), dtype=np.float32))
            arm_states.append(np.asarray(states.get("arm_state", []), dtype=np.float32))

        action_sequences: Dict[str, List[np.ndarray]] = {}
        for i in action_indices:
            frame = data_list[i]
            action_dict = frame.get("action", {}) or {}
            flat = self._flatten_action(action_dict)
            for key, value in flat.items():
                action_sequences.setdefault(key, []).append(value)

        action_arrays = {key: np.stack(vals, axis=0) for key, vals in action_sequences.items()}
        instruction = ep_info["instruction"]
        if not instruction:
            # Fallback to episode directory name
            instruction = ep_info["episode_path"].split("/")[1].replace("_", " ").lower()
        return {
            "observation.images.egocentric": torch.stack(images, dim=0),
            "observation.hand_joints": np.stack(hand_states, axis=0),
            "observation.arm_joints": np.stack(arm_states, axis=0),
            **action_arrays,
            "episode_index": ep_info["episode_index"],
            "frame_id": frame_id,
            "task": instruction,
            "dataset_name": "humanoid-everyday",
            "obs_mask": np.array(obs_mask, dtype=np.float32),
            "action_mask": np.array(action_mask, dtype=np.float32),
        }

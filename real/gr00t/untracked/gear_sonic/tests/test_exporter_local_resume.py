"""Offline recording regression tests. Never connect to a robot or camera.

Run: .venv_data_collection/bin/python -m unittest discover -s gear_sonic/tests
"""

from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import numpy as np

from gear_sonic.data.exporter import Gr00tDataExporter


class LocalResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "dataset"
        self.features = {
            "observation.state": {"dtype": "float32", "shape": (2,), "names": None},
            "action": {"dtype": "float32", "shape": (2,), "names": None},
            "observation.images.ego_view": {
                "dtype": "video", "shape": (16, 16, 3),
                "names": ["height", "width", "channel"],
            },
        }
        self.writers = []
        self.addCleanup(lambda: [writer.stop_video_writers() for writer in self.writers])
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.connect = stack.enter_context(patch.object(
            socket.socket, "connect", side_effect=AssertionError("Network is forbidden")
        ))
        self.hub = stack.enter_context(patch(
            "lerobot.common.datasets.utils.get_repo_versions",
            side_effect=AssertionError("Hub fallback is forbidden"),
        ))

    def tearDown(self):
        self.connect.assert_not_called()
        self.hub.assert_not_called()

    def create(self):
        writer = Gr00tDataExporter.create(
            save_root=self.root, fps=30, features=self.features,
            modality_config={key: {} for key in ("state", "action", "video", "annotation")},
            task="Offline test task.",
        )
        self.writers.append(writer)
        return writer

    def hashes(self):
        return {
            str(path.relative_to(self.root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in self.root.rglob("*") if path.is_file()
        }

    def episode(self, writer, discarded=False):
        writer.begin_episode()
        for index in range(3):
            writer.add_frame({
                "observation.state": np.array([index, index + 1], dtype=np.float32),
                "action": np.array([index + 2, index + 3], dtype=np.float32),
                "observation.images.ego_view": np.full((16, 16, 3), index * 60, np.uint8),
            })
        writer.save_episode(discarded=discarded)

    def test_restart_before_first_episode_is_read_only(self):
        first = self.create()
        first.stop_video_writers()
        self.assertFalse((self.root / "meta/tasks.jsonl").exists())
        before = self.hashes()
        resumed = self.create()
        self.assertEqual(resumed.meta.total_episodes, 0)
        self.assertEqual(resumed.episode_buffer["episode_index"], 0)
        self.assertEqual(resumed.meta.tasks, {})
        self.assertEqual(self.hashes(), before)
        self.episode(resumed)
        self.assertEqual(resumed.meta.total_episodes, 1)

    def test_resume_preserves_saved_and_discarded_then_appends(self):
        first = self.create()
        self.episode(first)
        self.episode(first, discarded=True)
        first.stop_video_writers()
        before = self.hashes()
        resumed = self.create()
        self.assertEqual(resumed.meta.total_episodes, 2)
        self.assertEqual(resumed.meta.total_frames, 6)
        self.assertEqual(resumed.meta.info["discarded_episode_indices"], [1])
        self.assertEqual(resumed.episode_buffer["episode_index"], 2)
        self.assertEqual(self.hashes(), before)
        self.episode(resumed)
        self.assertEqual(resumed.meta.total_episodes, 3)
        self.assertEqual(resumed.meta.total_frames, 9)
        self.assertEqual(resumed.meta.info["discarded_episode_indices"], [1])
        after = self.hashes()
        for path, digest in before.items():
            if path.startswith(("data/", "videos/", "meta/recording_outcomes/")):
                self.assertEqual(after[path], digest)

    def test_nonempty_missing_index_fails_locally_without_writes(self):
        first = self.create()
        self.episode(first)
        first.stop_video_writers()
        (self.root / "meta/tasks.jsonl").unlink()
        before = self.hashes()
        with self.assertRaisesRegex(ValueError, "Failed to resume local dataset"):
            self.create()
        self.assertEqual(self.hashes(), before)

    def test_empty_metadata_with_unindexed_data_is_rejected(self):
        self.create().stop_video_writers()
        orphan = self.root / "data/chunk-000/episode_000000.parquet"
        orphan.parent.mkdir(parents=True)
        orphan.write_bytes(b"unfinalized recording")
        before = self.hashes()
        with self.assertRaisesRegex(ValueError, "conflicts with existing recording data"):
            self.create()
        self.assertEqual(self.hashes(), before)

    def test_unfinished_outcome_still_requires_recovery(self):
        self.create().stop_video_writers()
        outcome = self.root / "meta/recording_outcomes/episode_000000.json"
        outcome.parent.mkdir(parents=True)
        outcome.write_text(json.dumps({"state": "failed"}))
        before = self.hashes()
        with self.assertRaisesRegex(RuntimeError, "Unfinished recording needs recovery"):
            self.create()
        self.assertEqual(self.hashes(), before)


if __name__ == "__main__":
    unittest.main()

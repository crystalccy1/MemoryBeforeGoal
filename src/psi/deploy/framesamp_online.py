from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from psi.data.lerobot.framesamp_features import (
    assemble_memory_tokens,
    even_sample_positions,
)


@dataclass
class _EpisodeBuffer:
    episode_index: int
    features: list[torch.Tensor] = field(default_factory=list)
    requests: int = 0


class FrameSampOnlineSessionStore:
    """Episode-local raw-frame feature buffers keyed by explicit client session.

    A session cannot be used before a reset request. Raw indices must be
    contiguous, so a policy client that skips an execution frame fails closed
    instead of silently changing the FrameSamp clock.
    """

    SCHEMA_VERSION = "psi0-framesamp-online-session/v1"

    def __init__(
        self,
        producer: Any,
        *,
        max_memory_frames: int = 32,
        tokens_per_frame: int = 16,
        feature_dim: int = 2048,
        max_sessions: int = 8,
    ) -> None:
        if max_sessions <= 0:
            raise ValueError("max_sessions must be positive")
        self.producer = producer
        self.max_memory_frames = int(max_memory_frames)
        self.tokens_per_frame = int(tokens_per_frame)
        self.feature_dim = int(feature_dim)
        self.max_sessions = int(max_sessions)
        self._sessions: OrderedDict[str, _EpisodeBuffer] = OrderedDict()
        self.total_encoded_frames = 0

    @staticmethod
    def _validate_session_id(session_id: Any) -> str:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("FrameSamp requests require a non-empty session_id")
        if len(session_id) > 128:
            raise ValueError("FrameSamp session_id is too long")
        return session_id

    @staticmethod
    def _validate_episode_index(episode_index: Any) -> int:
        if isinstance(episode_index, bool) or not isinstance(
            episode_index, (int, np.integer)
        ):
            raise TypeError("FrameSamp episode_index must be an integer")
        return int(episode_index)

    def reset(self, session_id: Any, episode_index: Any) -> None:
        session_id = self._validate_session_id(session_id)
        episode_index = self._validate_episode_index(episode_index)
        self._sessions[session_id] = _EpisodeBuffer(episode_index=episode_index)
        self._sessions.move_to_end(session_id)
        while len(self._sessions) > self.max_sessions:
            self._sessions.popitem(last=False)

    def drop(self, session_id: Any) -> bool:
        session_id = self._validate_session_id(session_id)
        return self._sessions.pop(session_id, None) is not None

    def snapshot(self, session_id: Any) -> _EpisodeBuffer | None:
        """Capture a cheap rollback point before a serialized request."""

        session_id = self._validate_session_id(session_id)
        state = self._sessions.get(session_id)
        if state is None:
            return None
        return _EpisodeBuffer(
            episode_index=state.episode_index,
            features=list(state.features),
            requests=state.requests,
        )

    def restore(self, session_id: Any, snapshot: _EpisodeBuffer | None) -> None:
        session_id = self._validate_session_id(session_id)
        if snapshot is None:
            self._sessions.pop(session_id, None)
        else:
            self._sessions[session_id] = snapshot
            self._sessions.move_to_end(session_id)

    def append_and_assemble(
        self,
        *,
        session_id: Any,
        episode_index: Any,
        reset: bool,
        raw_frames: Any,
        raw_frame_indices: Any,
        current_rgb: Any,
        instruction: str,
    ) -> dict[str, torch.Tensor]:
        session_id = self._validate_session_id(session_id)
        episode_index = self._validate_episode_index(episode_index)
        if not isinstance(reset, bool):
            raise TypeError("FrameSamp reset flag must be boolean")
        if reset:
            self.reset(session_id, episode_index)
        state = self._sessions.get(session_id)
        if state is None:
            raise RuntimeError(
                "FrameSamp session has no episode buffer; the first request of "
                "every episode must set history.reset=true"
            )
        if state.episode_index != episode_index:
            raise RuntimeError(
                "FrameSamp episode changed without reset: "
                f"{state.episode_index} -> {episode_index}"
            )

        frames = np.asarray(raw_frames)
        indices = np.asarray(raw_frame_indices)
        if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
            raise ValueError(
                "framesamp_raw_rgb must be uint8 with shape [N,H,W,3]"
            )
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("framesamp_raw_frame_indices must be a 1-D integer array")
        if len(frames) == 0 or len(frames) != len(indices):
            raise ValueError("FrameSamp request must upload one index per non-empty frame")
        expected = np.arange(
            len(state.features), len(state.features) + len(frames), dtype=np.int64
        )
        if not np.array_equal(indices.astype(np.int64, copy=False), expected):
            raise RuntimeError(
                "FrameSamp raw frames are missing, duplicated, or out of order: "
                f"expected {expected.tolist()}, got {indices.tolist()}"
            )
        current = np.asarray(current_rgb)
        if current.dtype != np.uint8 or not np.array_equal(frames[-1], current):
            raise RuntimeError(
                "The uploaded raw-frame sequence must end at the current System-2 RGB"
            )

        encoded: list[torch.Tensor] = []
        for frame in frames:
            feature, _ = self.producer.encode(frame, instruction)
            if tuple(feature.shape) != (1, self.tokens_per_frame, self.feature_dim):
                raise RuntimeError("Online Qwen producer returned an incompatible shape")
            if feature.dtype != torch.bfloat16 or not torch.isfinite(feature.float()).all():
                raise RuntimeError("Online Qwen producer returned invalid features")
            encoded.append(feature[0].detach().to(device="cpu").contiguous())
        state.features.extend(encoded)
        state.requests += 1
        self.total_encoded_frames += len(encoded)
        self._sessions.move_to_end(session_id)

        positions = even_sample_positions(
            len(state.features), self.max_memory_frames
        )
        selected_features = torch.stack(
            [state.features[int(position)] for position in positions]
        )
        selected_raw = torch.from_numpy(positions.copy()).to(torch.int64)
        memory = assemble_memory_tokens(
            selected_features,
            selected_raw,
            max_memory_frames=self.max_memory_frames,
            tokens_per_frame=self.tokens_per_frame,
            feature_dim=self.feature_dim,
        )
        return {name: value.unsqueeze(0) for name, value in memory.items()}

    def health(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "sessions": len(self._sessions),
            "max_sessions": self.max_sessions,
            "total_encoded_frames": self.total_encoded_frames,
            "buffers": {
                session_id: {
                    "episode_index": state.episode_index,
                    "raw_frames": len(state.features),
                    "requests": state.requests,
                }
                for session_id, state in self._sessions.items()
            },
        }

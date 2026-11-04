# Derived from Psi0 (https://github.com/physical-superintelligence-lab/Psi0, Apache License 2.0) and modified; see NOTICE.
from pydantic import BaseModel, Field, model_validator
from typing import Any, Optional, Dict, List, TYPE_CHECKING, Literal
from psi.config.config import DataConfig
from pathlib import Path
from psi.utils import resolve_data_path
import os
import json

from psi.config.transform import ActionStateTransform
class LerobotDataConfig(DataConfig):
    root_dir: str
    train_repo_ids: List[str] = Field(default_factory=list)
    val_repo_ids: List[str] = Field(default_factory=list)
    # The released Psi0 real-task run configs used Decord.  Keep the backend
    # in the typed config so it is visible in run_config.json rather than
    # relying only on an unrecorded environment override.
    video_backend: str = "decord"
    symbolic_memory_mode: Literal[
        "disabled", "correct", "empty", "shuffled", "wrong"
    ] = "disabled"
    symbolic_memory_rows_path: str | None = None
    symbolic_use_phase_action_mask: bool = True
    symbolic_memory_seed: int = Field(default=292285, ge=0, le=2**32 - 1)

    @model_validator(mode="after")
    def check_repo_ids(self):
        if len(self.train_repo_ids) == 0:
            raise ValueError("train_repo_ids must be provided")
        if len(self.val_repo_ids) == 0:
            self.val_repo_ids = [self.train_repo_ids[0]]
        if (
            self.symbolic_memory_mode != "disabled"
            and not self.symbolic_memory_rows_path
        ):
            raise ValueError(
                "symbolic_memory_rows_path is required when symbolic memory "
                "is enabled"
            )
        if (
            self.symbolic_memory_mode == "disabled"
            and self.symbolic_memory_rows_path is not None
        ):
            raise ValueError(
                "symbolic_memory_rows_path must be omitted when "
                "symbolic_memory_mode=disabled"
            )
        return self
    
    @model_validator(mode="after")
    def load_stats(self):
        if not isinstance(self.transform.field, ActionStateTransform):
            return self
        if (
            not Path(self.transform.field.stat_path).is_absolute() and 
            self.transform.field.action_max is None
        ):
            fpath = resolve_data_path(
                Path(self.root_dir) / self.train_repo_ids[0] / self.transform.field.stat_path
            )
            if not os.path.exists(fpath):
                return self
            with open(fpath, "r") as f:
                stats = json.load(f)
                self.transform.field.populate_stats(stats)
        return self

    def __call__(self, split: str = "train", transform_kwargs={}, **kwargs) -> Any:
        from psi.data.lerobot import LeRobotDatasetWrapper
        from psi.data.dataset import Dataset as MapStyleDataset

        train_dataset = LeRobotDatasetWrapper(self, split=split)
        return MapStyleDataset(self, train_dataset, transform_kwargs=transform_kwargs)

    def mock(self, split: str = "train", transform_kwargs={}, **kwargs) -> Any:
        dataset = self.__call__(split, transform_kwargs=transform_kwargs, **kwargs)
        return dataset[0]

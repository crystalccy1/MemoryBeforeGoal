# Derived from Psi0 (https://github.com/physical-superintelligence-lab/Psi0, Apache License 2.0) and modified; see NOTICE.
from __future__ import annotations

try:
    from lerobot.common.datasets.lerobot_dataset import (  # type: ignore
        LeRobotDataset,
        LeRobotDatasetMetadata,
        MultiLeRobotDataset,
    )

    LEROBOT_LAYOUT = "common"
except ModuleNotFoundError:
    from lerobot.datasets.lerobot_dataset import (  # type: ignore
        LeRobotDataset,
        LeRobotDatasetMetadata,
        MultiLeRobotDataset,
    )

    LEROBOT_LAYOUT = "datasets"

__all__ = [
    "LEROBOT_LAYOUT",
    "LeRobotDataset",
    "LeRobotDatasetMetadata",
    "MultiLeRobotDataset",
]

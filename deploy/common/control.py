"""Robot state and control target types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray


class StateName(Enum):
    DAMPING = auto()
    FIXEDPOSE = auto()
    LOCO = auto()
    TRACKING = auto()


@dataclass(frozen=True)
class RobotState:
    joint_pos: NDArray[np.float32]
    joint_vel: NDArray[np.float32]
    base_quat: NDArray[np.float32]
    base_ang_vel: NDArray[np.float32]
    projected_gravity: NDArray[np.float32]
    torso_quat: NDArray[np.float32]
    base_height: float


@dataclass(frozen=True)
class ControlTarget:
    joint_pos: NDArray[np.float32]
    stiffness: NDArray[np.float32]
    damping: NDArray[np.float32]
    effort_limit: NDArray[np.float32]
    stabilize_base: bool = False

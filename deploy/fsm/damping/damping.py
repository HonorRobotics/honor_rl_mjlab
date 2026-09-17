"""Damping state implementation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from common.config import joint_array, load_state_config
from common.control import ControlTarget, RobotState, StateName
from fsm.base import State
from numpy.typing import NDArray


class DampingState(State):
    name = StateName.DAMPING

    def __init__(self) -> None:
        config = load_state_config(Path(__file__).with_name("config.yaml"))
        self.damping = joint_array(config, "kds")
        self.effort_limit = joint_array(config, "effort_limit")

    def step(
        self,
        robot: RobotState,
        velocity_command: NDArray[np.float32],
    ) -> ControlTarget:
        del velocity_command
        return ControlTarget(
            joint_pos=robot.joint_pos.copy(),
            stiffness=np.zeros(29, dtype=np.float32),
            damping=self.damping.copy(),
            effort_limit=self.effort_limit.copy(),
        )

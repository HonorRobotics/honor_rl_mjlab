"""Joint interpolation to the VITA BOY default pose."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from common.config import joint_array, load_state_config
from common.control import ControlTarget, RobotState, StateName
from fsm.base import State
from numpy.typing import NDArray


class FixedPoseState(State):
    name = StateName.FIXEDPOSE

    def __init__(self, control_dt: float) -> None:
        config = load_state_config(Path(__file__).with_name("config.yaml"))
        self.default_angles = joint_array(config, "default_angles")
        self.stiffness = joint_array(config, "kps")
        self.damping = joint_array(config, "kds")
        self.effort_limit = joint_array(config, "effort_limit")
        self.total_steps = max(1, round(float(config["duration"]) / control_dt))
        self.current_step = 0
        self.start = self.default_angles.copy()

    def enter(self, robot: RobotState) -> None:
        self.current_step = 0
        self.start = robot.joint_pos.copy()

    def step(
        self,
        robot: RobotState,
        velocity_command: NDArray[np.float32],
    ) -> ControlTarget:
        del robot, velocity_command
        self.current_step += 1
        alpha = min(self.current_step / self.total_steps, 1.0)
        target = (1.0 - alpha) * self.start + alpha * self.default_angles
        return ControlTarget(
            joint_pos=target.astype(np.float32),
            stiffness=self.stiffness.copy(),
            damping=self.damping.copy(),
            effort_limit=self.effort_limit.copy(),
            stabilize_base=True,
        )

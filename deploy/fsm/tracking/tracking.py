"""Single-motion tracking-policy state."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from common.config import deploy_path, joint_array, load_state_config
from common.control import ControlTarget, RobotState, StateName
from common.math import matrix_from_quat, quat_conjugate, quat_multiply, yaw_from_quat, yaw_quat
from common.policy import OnnxPolicy
from fsm.base import State
from numpy.typing import NDArray

from .motion import Motion


class TrackingState(State):
    name = StateName.TRACKING

    def __init__(self, control_dt: float) -> None:
        config = load_state_config(Path(__file__).with_name("config.yaml"))
        self.default_angles = joint_array(config, "default_angles")
        self.stiffness = joint_array(config, "kps")
        self.damping = joint_array(config, "kds")
        self.effort_limit = joint_array(config, "effort_limit")
        self.action_scale = float(config["action_scale"]) * self.effort_limit / self.stiffness
        self.motion_torso_index = int(config["motion_torso_index"])

        policy_path = deploy_path(config, "policy_path")
        motion_path = deploy_path(config, "motion_path")
        self.policy = OnnxPolicy(policy_path, observation_size=154)
        self.motion = Motion.load(motion_path)
        expected_fps = 1.0 / control_dt
        if not np.isclose(self.motion.fps, expected_fps):
            raise ValueError(f"Motion fps is {self.motion.fps:g}, expected {expected_fps:g}")
        if not 0 <= self.motion_torso_index < self.motion.body_quat_w.shape[1]:
            raise ValueError("Motion does not contain the torso body")

        self.frame = 0
        self.yaw_offset = 0.0
        self.action = np.zeros(29, dtype=np.float32)

    def enter(self, robot: RobotState) -> None:
        self.frame = 0
        self.action.fill(0.0)
        self.policy.reset()
        reference = self.motion.body_quat_w[0, self.motion_torso_index]
        self.yaw_offset = yaw_from_quat(robot.torso_quat) - yaw_from_quat(reference)

    def step(
        self,
        robot: RobotState,
        velocity_command: NDArray[np.float32],
    ) -> ControlTarget:
        del velocity_command
        reference_quat = self.motion.body_quat_w[self.frame, self.motion_torso_index]
        aligned_quat = quat_multiply(yaw_quat(self.yaw_offset), reference_quat)
        relative_quat = quat_multiply(quat_conjugate(robot.torso_quat), aligned_quat)
        orientation = matrix_from_quat(relative_quat)[:, :2].reshape(-1)

        observation = np.concatenate(
            (
                self.motion.joint_pos[self.frame],
                self.motion.joint_vel[self.frame],
                orientation,
                robot.base_ang_vel,
                robot.joint_pos - self.default_angles,
                robot.joint_vel,
                self.action,
            )
        ).astype(np.float32)
        self.action = self.policy(observation)
        self.frame = min(self.frame + 1, self.motion.frame_count - 1)
        return ControlTarget(
            joint_pos=self.default_angles + self.action_scale * self.action,
            stiffness=self.stiffness.copy(),
            damping=self.damping.copy(),
            effort_limit=self.effort_limit.copy(),
        )

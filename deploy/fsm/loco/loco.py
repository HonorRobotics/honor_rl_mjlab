"""Velocity-policy state."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
from common.config import deploy_path, joint_array, load_state_config
from common.control import ControlTarget, RobotState, StateName
from common.policy import OnnxPolicy, read_metadata
from common.robot_interface import JOINT_NAMES
from fsm.base import State
from numpy.typing import NDArray


class TermHistory:
    """Observation history buffer."""

    def __init__(self, length: int) -> None:
        self.length = length
        self.values: deque[NDArray[np.float32]] = deque(maxlen=length)

    def push(self, value: NDArray[np.float32]) -> NDArray[np.float32]:
        if not self.values:
            self.values.extend(value.copy() for _ in range(self.length))
        else:
            self.values.append(value.copy())
        return np.concatenate(self.values)

    def reset(self) -> None:
        self.values.clear()


class LocomotionState(State):
    name = StateName.LOCO

    def __init__(self) -> None:
        config = load_state_config(Path(__file__).with_name("config.yaml"))
        self.policy_path = deploy_path(config, "policy_path")
        self.default_angles = joint_array(config, "default_angles")
        self.stiffness = joint_array(config, "kps")
        self.damping = joint_array(config, "kds")
        self.effort_limit = joint_array(config, "effort_limit")
        self.action_scale = float(config["action_scale"])
        self.command_scale = np.array(
            [
                config["linear_velocity"],
                config["linear_velocity"],
                config["angular_velocity"],
            ],
            dtype=np.float32,
        )

        self.policy = OnnxPolicy(self.policy_path)
        if self.policy.observation_size % 96 != 0:
            raise ValueError(f"Velocity policy observation size must be a multiple of 96: {self.policy_path}")
        self.history_length = self.policy.observation_size // 96
        if self.history_length not in (1, 10):
            raise ValueError(f"Unsupported velocity history: {self.history_length}")

        self.histories = [TermHistory(self.history_length) for _ in range(6)]
        self.action = np.zeros(29, dtype=np.float32)
        self._validate_metadata()

    def _validate_metadata(self) -> None:
        metadata = read_metadata(self.policy_path)
        names = metadata.get("joint_names")
        if names and tuple(names.split(",")) != JOINT_NAMES:
            raise ValueError("Velocity policy joint order does not match VITA BOY")

        observation_names = metadata.get("observation_names")
        expected_observations = (
            "base_ang_vel,projected_gravity,velocity_commands,joint_pos_rel,joint_vel_rel,last_action"
        )
        if observation_names and observation_names != expected_observations:
            raise ValueError("Velocity policy observation order is unsupported")

        default_position = metadata.get("default_joint_pos")
        if default_position:
            exported = np.fromstring(default_position, sep=",")
            if exported.shape != self.default_angles.shape or not np.allclose(exported, self.default_angles, atol=1e-5):
                raise ValueError("Velocity policy default pose does not match VITA BOY")

    def enter(self, robot: RobotState) -> None:
        del robot
        self.action.fill(0.0)
        self.policy.reset()
        for history in self.histories:
            history.reset()

    def step(
        self,
        robot: RobotState,
        velocity_command: NDArray[np.float32],
    ) -> ControlTarget:
        terms = (
            robot.base_ang_vel,
            robot.projected_gravity,
            velocity_command * self.command_scale,
            robot.joint_pos - self.default_angles,
            robot.joint_vel,
            self.action,
        )
        observation = np.concatenate([history.push(term) for history, term in zip(self.histories, terms, strict=True)])
        self.action = self.policy(np.clip(observation, -100.0, 100.0))
        return ControlTarget(
            joint_pos=self.default_angles + self.action_scale * self.action,
            stiffness=self.stiffness.copy(),
            damping=self.damping.copy(),
            effort_limit=self.effort_limit.copy(),
        )

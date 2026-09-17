"""Robot interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .control import ControlTarget, RobotState

JOINT_NAMES = (
    # Left leg
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # Right leg
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # Waist
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    # Left arm
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    # Right arm
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


class RobotInterface(ABC):
    @property
    @abstractmethod
    def control_dt(self) -> float:
        """Control-loop period in seconds."""

    @abstractmethod
    def get_state(self) -> RobotState:
        """Return the latest robot state."""

    @abstractmethod
    def send_command(self, target: ControlTarget) -> None:
        """Apply one control-period command."""

    @abstractmethod
    def reset(self) -> None:
        """Reset the backend to its initial state."""

    @abstractmethod
    def is_running(self) -> bool:
        """Return whether the backend should keep running."""

    @abstractmethod
    def close(self) -> None:
        """Release backend resources."""

"""Motion data owned by the tracking state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Motion:
    joint_pos: NDArray[np.float32]
    joint_vel: NDArray[np.float32]
    body_quat_w: NDArray[np.float32]
    fps: float

    @property
    def frame_count(self) -> int:
        return self.joint_pos.shape[0]

    @classmethod
    def load(cls, path: Path) -> Motion:
        if not path.is_file():
            raise FileNotFoundError(f"Motion not found: {path}")
        with np.load(path, allow_pickle=False) as data:
            required = {"joint_pos", "joint_vel", "body_quat_w"}
            missing = required.difference(data.files)
            if missing:
                raise ValueError(f"Motion is missing fields {sorted(missing)}: {path}")
            joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
            joint_vel = np.asarray(data["joint_vel"], dtype=np.float32)
            body_quat_w = np.asarray(data["body_quat_w"], dtype=np.float32)
            fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data else 50.0

        if joint_pos.ndim != 2 or joint_pos.shape[0] == 0 or joint_pos.shape[1] != 29:
            raise ValueError(f"Expected joint_pos shape (frames, 29), got {joint_pos.shape}")
        if (
            joint_vel.shape != joint_pos.shape
            or body_quat_w.ndim != 3
            or body_quat_w.shape[0] != joint_pos.shape[0]
            or body_quat_w.shape[2] != 4
        ):
            raise ValueError("Motion arrays have inconsistent shapes")
        if fps <= 0.0:
            raise ValueError(f"Motion fps must be positive, got {fps:g}")
        if not all(np.all(np.isfinite(item)) for item in (joint_pos, joint_vel, body_quat_w)):
            raise ValueError(f"Motion contains NaN or Inf: {path}")
        return cls(joint_pos, joint_vel, body_quat_w, fps)

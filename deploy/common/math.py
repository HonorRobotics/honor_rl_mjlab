"""Quaternion helpers using MuJoCo's wxyz convention."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def quat_conjugate(quat: NDArray[np.floating]) -> NDArray[np.floating]:
    result = quat.copy()
    result[1:] *= -1.0
    return result


def quat_multiply(lhs: NDArray[np.floating], rhs: NDArray[np.floating]) -> NDArray[np.floating]:
    w1, x1, y1, z1 = lhs
    w2, x2, y2, z2 = rhs
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=lhs.dtype,
    )


def yaw_from_quat(quat: NDArray[np.floating]) -> float:
    w, x, y, z = quat
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def yaw_quat(yaw: float) -> NDArray[np.float32]:
    half = 0.5 * yaw
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float32)


def matrix_from_quat(quat: NDArray[np.floating]) -> NDArray[np.floating]:
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=quat.dtype,
    )


def projected_gravity(base_quat: NDArray[np.floating]) -> NDArray[np.float32]:
    rotation = matrix_from_quat(base_quat)
    return (rotation.T @ np.array([0.0, 0.0, -1.0])).astype(np.float32)

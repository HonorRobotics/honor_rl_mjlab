"""Deployment configuration loading."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

_DEPLOY_ROOT = Path(__file__).resolve().parents[1]
_JOINT_COUNT = 29


@dataclass(frozen=True)
class SimulationConfig:
    physics_dt: float
    control_dt: float


def load_state_config(path: Path) -> dict[str, Any]:
    """Load a state configuration from YAML."""
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"State configuration must be a mapping: {path}")
    return config


def joint_array(config: dict[str, Any], key: str) -> NDArray[np.float32]:
    """Read and validate a joint array from a state configuration."""
    values = np.asarray(config[key], dtype=np.float32)
    if values.shape != (_JOINT_COUNT,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{key} must contain {_JOINT_COUNT} finite values")
    return values


def deploy_path(config: dict[str, Any], key: str) -> Path:
    """Resolve a path relative to the deployment directory."""
    value = config[key]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty path")
    return (_DEPLOY_ROOT / value).resolve()


def load_config(path: Path) -> SimulationConfig:
    """Load the simulation configuration."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("rb") as stream:
        raw = tomllib.load(stream)

    simulation = raw["simulation"]
    physics_dt = float(simulation["physics_dt"])
    control_dt = float(simulation["control_dt"])
    ratio = control_dt / physics_dt
    if physics_dt <= 0.0 or control_dt <= 0.0 or abs(ratio - round(ratio)) > 1e-9:
        raise ValueError("control_dt must be a positive integer multiple of physics_dt")

    return SimulationConfig(
        physics_dt=physics_dt,
        control_dt=control_dt,
    )

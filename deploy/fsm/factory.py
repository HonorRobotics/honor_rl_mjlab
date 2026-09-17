"""FSM construction."""

from __future__ import annotations

from collections.abc import Callable

from common.config import SimulationConfig
from common.control import RobotState, StateName
from fsm.base import State, StateMachine
from fsm.damping import DampingState
from fsm.fixedpose import FixedPoseState
from fsm.loco import LocomotionState
from fsm.tracking import TrackingState


def _optional_state(name: StateName, factory: Callable[[], State]) -> State | None:
    try:
        return factory()
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"[WARN] {name.name} unavailable: {error}")
        return None


def build_state_machine(config: SimulationConfig, robot: RobotState) -> StateMachine:
    """Build the deployment state machine."""
    states: list[State | None] = [
        DampingState(),
        FixedPoseState(config.control_dt),
        _optional_state(
            StateName.LOCO,
            LocomotionState,
        ),
        _optional_state(
            StateName.TRACKING,
            lambda: TrackingState(config.control_dt),
        ),
    ]
    return StateMachine([state for state in states if state is not None], robot)

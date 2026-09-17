"""State transition management."""

from __future__ import annotations

import numpy as np
from common.control import ControlTarget, RobotState, StateName
from numpy.typing import NDArray

from .state import State


class StateMachine:
    """Deployment state machine."""

    def __init__(self, states: list[State], robot: RobotState) -> None:
        self.states = {state.name: state for state in states}
        self.current = self.states[StateName.DAMPING]
        self.current.enter(robot)

    def reset(self, robot: RobotState) -> None:
        self.current.exit()
        self.current = self.states[StateName.DAMPING]
        self.current.enter(robot)
        print("FSM -> DAMPING")

    def transition(self, target: StateName, robot: RobotState) -> bool:
        if target == self.current.name:
            return True
        if target not in self.states:
            print(f"[WARN] {target.name} is unavailable")
            return False

        policy_states = {StateName.LOCO, StateName.TRACKING}
        if target in policy_states and self.current.name not in {
            StateName.FIXEDPOSE,
            *policy_states,
        }:
            print(f"[WARN] Enter FIXEDPOSE before {target.name}")
            return False

        self.current.exit()
        self.current = self.states[target]
        self.current.enter(robot)
        print(f"FSM -> {target.name}")
        return True

    def step(
        self,
        robot: RobotState,
        velocity_command: NDArray[np.float32],
    ) -> ControlTarget:
        return self.current.step(robot, velocity_command)

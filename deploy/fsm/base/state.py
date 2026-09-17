"""Base class for deployment states."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from common.control import ControlTarget, RobotState, StateName
from numpy.typing import NDArray


class State(ABC):
    name: StateName

    def enter(self, robot: RobotState) -> None:
        del robot

    @abstractmethod
    def step(
        self,
        robot: RobotState,
        velocity_command: NDArray[np.float32],
    ) -> ControlTarget:
        """Compute one policy-period target."""

    def exit(self) -> None:
        pass

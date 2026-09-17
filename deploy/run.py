#!/usr/bin/env python3
"""Run VITA BOY MuJoCo deployment."""

from __future__ import annotations

import argparse
import contextlib
import time
from pathlib import Path

import numpy as np
from common.config import SimulationConfig, load_config
from common.control import StateName
from fsm import build_state_machine
from inference.mujoco_robot import MuJoCoRobot
from input.keyboard import Keyboard, KeyCommand

_STATE_COMMANDS = {
    KeyCommand.DAMPING: StateName.DAMPING,
    KeyCommand.FIXEDPOSE: StateName.FIXEDPOSE,
    KeyCommand.LOCO: StateName.LOCO,
    KeyCommand.TRACKING: StateName.TRACKING,
}


class Runtime:
    """Run the deployment control loop."""

    def __init__(self, config: SimulationConfig, launch_viewer: bool = True) -> None:
        self.config = config
        self.robot = MuJoCoRobot(config, launch_viewer=launch_viewer)
        self.keyboard = Keyboard(enabled=launch_viewer)
        self.fsm = build_state_machine(config, self.robot.get_state())
        self.velocity_command = np.zeros(3, dtype=np.float32)
        self.running = True

    def _handle_inputs(self) -> None:
        for command in self.keyboard.poll():
            if command in _STATE_COMMANDS:
                self.fsm.transition(_STATE_COMMANDS[command], self.robot.get_state())
            elif command == KeyCommand.EXIT:
                self.running = False
        forward, lateral, yaw = self.keyboard.movement()
        self.velocity_command[:] = (forward, lateral, yaw)

    def step(self) -> None:
        """Run one control step."""
        self._handle_inputs()
        if not self.running or not self.robot.is_running():
            return

        robot_state = self.robot.get_state()
        values = (robot_state.joint_pos, robot_state.joint_vel, robot_state.base_quat)
        unsafe = not all(np.all(np.isfinite(value)) for value in values)
        fallen = robot_state.base_height < 0.25 and self.fsm.current.name in {
            StateName.LOCO,
            StateName.TRACKING,
        }
        if (unsafe and self.fsm.current.name != StateName.DAMPING) or fallen:
            reason = "non-finite state" if unsafe else f"base height {robot_state.base_height:.3f} m"
            print(f"[SAFETY] {reason}; switching to DAMPING")
            self.fsm.transition(StateName.DAMPING, robot_state)

        target = self.fsm.step(robot_state, self.velocity_command)
        self.robot.send_command(target)

    def run(self) -> None:
        print("Controls: P DAMPING | R FIXEDPOSE | L LOCO | M TRACKING | hold arrows and Q/E to move | Esc exit")
        try:
            with contextlib.suppress(KeyboardInterrupt):
                while self.running and self.robot.is_running():
                    started = time.perf_counter()
                    self.step()
                    remaining = self.robot.control_dt - (time.perf_counter() - started)
                    if remaining > 0.0:
                        time.sleep(remaining)
        finally:
            self.keyboard.close()
            self.robot.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.toml"),
        help="Deployment configuration file (default: deploy/config.toml).",
    )
    args = parser.parse_args()
    Runtime(load_config(args.config)).run()


if __name__ == "__main__":
    main()

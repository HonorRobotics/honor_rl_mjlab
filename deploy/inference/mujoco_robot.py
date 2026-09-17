"""VITA BOY MuJoCo implementation of the deployment robot interface."""

from __future__ import annotations

import contextlib
from pathlib import Path

import mujoco
import mujoco.viewer
from common.config import SimulationConfig
from common.control import ControlTarget, RobotState
from common.math import projected_gravity, yaw_from_quat, yaw_quat
from common.robot_interface import JOINT_NAMES, RobotInterface

_JOINT_ARMATURE = (
    # Left leg
    *(0.08973, 0.0334, 0.01334, 0.0334, 0.00757, 0.00757),
    # Right leg
    *(0.08973, 0.0334, 0.01334, 0.0334, 0.00757, 0.00757),
    # Waist
    *(0.08973, 0.00757, 0.00757),
    # Left arm
    *(0.003785, 0.003785, 0.003785, 0.003785, 0.003785, 0.01, 0.01),
    # Right arm
    *(0.003785, 0.003785, 0.003785, 0.003785, 0.003785, 0.01, 0.01),
)


class MuJoCoRobot(RobotInterface):
    """VITA BOY MuJoCo backend."""

    def __init__(
        self,
        config: SimulationConfig,
        launch_viewer: bool = True,
    ) -> None:
        self._control_dt = config.control_dt
        self._decimation = round(config.control_dt / config.physics_dt)
        self.model = self._build_model(config.physics_dt)
        self.data = mujoco.MjData(self.model)

        self.qpos_indices: list[int] = []
        self.dof_indices: list[int] = []
        for name in JOINT_NAMES:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"Joint not found in MJCF: {name}")
            self.qpos_indices.append(int(self.model.jnt_qposadr[joint_id]))
            self.dof_indices.append(int(self.model.jnt_dofadr[joint_id]))

        self.pelvis_body_id = self._body_id("pelvis_link")
        self.torso_body_id = self._body_id("torso_link")
        sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
        sensor_start = int(self.model.sensor_adr[sensor_id])
        self.ang_vel_slice = slice(
            sensor_start,
            sensor_start + int(self.model.sensor_dim[sensor_id]),
        )

        self.viewer = None
        self.reset()
        if launch_viewer:
            self.viewer = mujoco.viewer.launch_passive(
                self.model,
                self.data,
            )
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.viewer.cam.trackbodyid = self.pelvis_body_id
            self.viewer.cam.distance = 3.5
            self.viewer.cam.azimuth = -140.0
            self.viewer.cam.elevation = -20.0

    @staticmethod
    def _build_model(physics_dt: float) -> mujoco.MjModel:
        xml_path = Path(__file__).with_name("scene.xml")
        spec = mujoco.MjSpec.from_file(str(xml_path))
        for actuator in list(spec.actuators):
            spec.delete(actuator)
        for geom in spec.geoms:
            if not geom.name.endswith("_collision"):
                continue
            is_foot = geom.name.startswith(("left_foot", "right_foot"))
            geom.condim = 3 if is_foot else 1
            geom.priority = 1 if is_foot else 0
            if is_foot:
                geom.friction[0] = 0.6

        for name, armature in zip(JOINT_NAMES, _JOINT_ARMATURE, strict=True):
            spec.joint(name).armature = armature
            actuator = spec.add_actuator(name=f"{name}_position", target=name)
            actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
            actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
            actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            actuator.gainprm[0] = 0.0
            actuator.biasprm[1] = 0.0
            actuator.biasprm[2] = 0.0
            actuator.inheritrange = 0.0
            actuator.ctrllimited = False
            actuator.forcelimited = True
            actuator.forcerange[:] = (-1.0, 1.0)

        model = spec.compile()
        model.opt.timestep = physics_dt
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        model.opt.iterations = 100
        model.opt.tolerance = 1e-8
        model.opt.ls_iterations = 50
        model.opt.ls_tolerance = 0.01
        model.opt.ccd_iterations = 128
        return model

    def _body_id(self, name: str) -> int:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"Body not found in MJCF: {name}")
        return body_id

    @property
    def control_dt(self) -> float:
        return self._control_dt

    def get_state(self) -> RobotState:
        base_quat = self.data.xquat[self.pelvis_body_id].astype("float32").copy()
        return RobotState(
            joint_pos=self.data.qpos[self.qpos_indices].astype("float32").copy(),
            joint_vel=self.data.qvel[self.dof_indices].astype("float32").copy(),
            base_quat=base_quat,
            base_ang_vel=self.data.sensordata[self.ang_vel_slice].astype("float32").copy(),
            projected_gravity=projected_gravity(base_quat),
            torso_quat=self.data.xquat[self.torso_body_id].astype("float32").copy(),
            base_height=float(self.data.xpos[self.pelvis_body_id, 2]),
        )

    def send_command(self, target: ControlTarget) -> None:
        self.model.actuator_gainprm[:, 0] = target.stiffness
        self.model.actuator_biasprm[:, 1] = -target.stiffness
        self.model.actuator_biasprm[:, 2] = -target.damping
        self.model.actuator_forcerange[:, 0] = -target.effort_limit
        self.model.actuator_forcerange[:, 1] = target.effort_limit
        self.data.ctrl[:] = target.joint_pos

        for _ in range(self._decimation):
            if target.stabilize_base:
                yaw = yaw_from_quat(self.data.qpos[3:7])
                self.data.qpos[2] = 0.82
                self.data.qpos[3:7] = yaw_quat(yaw)
                self.data.qvel[:6] = 0.0
            mujoco.mj_step(self.model, self.data)
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = (0.0, 0.0, 0.82)
        self.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qpos[self.qpos_indices] = 0.0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def is_running(self) -> bool:
        return self.viewer is None or self.viewer.is_running()

    def close(self) -> None:
        viewer = self.viewer
        self.viewer = None
        with contextlib.suppress(Exception):
            if viewer is not None and viewer.is_running():
                viewer.close()

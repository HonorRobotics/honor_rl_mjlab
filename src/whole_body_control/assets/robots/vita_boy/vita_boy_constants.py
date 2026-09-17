"""VITA BOY robot constants in mjlab BuiltinPositionActuatorCfg style."""

from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

VITA_BOY_XML = Path(__file__).resolve().parent / "xml" / "main.xml"

assert VITA_BOY_XML.exists()


def get_spec() -> mujoco.MjSpec:
    """Load the standard model with terrain and position control supplied by mjlab."""
    spec = mujoco.MjSpec.from_file(str(VITA_BOY_XML))
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    # Keep the observation names used by the training tasks.
    spec.sensor("imu_gyro").name = "imu_ang_vel"
    spec.sensor("imu_acc").name = "imu_lin_acc"
    spec.add_sensor(
        name="imu_lin_vel",
        type=mujoco.mjtSensor.mjSENS_VELOCIMETER,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname="imu_in_pelvis",
    )
    return spec


ARMATURE_PG20 = 3.785e-3
STIFFNESS_PG20 = 15.0
DAMPING_PG20 = 1.0
EFFORT_LIMIT_PG20 = 30.0

ARMATURE_PG20_PARALLEL = ARMATURE_PG20 * 2
STIFFNESS_PG20_PARALLEL = STIFFNESS_PG20 * 2
DAMPING_PG20_PARALLEL = DAMPING_PG20 * 2
EFFORT_LIMIT_PG20_PARALLEL = EFFORT_LIMIT_PG20 * 2

ARMATURE_PG30 = 1.334e-2
STIFFNESS_PG30 = 100.0
DAMPING_PG30 = 3.0
EFFORT_LIMIT_PG30 = 88.0

ARMATURE_PG40 = 3.340e-2
STIFFNESS_PG40 = 150.0
DAMPING_PG40 = 4.0
EFFORT_LIMIT_PG40 = 139.0

ARMATURE_PG50 = 8.973e-2
STIFFNESS_PG50 = 200.0
DAMPING_PG50 = 5.0
EFFORT_LIMIT_PG50 = 150.0

ARMATURE_DM3410 = 0.01
STIFFNESS_DM3410 = 10.0
DAMPING_DM3410 = 1.0
EFFORT_LIMIT_DM3410 = 7.0

VITA_BOY_ACTUATOR_PG20 = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*_shoulder_pitch_joint",
        ".*_shoulder_roll_joint",
        ".*_shoulder_yaw_joint",
        ".*_elbow_joint",
        ".*_wrist_roll_joint",
    ),
    stiffness=STIFFNESS_PG20,
    damping=DAMPING_PG20,
    effort_limit=EFFORT_LIMIT_PG20,
    armature=ARMATURE_PG20,
)

VITA_BOY_ACTUATOR_PG20_PARALLEL = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "waist_roll_joint",
        "waist_pitch_joint",
        ".*_ankle_pitch_joint",
        ".*_ankle_roll_joint",
    ),
    stiffness=STIFFNESS_PG20_PARALLEL,
    damping=DAMPING_PG20_PARALLEL,
    effort_limit=EFFORT_LIMIT_PG20_PARALLEL,
    armature=ARMATURE_PG20_PARALLEL,
)

VITA_BOY_ACTUATOR_PG30 = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_hip_yaw_joint",),
    stiffness=STIFFNESS_PG30,
    damping=DAMPING_PG30,
    effort_limit=EFFORT_LIMIT_PG30,
    armature=ARMATURE_PG30,
)

VITA_BOY_ACTUATOR_PG40 = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_hip_roll_joint", ".*_knee_joint"),
    stiffness=STIFFNESS_PG40,
    damping=DAMPING_PG40,
    effort_limit=EFFORT_LIMIT_PG40,
    armature=ARMATURE_PG40,
)

VITA_BOY_ACTUATOR_PG50 = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_hip_pitch_joint", "waist_yaw_joint"),
    stiffness=STIFFNESS_PG50,
    damping=DAMPING_PG50,
    effort_limit=EFFORT_LIMIT_PG50,
    armature=ARMATURE_PG50,
)

VITA_BOY_ACTUATOR_DM3410 = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
    stiffness=STIFFNESS_DM3410,
    damping=DAMPING_DM3410,
    effort_limit=EFFORT_LIMIT_DM3410,
    armature=ARMATURE_DM3410,
)

VITA_BOY_ACTUATOR_JOINT_MAP: dict[str, list[str]] = {
    "PG20": list(VITA_BOY_ACTUATOR_PG20.target_names_expr),
    "PG20_Parallel": list(VITA_BOY_ACTUATOR_PG20_PARALLEL.target_names_expr),
    "PG30": list(VITA_BOY_ACTUATOR_PG30.target_names_expr),
    "PG40": list(VITA_BOY_ACTUATOR_PG40.target_names_expr),
    "PG50": list(VITA_BOY_ACTUATOR_PG50.target_names_expr),
    "DM3410": list(VITA_BOY_ACTUATOR_DM3410.target_names_expr),
}


VITA_BOY_BASE_INIT_POS: tuple[float, float, float] = (0.0, 0.0, 0.82)

VITA_BOY_INIT_JOINT_POS: dict[str, float] = {
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
}

KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=VITA_BOY_BASE_INIT_POS,
    joint_pos=VITA_BOY_INIT_JOINT_POS,
    joint_vel={".*": 0.0},
)


VITA_BOY_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        VITA_BOY_ACTUATOR_PG20,
        VITA_BOY_ACTUATOR_PG20_PARALLEL,
        VITA_BOY_ACTUATOR_PG30,
        VITA_BOY_ACTUATOR_PG40,
        VITA_BOY_ACTUATOR_PG50,
        VITA_BOY_ACTUATOR_DM3410,
    ),
    soft_joint_pos_limit_factor=0.95,
)

FULL_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    condim={r"^(left|right)_foot[1-9]_collision$": 3, ".*_collision": 1},
    priority={r"^(left|right)_foot[1-9]_collision$": 1},
    friction={r"^(left|right)_foot[1-9]_collision$": (0.6,)},
)


def get_vita_boy_robot_cfg() -> EntityCfg:
    """Get a fresh VITA BOY robot configuration instance."""
    return EntityCfg(
        init_state=KNEES_BENT_KEYFRAME,
        spec_fn=get_spec,
        articulation=VITA_BOY_ARTICULATION,
        collisions=(FULL_COLLISION,),
    )


VITA_BOY_ACTION_SCALE: dict[str, float] = {}
for a in VITA_BOY_ARTICULATION.actuators:
    assert isinstance(a, BuiltinPositionActuatorCfg)
    e = a.effort_limit
    s = a.stiffness
    names = a.target_names_expr
    assert e is not None
    for n in names:
        VITA_BOY_ACTION_SCALE[n] = 0.25 * e / s

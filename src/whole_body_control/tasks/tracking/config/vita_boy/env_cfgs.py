"""VITA BOY flat tracking environment configuration (no state estimation)."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg

from whole_body_control.assets.robots import VITA_BOY_ACTION_SCALE, get_vita_boy_robot_cfg
from whole_body_control.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg


def vita_boy_flat_tracking_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create VITA BOY flat terrain tracking configuration (no state estimation)."""
    cfg = make_tracking_env_cfg()

    cfg.scene.entities = {"robot": get_vita_boy_robot_cfg()}

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="pelvis_link", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="pelvis_link", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    cfg.scene.sensors = (self_collision_cfg,)

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = VITA_BOY_ACTION_SCALE

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.anchor_body_name = "torso_link"
    motion_cmd.body_names = (
        "pelvis_link",
        "left_hip_roll_link",
        "left_knee_link",
        "left_ankle_roll_link",
        "right_hip_roll_link",
        "right_knee_link",
        "right_ankle_roll_link",
        "torso_link",
        "left_shoulder_roll_link",
        "left_elbow_link",
        "left_wrist_yaw_link",
        "right_shoulder_roll_link",
        "right_elbow_link",
        "right_wrist_yaw_link",
    )

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = r"^(left|right)_foot[1-9]_collision$"
    cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

    cfg.terminations["ee_body_pos"].params["body_names"] = (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    )

    cfg.viewer.body_name = "torso_link"

    # Apply play mode overrides.
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        motion_cmd.pose_range = {}
        motion_cmd.velocity_range = {}
        motion_cmd.sampling_mode = "start"

    return cfg

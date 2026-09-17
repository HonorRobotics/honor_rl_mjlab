"""Manager-based locomotion configuration for the VITA BOY humanoid."""

import math
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

import whole_body_control.tasks.velocity.mdp as mdp
from whole_body_control.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg
from whole_body_control.tasks.velocity.terrain_cfg import ROUGH_TERRAINS_CFG

OBSERVATION_CLIP = (-100.0, 100.0)


def make_velocity_env_cfg() -> ManagerBasedRlEnvCfg:
    """Create the VITA BOY locomotion configuration shared by both terrains."""

    actor_terms = {
        "base_ang_vel": ObservationTermCfg(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=OBSERVATION_CLIP,
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=OBSERVATION_CLIP,
        ),
        "velocity_commands": ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=OBSERVATION_CLIP,
        ),
        "joint_pos_rel": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=OBSERVATION_CLIP,
        ),
        "joint_vel_rel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=OBSERVATION_CLIP,
        ),
        "last_action": ObservationTermCfg(func=mdp.last_action, clip=OBSERVATION_CLIP),
    }
    critic_terms = {
        "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel, clip=OBSERVATION_CLIP),
        "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity, clip=OBSERVATION_CLIP),
        "velocity_commands": ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=OBSERVATION_CLIP,
        ),
        "joint_pos_rel": ObservationTermCfg(func=mdp.joint_pos_rel, clip=OBSERVATION_CLIP),
        "joint_vel_rel": ObservationTermCfg(func=mdp.joint_vel_rel, clip=OBSERVATION_CLIP),
        "last_action": ObservationTermCfg(func=mdp.last_action, clip=OBSERVATION_CLIP),
        "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel, clip=OBSERVATION_CLIP),
        "feet_contact": ObservationTermCfg(
            func=mdp.feet_contact_state,
            params={"sensor_name": "feet_ground_contact", "threshold": 0.5},
            clip=OBSERVATION_CLIP,
        ),
    }
    observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=True,
            history_length=1,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
            history_length=1,
        ),
    }

    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=0.25,
            use_default_offset=True,
        )
    }

    commands: dict[str, CommandTermCfg] = {
        "base_velocity": UniformVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(10.0, 10.0),
            rel_standing_envs=0.2,
            rel_heading_envs=1.0,
            heading_command=True,
            heading_control_stiffness=0.5,
            debug_vis=True,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.6, 1.0),
                lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(-1.57, 1.57),
                heading=(-math.pi, math.pi),
            ),
        )
    }

    events = {
        "physics_material": EventTermCfg(
            func=dr.geom_friction,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=()),
                "ranges": (0.6, 1.0),
                "operation": "abs",
                "shared_random": True,
            },
        ),
        "add_base_mass": EventTermCfg(
            func=dr.body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=()),
                "ranges": (-5.0, 5.0),
                "operation": "add",
            },
        ),
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "yaw": (-3.14, 3.14),
                },
                "velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (-0.5, 0.5),
                    "roll": (-0.5, 0.5),
                    "pitch": (-0.5, 0.5),
                    "yaw": (-0.5, 0.5),
                },
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_scale,
            mode="reset",
            params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)},
        ),
        "push_robot": EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
        ),
    }

    rewards = {
        "track_lin_vel_xy_exp": RewardTermCfg(
            func=mdp.track_lin_vel_xy_yaw_frame_exp,
            weight=1.0,
            params={"std": 0.5, "command_name": "base_velocity"},
        ),
        "track_ang_vel_z_exp": RewardTermCfg(
            func=mdp.track_ang_vel_z_world_exp,
            weight=1.0,
            params={"std": 0.5, "command_name": "base_velocity"},
        ),
        "lin_vel_z_l2": RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-1.0),
        "ang_vel_xy_l2": RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.05),
        "energy": RewardTermCfg(func=mdp.energy, weight=-1.0e-3),
        "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
        "undesired_contacts": RewardTermCfg(
            func=mdp.undesired_contacts,
            weight=-1.0,
            params={"sensor_name": "undesired_ground_contact", "threshold": 1.0},
        ),
        "fly": RewardTermCfg(
            func=mdp.fly,
            weight=-1.0,
            params={"sensor_name": "feet_ground_contact", "threshold": 1.0},
        ),
        "body_orientation_l2": RewardTermCfg(
            func=mdp.body_orientation_l2,
            weight=-2.0,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",))},
        ),
        "flat_orientation_l2": RewardTermCfg(func=mdp.flat_orientation_l2, weight=-1.0),
        "termination_penalty": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
        "feet_air_time": RewardTermCfg(
            func=mdp.feet_air_time_positive_biped,
            weight=0.15,
            params={
                "sensor_name": "feet_ground_contact",
                "threshold": 0.4,
                "command_name": "base_velocity",
            },
        ),
        "feet_slide": RewardTermCfg(
            func=mdp.feet_slide,
            weight=-0.25,
            params={
                "sensor_name": "feet_ground_contact",
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
                ),
            },
        ),
        "feet_force": RewardTermCfg(
            func=mdp.body_force,
            weight=-3.0e-3,
            params={
                "sensor_name": "feet_ground_contact",
                "threshold": 500.0,
                "max_reward": 400.0,
            },
        ),
        "feet_too_near": RewardTermCfg(
            func=mdp.biped_feet_too_near,
            weight=-2.0,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
                ),
                "threshold": 0.2,
            },
        ),
        "feet_stumble": RewardTermCfg(
            func=mdp.feet_stumble,
            weight=-2.0,
            params={"sensor_name": "feet_ground_contact"},
        ),
        "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-2.0),
        "joint_deviation_hip": RewardTermCfg(
            func=mdp.joint_deviation_l1,
            weight=-0.15,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=(
                        ".*_hip_yaw.*",
                        ".*_hip_roll.*",
                        ".*_shoulder_pitch.*",
                        ".*_elbow.*",
                    ),
                )
            },
        ),
        "joint_deviation_arms": RewardTermCfg(
            func=mdp.joint_deviation_l1,
            weight=-0.2,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=(
                        ".*waist.*",
                        ".*_shoulder_roll.*",
                        ".*_shoulder_yaw.*",
                        ".*_wrist.*",
                    ),
                )
            },
        ),
        "joint_deviation_legs": RewardTermCfg(
            func=mdp.joint_deviation_l1,
            weight=-0.02,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=(".*_hip_pitch.*", ".*_knee.*", ".*_ankle.*"),
                )
            },
        ),
    }

    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "nan_state": TerminationTermCfg(func=mdp.nan_detection),
        "torso_contact": TerminationTermCfg(
            func=mdp.illegal_contact,
            params={"sensor_name": "torso_ground_contact", "force_threshold": 1.0},
        ),
    }

    curriculum = {
        "terrain_levels": CurriculumTermCfg(
            func=mdp.terrain_levels_vel,
            params={"command_name": "base_velocity"},
        )
    }

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(
                terrain_type="generator",
                terrain_generator=deepcopy(ROUGH_TERRAINS_CFG),
                max_init_terrain_level=5,
            ),
            num_envs=4096,
            extent=2.5,
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum=curriculum,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="torso_link",
            distance=3.0,
            elevation=-5.0,
            azimuth=90.0,
        ),
        sim=SimulationCfg(
            nconmax=128,
            njmax=1500,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
            ),
        ),
        decimation=4,
        episode_length_s=20.0,
    )

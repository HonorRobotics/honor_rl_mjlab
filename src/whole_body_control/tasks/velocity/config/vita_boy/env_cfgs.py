"""VITA BOY locomotion environment configurations."""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from whole_body_control.assets.robots import get_vita_boy_robot_cfg
from whole_body_control.tasks.velocity.terrain_cfg import GRAVEL_TERRAINS_CFG
from whole_body_control.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

_FOOT_BODIES = r"^(left_ankle_roll_link|right_ankle_roll_link)$"
_FOOT_GEOMS = tuple(f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 10))


def _configure_vita_boy(cfg: ManagerBasedRlEnvCfg) -> None:
    """Attach the VITA BOY model, contact sensors, and robot-specific DR selectors."""
    cfg.sim.mujoco.ccd_iterations = 128
    cfg.sim.contact_sensor_maxmatch = 256
    cfg.scene.entities = {"robot": get_vita_boy_robot_cfg()}

    feet_ground = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="body", pattern=_FOOT_BODIES, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        track_air_time=True,
        history_length=3,
    )
    undesired_ground = ContactSensorCfg(
        name="undesired_ground_contact",
        primary=ContactMatch(
            mode="body",
            pattern=".*",
            entity="robot",
            exclude=(r".*ankle.*",),
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        history_length=3,
    )
    torso_ground = ContactSensorCfg(
        name="torso_ground_contact",
        primary=ContactMatch(mode="body", pattern="torso_link", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        history_length=3,
    )
    cfg.scene.sensors = (feet_ground, undesired_ground, torso_ground)

    cfg.events["physics_material"].params["asset_cfg"].geom_names = _FOOT_GEOMS
    cfg.events["add_base_mass"].params["asset_cfg"].body_names = ("torso_link",)

    command = cfg.commands["base_velocity"]
    command.viz.z_offset = 1.15


def _set_play_overrides(cfg: ManagerBasedRlEnvCfg) -> None:
    cfg.scene.num_envs = 32
    terrain = cfg.scene.terrain
    if terrain is not None and terrain.terrain_generator is not None:
        terrain.terrain_generator.num_rows = 2
        terrain.terrain_generator.num_cols = 10


def vita_boy_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the VITA BOY rough-terrain locomotion task."""
    cfg = make_velocity_env_cfg()
    _configure_vita_boy(cfg)

    assert cfg.scene.terrain is not None
    assert cfg.scene.terrain.terrain_generator is not None
    cfg.scene.terrain.terrain_generator.curriculum = True

    cfg.rewards["feet_air_time"].weight = 0.25
    cfg.rewards["track_lin_vel_xy_exp"].weight = 1.5
    cfg.rewards["track_ang_vel_z_exp"].weight = 1.5
    cfg.rewards["lin_vel_z_l2"].weight = -0.25

    if play:
        _set_play_overrides(cfg)
    return cfg


def vita_boy_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the VITA BOY gravel-terrain locomotion task."""
    cfg = make_velocity_env_cfg()
    _configure_vita_boy(cfg)

    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_generator = deepcopy(GRAVEL_TERRAINS_CFG)
    cfg.scene.terrain.max_init_terrain_level = 5
    cfg.observations["actor"].history_length = 10
    cfg.observations["critic"].history_length = 10
    cfg.curriculum.pop("terrain_levels", None)

    cfg.sim.njmax = 640
    cfg.sim.nconmax = None

    if play:
        _set_play_overrides(cfg)
    return cfg

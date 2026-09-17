"""Reward terms for VITA BOY locomotion tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_lin_vel_xy_yaw_frame_exp(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward planar velocity tracking in the heading-only frame."""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    velocity_yaw = quat_apply_inverse(yaw_quat(asset.data.root_link_quat_w), asset.data.root_link_lin_vel_w)
    error = torch.sum(torch.square(command[:, :2] - velocity_yaw[:, :2]), dim=1)
    return torch.exp(-error / std**2)


def track_ang_vel_z_world_exp(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward world-frame yaw-rate tracking."""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    error = torch.square(command[:, 2] - asset.data.root_link_ang_vel_w[:, 2])
    return torch.exp(-error / std**2)


def lin_vel_z_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_link_lin_vel_b[:, 2])


def ang_vel_xy_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)


def energy(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize the norm of absolute mechanical joint power."""
    asset: Entity = env.scene[asset_cfg.name]
    power = torch.abs(asset.data.qfrc_actuator[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids])
    return torch.linalg.norm(power, dim=-1)


def undesired_contacts(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    threshold: float,
) -> torch.Tensor:
    """Count non-foot bodies whose contact force exceeds the threshold."""
    sensor: ContactSensor = env.scene[sensor_name]
    if sensor.data.force_history is not None:
        force = torch.linalg.norm(sensor.data.force_history, dim=-1).amax(dim=-1)
    else:
        assert sensor.data.force is not None
        force = torch.linalg.norm(sensor.data.force, dim=-1)
    return torch.sum(force > threshold, dim=1)


def fly(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    threshold: float,
) -> torch.Tensor:
    """Penalize phases in which neither foot contacts the ground."""
    sensor: ContactSensor = env.scene[sensor_name]
    if sensor.data.force_history is not None:
        force = torch.linalg.norm(sensor.data.force_history, dim=-1).amax(dim=-1)
    else:
        assert sensor.data.force is not None
        force = torch.linalg.norm(sensor.data.force, dim=-1)
    return (torch.sum(force > threshold, dim=1) < 0.5).float()


def body_orientation_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize roll and pitch tilt of a selected body."""
    asset: Entity = env.scene[asset_cfg.name]
    body_quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids[0]]
    projected_gravity = quat_apply_inverse(body_quat, asset.data.gravity_vec_w)
    return torch.sum(torch.square(projected_gravity[:, :2]), dim=1)


def feet_air_time_positive_biped(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    threshold: float,
    command_name: str,
) -> torch.Tensor:
    """Reward sustained single-foot support while a command is active."""
    sensor: ContactSensor = env.scene[sensor_name]
    air_time = sensor.data.current_air_time
    contact_time = sensor.data.current_contact_time
    assert air_time is not None and contact_time is not None
    in_contact = contact_time > 0.0
    mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), mode_time, 0.0), dim=1).values
    reward = torch.clamp(reward, max=threshold)
    command = env.command_manager.get_command(command_name)
    assert command is not None
    moving = torch.linalg.norm(command[:, :2], dim=1) + torch.abs(command[:, 2]) > 0.1
    return reward * moving


def feet_slide(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize horizontal foot velocity while in contact."""
    sensor: ContactSensor = env.scene[sensor_name]
    if sensor.data.force_history is not None:
        contacts = torch.linalg.norm(sensor.data.force_history, dim=-1).amax(dim=-1) > 1.0
    else:
        assert sensor.data.force is not None
        contacts = torch.linalg.norm(sensor.data.force, dim=-1) > 1.0
    asset: Entity = env.scene[asset_cfg.name]
    foot_vel_xy = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :2]
    return torch.sum(torch.linalg.norm(foot_vel_xy, dim=-1) * contacts, dim=1)


def body_force(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    threshold: float = 500.0,
    max_reward: float = 400.0,
) -> torch.Tensor:
    """Penalize excessive combined vertical force on the feet."""
    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.force is not None
    force = torch.linalg.norm(sensor.data.force[..., 2], dim=-1)
    return torch.clamp(torch.where(force < threshold, 0.0, force - threshold), max=max_reward)


def biped_feet_too_near(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    threshold: float = 0.2,
) -> torch.Tensor:
    """Penalize three-dimensional foot separation below a threshold."""
    if len(asset_cfg.body_ids) != 2:
        raise ValueError("biped_feet_too_near requires exactly two selected bodies")
    asset: Entity = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    distance = torch.linalg.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return torch.clamp(threshold - distance, min=0.0)


def feet_stumble(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """Penalize feet whose horizontal contact force dominates vertical force."""
    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.force is not None
    force = sensor.data.force
    return torch.any(
        torch.linalg.norm(force[..., :2], dim=-1) > 5.0 * torch.abs(force[..., 2]),
        dim=1,
    )


def joint_deviation_l1(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize absolute deviation from the default joint positions."""
    asset: Entity = env.scene[asset_cfg.name]
    error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(error), dim=1)

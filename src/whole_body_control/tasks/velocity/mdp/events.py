"""Reset events for VITA BOY velocity locomotion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def reset_joints_by_scale(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset joints by scaling their default positions and velocities."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]
    joint_pos = asset.data.default_joint_pos[env_ids][:, asset_cfg.joint_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids][:, asset_cfg.joint_ids].clone()
    joint_pos *= torch.empty_like(joint_pos).uniform_(*position_range)
    joint_vel *= torch.empty_like(joint_vel).uniform_(*velocity_range)

    limits = asset.data.soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids]
    joint_pos.clamp_(limits[..., 0], limits[..., 1])

    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, list):
        joint_ids = torch.tensor(joint_ids, device=env.device)
    asset.write_joint_state_to_sim(
        joint_pos,
        joint_vel,
        env_ids=env_ids,
        joint_ids=joint_ids,
    )

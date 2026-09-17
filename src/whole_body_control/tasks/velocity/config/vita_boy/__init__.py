from mjlab.tasks.registry import register_mjlab_task

from whole_body_control.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import vita_boy_flat_env_cfg, vita_boy_rough_env_cfg
from .rl_cfg import vita_boy_flat_ppo_runner_cfg, vita_boy_rough_ppo_runner_cfg

register_mjlab_task(
    task_id="VitaBoy-Velocity-Rough",
    env_cfg=vita_boy_rough_env_cfg(),
    play_env_cfg=vita_boy_rough_env_cfg(play=True),
    rl_cfg=vita_boy_rough_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="VitaBoy-Velocity-Flat",
    env_cfg=vita_boy_flat_env_cfg(),
    play_env_cfg=vita_boy_flat_env_cfg(play=True),
    rl_cfg=vita_boy_flat_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

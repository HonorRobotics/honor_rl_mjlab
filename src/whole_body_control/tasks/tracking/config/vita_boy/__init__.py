from mjlab.tasks.registry import register_mjlab_task

from whole_body_control.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import vita_boy_flat_tracking_env_cfg
from .rl_cfg import vita_boy_tracking_ppo_runner_cfg

register_mjlab_task(
    task_id="VitaBoy-Tracking",
    env_cfg=vita_boy_flat_tracking_env_cfg(),
    play_env_cfg=vita_boy_flat_tracking_env_cfg(play=True),
    rl_cfg=vita_boy_tracking_ppo_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

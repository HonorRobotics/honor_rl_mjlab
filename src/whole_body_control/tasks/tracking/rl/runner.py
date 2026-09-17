from pathlib import Path
from typing import cast

from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.tracking.mdp import MotionCommand


class MotionTrackingOnPolicyRunner(MjlabOnPolicyRunner):
    """Save a pure policy ONNX; reference motions are loaded separately from NPZ."""

    def save(self, path: str, infos=None) -> None:
        super().save(path, infos)
        export_dir = Path(path).parent
        self.export_policy_to_onnx(str(export_dir), "policy.onnx")
        metadata = get_base_metadata(self.env.unwrapped, "local")
        motion_term = cast(MotionCommand, self.env.unwrapped.command_manager.get_term("motion"))
        metadata.update(
            {
                "anchor_body_name": motion_term.cfg.anchor_body_name,
                "body_names": list(motion_term.cfg.body_names),
            }
        )
        attach_metadata_to_onnx(str(export_dir / "policy.onnx"), metadata)

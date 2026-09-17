from pathlib import Path

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.rl.runner import MjlabOnPolicyRunner


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
    env: RslRlVecEnvWrapper

    def save(self, path: str, infos=None):
        super().save(path, infos)
        policy_path = Path(path).parent
        filename = "policy.onnx"
        self.export_policy_to_onnx(str(policy_path), filename)
        onnx_path = policy_path / filename
        metadata = get_base_metadata(self.env.unwrapped, "local")
        attach_metadata_to_onnx(str(onnx_path), metadata)

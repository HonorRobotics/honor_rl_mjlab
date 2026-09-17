"""ONNX policy loading."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class OnnxPolicy:
    """ONNX Runtime wrapper for feed-forward and LSTM policies."""

    def __init__(self, path: Path, observation_size: int | None = None) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Policy not found: {path}")

        os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError(
                "onnxruntime is required; run `uv sync`, then launch with `uv run python deploy/run.py`"
            ) from error

        ort.disable_telemetry_events()
        self.path = path
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.inputs = {item.name: item for item in self.session.get_inputs()}
        self.outputs = {item.name: item for item in self.session.get_outputs()}
        if "obs" not in self.inputs:
            raise ValueError(f"Policy {path} has no 'obs' input")

        self.observation_size = int(self.inputs["obs"].shape[-1])
        if observation_size is not None and self.observation_size != observation_size:
            raise ValueError(f"Policy observation size is {self.observation_size}, expected {observation_size}: {path}")

        self._state: dict[str, NDArray[np.float32]] = {}
        self.reset()

    def reset(self) -> None:
        self._state.clear()
        for name in ("h_in", "c_in"):
            if name not in self.inputs:
                continue
            shape = self.inputs[name].shape
            if not all(isinstance(dim, int) for dim in shape):
                raise ValueError(f"Dynamic recurrent input is unsupported: {name} {shape}")
            self._state[name] = np.zeros(shape, dtype=np.float32)

    def __call__(self, observation: NDArray[np.floating]) -> NDArray[np.float32]:
        observation = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        if observation.shape[1] != self.observation_size:
            raise ValueError(f"Observation size is {observation.shape[1]}, expected {self.observation_size}")

        result = self.session.run(None, {"obs": observation, **self._state})
        named_result = dict(zip(self.outputs, result, strict=True))
        if "h_out" in named_result:
            self._state["h_in"] = named_result["h_out"]
        if "c_out" in named_result:
            self._state["c_in"] = named_result["c_out"]

        action = np.asarray(named_result.get("actions", result[0]), dtype=np.float32).reshape(-1)
        if action.size != 29 or not np.all(np.isfinite(action)):
            finite = np.all(np.isfinite(action))
            raise ValueError(f"Invalid policy output: shape={action.shape}, finite={finite}")
        return np.clip(action, -100.0, 100.0)


def read_metadata(path: Path) -> dict[str, str]:
    """Read ONNX metadata without creating an inference session."""
    try:
        import onnx
    except ImportError:
        return {}
    model = onnx.load(str(path), load_external_data=False)
    return {item.key: item.value for item in model.metadata_props}

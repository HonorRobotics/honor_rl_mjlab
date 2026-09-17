# Honor RL Mjlab

<div align="center">

**Reinforcement learning environments for locomotion and whole-body motion tracking on the VITA BOY humanoid robot.**

[![MuJoCo](https://img.shields.io/badge/MuJoCo-3.10.0-007ACC)](https://github.com/google-deepmind/mujoco)
[![mjlab](https://img.shields.io/badge/mjlab-1.5.2-4B8BBE)](https://github.com/mujocolab/mjlab)
[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](LICENCE)

[English](README.md) | [简体中文](README_zh.md)

</div>

## ✨ Overview

Honor RL Mjlab is an [mjlab](https://github.com/mujocolab/mjlab)-based reinforcement learning project for the VITA BOY humanoid robot, using [MuJoCo](https://github.com/google-deepmind/mujoco) as its simulation backend. It provides an integrated workflow for training and evaluating locomotion and whole-body motion-tracking policies with RSL-RL.

### Features

- **Locomotion** — train velocity-tracking policies on gravel or curriculum-based rough terrain.
- **Motion tracking** — train whole-body policies from custom motion data.
- **MuJoCo deployment** — validate exported locomotion and tracking policies in a lightweight simulator.
- **Configurable tasks** — modular observations, rewards, events, commands, and termination conditions.
- **Developer friendly** — editable installation with a clear separation between robot assets, task logic, and learning configurations.

## 🧩 Tasks

| Task ID | Description |
| --- | --- |
| `VitaBoy-Velocity-Flat` | Velocity tracking on low-amplitude gravel terrain |
| `VitaBoy-Velocity-Rough` | Velocity tracking on curriculum-based rough terrain |
| `VitaBoy-Tracking` | Whole-body reference motion tracking |

## 📦 Setup

### Requirements

- Linux (tested on Ubuntu 22.04)
- Python 3.12 or 3.13
- NVIDIA GPU with a compatible CUDA driver for GPU training

MuJoCo 3.10.0, mjlab 1.5.2, and the remaining dependencies are managed by `uv`.

### Install

Install [uv](https://docs.astral.sh/uv/) if it is not already available, then synchronize the project environment:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cd honor_rl_mjlab
uv sync
```

Verify the installation by listing the registered VITA BOY environments:

```bash
uv run python scripts/list_envs.py --keyword VitaBoy
```

## 🚀 Usage

### Locomotion

Train a gravel-terrain policy:

```bash
uv run python scripts/train.py VitaBoy-Velocity-Flat \
  --env.scene.num-envs=4096
```

To train with the rough-terrain curriculum, replace the task ID with `VitaBoy-Velocity-Rough`.

Evaluate a trained policy:

```bash
uv run python scripts/play.py VitaBoy-Velocity-Flat \
  --checkpoint-file=logs/rsl_rl/<experiment_name>/<date_time>/model_<iteration>.pt
```

### Motion Tracking

#### Prepare Data

Convert a CSV motion file to the NPZ format used by the tracking environment:

```bash
uv run python datasets/csv_to_npz.py \
  --input-file=datasets/motions/vita_boy/dance.csv \
  --output-name=dance.npz
```

The converted file is written to `datasets/motions/vita_boy/`.

#### Train

```bash
uv run python scripts/train.py VitaBoy-Tracking \
  --motion-file=datasets/motions/vita_boy/dance.npz \
  --env.scene.num-envs=4096
```

#### Evaluate

```bash
uv run python scripts/play.py VitaBoy-Tracking \
  --motion-file=datasets/motions/vita_boy/dance.npz \
  --checkpoint-file=logs/rsl_rl/<experiment_name>/<date_time>/model_<iteration>.pt
```

> 💡 **Tip:** `4096` environments are a starting point for training. Reduce `--env.scene.num-envs` if GPU memory is limited.

Training outputs are written to:

```text
logs/rsl_rl/<experiment_name>/<date_time>/model_<iteration>.pt
```

## 🤖 MuJoCo Deployment

Training automatically exports `policy.onnx` when a checkpoint is saved. Copy
the required policies into the deployment directory:

```bash
cp logs/rsl_rl/<velocity_run>/policy.onnx \
  deploy/checkpoints/loco.onnx

cp logs/rsl_rl/<tracking_run>/policy.onnx \
  deploy/checkpoints/tracking.onnx
```

Configure each state in its own `config.yaml`, then start the simulator:

```bash
uv run python deploy/run.py
```

Press `R` to enter FIXEDPOSE, followed by `L` for LOCO or `M` for TRACKING.
Press `Esc` to exit. See the [deployment guide](deploy/README.md) for the full
configuration and controls.

## 🛠️ Development

The main package is located in `src/whole_body_control`. These are the common starting points for secondary development:

```text
honor_rl_mjlab/
├── deploy/                         # MuJoCo policy deployment
├── datasets/                       # Motion data and conversion tools
├── scripts/                        # Training and evaluation
└── src/whole_body_control/
    ├── assets/robots/vita_boy/        # MuJoCo model and robot configuration
    └── tasks/
        ├── velocity/               # Locomotion
        │   ├── config/vita_boy/       # Environment and agent configuration
        │   ├── mdp/                # Task logic
        │   └── velocity_env_cfg.py
        └── tracking/               # Motion tracking
            ├── config/vita_boy/       # Environment and agent configuration
            ├── mdp/                # Task logic
            └── tracking_env_cfg.py
```

Typical customization points:

1. Update the robot MJCF and actuator parameters under `assets/robots/vita_boy/`.
2. Adjust observations, rewards, commands, events, or termination conditions under the relevant `tasks/*/mdp/` directory.
3. Tune environment and terrain settings in `*_env_cfg.py` and `terrain_cfg.py`.
4. Tune RSL-RL hyperparameters in `tasks/*/config/vita_boy/rl_cfg.py`.
5. Register a new task in the corresponding `config/vita_boy/__init__.py`, then confirm it with `uv run python scripts/list_envs.py --keyword VitaBoy`.

## 📄 License

This project is released under the [Apache License 2.0](LICENCE).

## 🙏 Thanks

Honor RL Mjlab is inspired by and built upon these excellent open-source projects:

| Project | Contribution |
| :--- | :--- |
| **[mjlab](https://github.com/mujocolab/mjlab)** | A lightweight, modular framework for reinforcement learning with MuJoCo. |
| **[MuJoCo](https://github.com/google-deepmind/mujoco)** | A fast and accurate physics engine for robotics simulation. |
| **[LeggedLab](https://github.com/Hellod035/LeggedLab)** | An open-source framework for training legged locomotion policies. |
| **[BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking)** | A learning framework for whole-body humanoid motion tracking. |

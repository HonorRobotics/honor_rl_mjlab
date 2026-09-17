# Honor RL Mjlab

<div align="center">

**面向元气仔（VITA BOY）人形机器人的运动控制与全身动作跟踪强化学习环境。**

[![MuJoCo](https://img.shields.io/badge/MuJoCo-3.10.0-007ACC)](https://github.com/google-deepmind/mujoco)
[![mjlab](https://img.shields.io/badge/mjlab-1.5.2-4B8BBE)](https://github.com/mujocolab/mjlab)
[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](LICENCE)

[English](README.md) | [简体中文](README_zh.md)

</div>

## ✨ 简介

Honor RL Mjlab 是一个面向元气仔人形机器人的强化学习项目，基于 [mjlab](https://github.com/mujocolab/mjlab) 构建，并使用 [MuJoCo](https://github.com/google-deepmind/mujoco) 作为仿真后端，提供使用 RSL-RL 训练和评估运动控制与全身动作跟踪策略的一体化工作流。

### 特性

- **运动控制** — 在碎石地形或基于课程学习的粗糙地形上训练速度跟踪策略。
- **动作跟踪** — 使用自定义动作数据训练全身动作跟踪策略。
- **MuJoCo 部署** — 在轻量级仿真器中验证导出的运动控制与动作跟踪策略。
- **灵活配置** — 以模块化方式配置观测、奖励、事件、命令和终止条件。
- **便于二次开发** — 支持可编辑安装，并清晰分离机器人资产、任务逻辑与训练配置。

## 🧩 任务

| 任务 ID | 说明 |
| --- | --- |
| `VitaBoy-Velocity-Flat` | 低幅度碎石地形速度跟踪 |
| `VitaBoy-Velocity-Rough` | 基于课程学习的粗糙地形速度跟踪 |
| `VitaBoy-Tracking` | 全身参考动作跟踪 |

## 📦 安装

### 要求

- Linux（已在 Ubuntu 22.04 上测试通过）
- Python 3.12 或 3.13
- 进行 GPU 训练时，需要 NVIDIA GPU 及兼容的 CUDA 驱动

MuJoCo 3.10.0、mjlab 1.5.2 及其余依赖均由 `uv` 管理。

### 安装项目

如果尚未安装 [uv](https://docs.astral.sh/uv/)，请先完成安装，再同步项目环境：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cd honor_rl_mjlab
uv sync
```

列出已注册的元气仔环境，验证安装是否成功：

```bash
uv run python scripts/list_envs.py --keyword VitaBoy
```

## 🚀 使用

### 运动控制

训练碎石地形运动策略：

```bash
uv run python scripts/train.py VitaBoy-Velocity-Flat \
  --env.scene.num-envs=4096
```

如需使用粗糙地形课程训练，请将任务 ID 替换为 `VitaBoy-Velocity-Rough`。

评估已训练的策略：

```bash
uv run python scripts/play.py VitaBoy-Velocity-Flat \
  --checkpoint-file=logs/rsl_rl/<experiment_name>/<date_time>/model_<iteration>.pt
```

### 动作跟踪

#### 准备数据

将 CSV 动作文件转换为跟踪环境使用的 NPZ 格式：

```bash
uv run python datasets/csv_to_npz.py \
  --input-file=datasets/motions/vita_boy/dance.csv \
  --output-name=dance.npz
```

转换后的文件保存在 `datasets/motions/vita_boy/`。

#### 训练

```bash
uv run python scripts/train.py VitaBoy-Tracking \
  --motion-file=datasets/motions/vita_boy/dance.npz \
  --env.scene.num-envs=4096
```

#### 评估

```bash
uv run python scripts/play.py VitaBoy-Tracking \
  --motion-file=datasets/motions/vita_boy/dance.npz \
  --checkpoint-file=logs/rsl_rl/<experiment_name>/<date_time>/model_<iteration>.pt
```

> 💡 **提示：** `4096` 个环境是建议的训练起点；如果 GPU 显存不足，可适当减小 `--env.scene.num-envs`。

训练结果默认保存在：

```text
logs/rsl_rl/<experiment_name>/<date_time>/model_<iteration>.pt
```

## 🤖 MuJoCo 部署

训练保存检查点时会自动导出 `policy.onnx`。将需要验证的策略复制到部署目录：

```bash
cp logs/rsl_rl/<velocity_run>/policy.onnx \
  deploy/checkpoints/loco.onnx

cp logs/rsl_rl/<tracking_run>/policy.onnx \
  deploy/checkpoints/tracking.onnx
```

根据需要修改各状态目录中的 `config.yaml`，然后启动仿真：

```bash
uv run python deploy/run.py
```

启动后先按 `R` 进入 FIXEDPOSE，再按 `L` 进入 LOCO，或按 `M` 进入
TRACKING。按 `Esc` 退出。完整配置和按键说明请参阅[部署指南](deploy/README.md)。

## 🛠️ 二次开发

项目的主要代码位于 `src/whole_body_control`。二次开发时可重点关注以下目录与文件：

```text
honor_rl_mjlab/
├── deploy/                         # MuJoCo 策略部署
├── datasets/                       # 动作数据与转换工具
├── scripts/                        # 训练与评估入口
└── src/whole_body_control/
    ├── assets/robots/vita_boy/        # MuJoCo 模型与机器人配置
    └── tasks/
        ├── velocity/               # 运动控制
        │   ├── config/vita_boy/       # 环境与智能体配置
        │   ├── mdp/                # 任务逻辑
        │   └── velocity_env_cfg.py
        └── tracking/               # 动作跟踪
            ├── config/vita_boy/       # 环境与智能体配置
            ├── mdp/                # 任务逻辑
            └── tracking_env_cfg.py
```

常见的定制入口：

1. 在 `assets/robots/vita_boy/` 中更新机器人 MJCF 与执行器参数。
2. 在对应的 `tasks/*/mdp/` 目录中调整观测、奖励、命令、事件或终止条件。
3. 在 `*_env_cfg.py` 和 `terrain_cfg.py` 中调整环境与地形参数。
4. 在 `tasks/*/config/vita_boy/rl_cfg.py` 中调整 RSL-RL 超参数。
5. 在对应的 `config/vita_boy/__init__.py` 中注册新任务，再运行 `uv run python scripts/list_envs.py --keyword VitaBoy` 确认任务可以正常加载。

## 📄 许可

本项目采用 [Apache License 2.0](LICENCE) 开源。

## 🙏 致谢

Honor RL Mjlab 的实现离不开以下优秀开源项目的启发与支持：

| 项目 | 贡献 |
| :--- | :--- |
| **[mjlab](https://github.com/mujocolab/mjlab)** | 面向 MuJoCo 强化学习的轻量化、模块化开发框架。 |
| **[MuJoCo](https://github.com/google-deepmind/mujoco)** | 面向机器人仿真的高效、高精度物理引擎。 |
| **[LeggedLab](https://github.com/Hellod035/LeggedLab)** | 面向足式机器人运动策略训练的开源框架。 |
| **[BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking)** | 面向人形机器人全身动作跟踪的学习框架。 |

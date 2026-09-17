# VITA BOY MuJoCo deployment

## Quick Start

From the `honor_rl_mjlab` root:

```bash
uv sync
uv run python deploy/run.py
```

Enter FIXEDPOSE with `R`, then select LOCO with `L` or TRACKING with `M`.

## Configuration

Deployment policies are stored in `deploy/checkpoints`:

```text
deploy/checkpoints/
├── loco.onnx
└── tracking.onnx
```

Overwrite the corresponding file to replace a policy. Each state has an
independent configuration:

```text
deploy/fsm/damping/config.yaml
deploy/fsm/fixedpose/config.yaml
deploy/fsm/loco/config.yaml
deploy/fsm/tracking/config.yaml
```

Policy and motion paths in these files are relative to `deploy`.
Simulation timing is configured in `deploy/config.toml`. Use another simulation
configuration with `--config path/to/config.toml`.

## Controls

| Key | Action |
| --- | --- |
| `P` | DAMPING |
| `R` | FIXEDPOSE |
| `L` | LOCO |
| `M` | TRACKING |
| Hold `↑` / `↓` | Forward/backward |
| Hold `←` / `→` | Left/right |
| Hold `Q` / `E` | Turn left/right |
| `Esc` | Exit |

State commands are applied when the key is released.

## State Transitions

Enter FIXEDPOSE before entering LOCO or TRACKING from DAMPING. LOCO and
TRACKING can switch directly between each other. If the robot falls in a policy
state, the safety check switches it to DAMPING.

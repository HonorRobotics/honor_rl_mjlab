"""RSL-RL configurations for VITA BOY locomotion."""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def _ppo_algorithm_cfg() -> RslRlPpoAlgorithmCfg:
    return RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
    )


def _model_cfg(
    hidden_dims: tuple[int, ...],
    *,
    recurrent: bool,
    stochastic: bool,
) -> RslRlModelCfg:
    return RslRlModelCfg(
        hidden_dims=hidden_dims,
        activation="elu",
        obs_normalization=False,
        class_name="RNNModel" if recurrent else "MLPModel",
        rnn_type="lstm" if recurrent else None,
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        distribution_cfg=(
            {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            }
            if stochastic
            else None
        ),
    )


def vita_boy_flat_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Feed-forward PPO configuration for gravel terrain."""
    hidden_dims = (512, 256, 128)
    return RslRlOnPolicyRunnerCfg(
        seed=42,
        actor=_model_cfg(hidden_dims, recurrent=False, stochastic=True),
        critic=_model_cfg(hidden_dims, recurrent=False, stochastic=False),
        algorithm=_ppo_algorithm_cfg(),
        logger="tensorboard",
        experiment_name="vita_boy_velocity",
        save_interval=1000,
        num_steps_per_env=24,
        max_iterations=50001,
        clip_actions=None,
    )


def vita_boy_rough_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Recurrent PPO configuration for rough terrain."""
    hidden_dims = (256, 256, 128)
    return RslRlOnPolicyRunnerCfg(
        seed=42,
        actor=_model_cfg(hidden_dims, recurrent=True, stochastic=True),
        critic=_model_cfg(hidden_dims, recurrent=True, stochastic=False),
        algorithm=_ppo_algorithm_cfg(),
        logger="tensorboard",
        experiment_name="vita_boy_velocity",
        save_interval=1000,
        num_steps_per_env=24,
        max_iterations=50001,
        clip_actions=None,
    )


def vita_boy_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Backward-compatible alias for the flat-terrain configuration."""
    return vita_boy_flat_ppo_runner_cfg()

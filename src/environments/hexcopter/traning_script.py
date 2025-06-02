from __future__ import annotations

import argparse
import functools
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from absl import logging
from brax import envs  # noqa: F401 – side‑effect import registers envs
from brax.training import types
from brax.training.agents.ppo.train import train as ppo_train
from brax.training.agents.ppo.networks import make_ppo_networks
from jax import numpy as jp
import jax
from ml_collections.config_dict import ConfigDict  # noqa: F401 – for user configs
from tensorboardX import SummaryWriter

from environments.hexcopter.config import ExperimentConfig, save_config
from environments.hexcopter.hexcopter import (
    Hexcopter3DEnv,
    get_domain_rand_fn_v2,
)

# -----------------------------------------------------------------------------
# Constants & global initialisation
# -----------------------------------------------------------------------------

logging.set_verbosity(logging.ERROR)

JAX_CACHE_DIR = "/tmp/jax_cache"

jax.config.update("jax_compilation_cache_dir", JAX_CACHE_DIR)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# NB: The following key exists on recent jax versions only.
# jax.config.update("jax_compilation_cache_enable_xla_caches", True)

BASE_DIR = Path(__file__).resolve().parent
Metrics = types.Metrics  # Alias for brevity

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def create_run_dirs(config: ExperimentConfig) -> tuple[Path, Path]:
    """Create log + checkpoint directories and return them."""
    run_base = BASE_DIR / config.logging.log_dir_base / config.run_name
    log_dir = run_base / "tb_logs"
    ckpt_dir = BASE_DIR / config.logging.checkpoint_dir_base / config.run_name

    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return log_dir, ckpt_dir


def make_network_factory(cfg: ExperimentConfig) -> Callable:
    return functools.partial(
        make_ppo_networks,
        policy_hidden_layer_sizes=cfg.network.policy_hidden_layer_sizes,
        value_hidden_layer_sizes=cfg.network.value_hidden_layer_sizes,
        policy_obs_key="actor",
        value_obs_key="critic"

    )


def make_train_fn(cfg: ExperimentConfig, network_factory: Callable, *,
                  checkpoint_dir: Path, restore_checkpoint: Optional[str]) -> Callable:
    """Partially‑applied Brax PPO ``train`` for this experiment."""

    return functools.partial(
        ppo_train,
        num_timesteps=cfg.train.num_timesteps,
        num_envs=cfg.train.num_envs,
        episode_length=cfg.train.episode_length,
        network_factory=network_factory,
        normalize_observations=cfg.train.normalize_observations,
        learning_rate=cfg.train.learning_rate,
        entropy_cost=cfg.train.entropy_cost,
        discounting=cfg.train.discounting,
        unroll_length=cfg.train.unroll_length,
        batch_size=cfg.train.batch_size,
        num_minibatches=cfg.train.num_minibatches,
        num_updates_per_batch=cfg.train.num_updates_per_batch,
        num_evals=cfg.train.num_evals,
        seed=cfg.train.seed,
        gae_lambda=cfg.train.gae_lambda,
        clipping_epsilon=cfg.train.clip_epsilon,
        reward_scaling=cfg.train.reward_scaling,
        save_checkpoint_path=str(checkpoint_dir),
        action_repeat=cfg.train.action_repeat,
        restore_checkpoint_path=restore_checkpoint,
        restore_value_fn=cfg.train.restore_value_fn,
        randomization_fn=get_domain_rand_fn_v2(cfg.env.stage_config.env_rand),
    )

# -----------------------------------------------------------------------------
# Core training routine
# -----------------------------------------------------------------------------

def train_drone(cfg: ExperimentConfig, *, restore_checkpoint: Optional[str] = None) -> None:
    """Runs the PPO training loop using the provided configuration."""

    print("--- Configuration (condensed) ---")
    # Directories & writers
    log_dir, ckpt_dir = create_run_dirs(cfg)
    writer = SummaryWriter(logdir=str(log_dir))

    # Environment & factories
    env = Hexcopter3DEnv(config=cfg.env)
    network_factory = make_network_factory(cfg)
    train_fn = make_train_fn(cfg, network_factory, checkpoint_dir=ckpt_dir, restore_checkpoint=restore_checkpoint)

    # Progress callback
    def progress(step: int, metrics: Metrics):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pct = step / cfg.train.num_timesteps
        ar = cfg.train.action_repeat
        print(f"[{timestamp}] Step {step:>10} ({pct:.2%}) | EvalReward {metrics['eval/episode_reward']:>8.2f} | AvgLen {metrics['eval/avg_episode_length'] / ar:>6.2f}")

        # TensorBoard logging
        for k, v in metrics.items():
            if jp.isscalar(v) and jp.isfinite(v):
                writer.add_scalar(k, (v / ar) if k == "eval/avg_episode_length" else v, step // ar)
        writer.flush()

    # Kick off training
    print("--- Starting training ---")
    print(f"JAX devices: {jax.devices()}")
    train_fn(environment=env, progress_fn=progress)  # discard params – saved in ckpts
    print("--- Training complete ---")

# -----------------------------------------------------------------------------
# CLI entrypoint
# -----------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a PPO agent in Hexcopter3DEnv")
    p.add_argument("--run-name", default="brax_ppo_" + datetime.now().strftime("%Y%m%d_%H%M%S"), help="Directory suffix for this training run")
    p.add_argument("--restore", type=str, help="Path to a checkpoint to resume from", default=None)
    return p.parse_args()

def main() -> None:
    args = _parse_args()

    # Populate *default* experiment config. Users can patch it externally before call.
    cfg = ExperimentConfig()
    cfg.run_name = args.run_name

    # Persist full config next to checkpoints for reproducibility
    _, ckpt_dir = create_run_dirs(cfg)
    save_config(cfg, ckpt_dir / "run_config.json")

    train_drone(cfg, restore_checkpoint=args.restore)


if __name__ == "__main__":
    main()

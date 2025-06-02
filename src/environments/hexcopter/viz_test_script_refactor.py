import argparse
import random
from typing import Any, Dict, Optional, Tuple
import IPython

from brax.training import acting
from environments.hexcopter.config import ExperimentConfig, Schedule
from environments.hexcopter.hexcopter import Hexcopter3DEnv, get_domain_rand_fn_v2
from environments.hexcopter.state_interfaces import AugmentedEnvState, AugmentedPipelineState, CurriculumProgressInfo
import json
from brax.training.agents.ppo import checkpoint
import jax
import jax.numpy as jp
import orbax.checkpoint as ocp  # Import Orbax
from brax import envs
from brax.io import html
from brax.training.agents.ppo import networks as ppo_networks
import functools
from pathlib import Path
import time
import json
import time
from typing import Set
from pynput import keyboard
from jax.core import ShapedArray

import mujoco
import numpy as np
from mujoco import mjx
import mujoco.viewer
import jax.numpy as jp
import jax

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets


import os
from brax.training.acme import running_statistics

import logging
np.set_printoptions(precision=4, suppress=True)



# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


def _create_key_listener() -> set[keyboard.Key]:
    """Starts a background listener that records currently pressed keys."""
    pressed: set[keyboard.Key] = set()

    def _on_press(key):  # noqa: ANN001
        pressed.add(key)

    def _on_release(key):  # noqa: ANN001
        pressed.discard(key)
        if key == keyboard.Key.esc:
            return False  # Stop the listener

    listener = keyboard.Listener(on_press=_on_press, on_release=_on_release)
    listener.start()
    return pressed

def _print_done_reason(metrics: Dict[str, jp.ndarray]) -> None:
    """Pretty‑print the termination reason encoded in `metrics`."""
    reasons = {
        "done/drone_outside_playground": "Drone outside playground",
        "done/ball_outside_playground": "Ball outside playground",
        "done/ball_hit_drone_side": "Ball hit drone side",
        "done/drone_hit_ball": "Drone hit ball",
        "done/ball_hit_goal_side": "Ball hit goal side",
    }
    for key, message in reasons.items():
        if metrics.get(key, 0):
            print(f"Done: {message}")

def _find_latest_checkpoint(runs_root: Path) -> Path:
    """Best‑effort search for the newest checkpoint directory inside `runs_root`."""
    latest_run = max((p for p in runs_root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)
    latest_checkpoint = max((p for p in latest_run.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)
    return latest_checkpoint


def _load_policy(checkpoint_dir: Path):
    """Restore a policy from `checkpoint_dir`."""
    make_network = functools.partial(ppo_networks.make_ppo_networks, policy_obs_key = "actor", value_obs_key = "critic")
    return checkpoint.load_policy(str(checkpoint_dir), deterministic=True, network_factory=make_network)



# -----------------------------------------------------------------------------
# Core logic
# -----------------------------------------------------------------------------


def visualize(env: Hexcopter3DEnv, inference_fn, action_repeat: int) -> None:
    """Launch an interactive MuJoCo viewer driven by `inference_fn`."""
    progress = CurriculumProgressInfo.get_default_with_progress(1.0)
    # env.setup_viz_regions(progress)

    model: mujoco.MjModel = env.mj_model
    mjdata = mujoco.MjData(model)

    # JIT compile hot paths
    inference_fn = jax.jit(jax.vmap(inference_fn))
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    rng = jax.random.PRNGKey(random.randint(0, 1 << 16)).reshape(1, 2)
    state: AugmentedEnvState = reset_fn(rng, progress)

    current_keys = _create_key_listener()
    total_reward, local_len = 0.0, 0

    with mujoco.viewer.launch_passive(model, mjdata) as viewer:
        viewer.cam.lookat[:] = 0, 0, 1
        viewer.cam.distance = 9
        viewer.cam.elevation = -30

        while viewer.is_running():
            step_start = time.time()
            if keyboard.Key.enter in current_keys or jp.isclose(state.done, 1):
                _print_done_reason(state.metrics)
                logging.info("Episode finished | steps=%d reward=%.2f", local_len, total_reward)
                rng = jax.random.PRNGKey(random.randint(0, 1 << 16)).reshape(1, 2)
                state = reset_fn(rng, progress)
                total_reward, local_len = 0.0, 0

            rng = jax.random.PRNGKey(random.randint(0, 1 << 16)).reshape(1, 2)
            # obs = state.obs if isinstance(state.obs, jax.Array) else state.obs["actor"]
            obs = state.obs
            # print(obs)
            action = inference_fn(obs, rng)[0]
            state = step_fn(state, action)

            # Ang diff
            print("---- ")
            next_sensor_data = state.pipeline_state.sensor_data_realtime
            orientation_xyzw = next_sensor_data.drone_imu_orientation_quat_wxyz[0][jp.array([1, 2, 3, 0])]
            print("Target:", state.pipeline_state.target_orientation_xyzw)
            print("Current:", orientation_xyzw)
            print("Angle error:", jp.rad2deg(env._ang_diff(state.pipeline_state.target_orientation_xyzw, orientation_xyzw)))

            # obs :ActorFullObservationWithBallAndHistory= env.actor_observation_model.from_array(state.obs['actor'])
            # print("Action Hist:", state.pipeline_state.action_history)
            # print("-----")

            total_reward += state.reward
            local_len += 1

            # Sync MuJoCo state
            pipeline_state = state.pipeline_state.original_pipeline_state
            mjdata.qpos[:] = np.asarray(pipeline_state.qpos)
            mjdata.qvel[:] = np.asarray(pipeline_state.qvel)
            mjdata.ctrl[:] = np.asarray(pipeline_state.ctrl)
            mujoco.mj_forward(model, mjdata)
            viewer.sync()

            # Real‑time pacing
            dt = model.opt.timestep * action_repeat - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)


def evaluate(env, inference_fn, exp_config: ExperimentConfig, num_eval_envs: int = 128) -> None:
    """Run a vectorised evaluation and print aggregate metrics."""
    evaluator = acting.EvaluatorProgress(
        eval_env=env,
        eval_policy_fn=lambda _: inference_fn,
        num_eval_envs=num_eval_envs,
        episode_length=exp_config.train.episode_length,
        action_repeat=exp_config.train.action_repeat,
        key=jax.random.PRNGKey(random.randint(0, 1 << 16)),
    )

    metrics = evaluator.run_evaluation(
        policy_params=None,
        training_metrics={},
        curriculum_progress_info=CurriculumProgressInfo.get_default_with_progress(1.0),
    )

    logging.info(
        "Eval reward: %.2f | Avg episode length: %.2f",
        metrics["eval/episode_reward"],
        metrics["eval/avg_episode_length"] / exp_config.train.action_repeat,
    )

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> None:  # noqa: D401
    logging.basicConfig(format="%(levelname)s | %(message)s", level=logging.INFO)

    parser = argparse.ArgumentParser(description="Visualise or evaluate a trained Drone PPO agent.")
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint directory. If omitted, the most recent one is used.")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path(__file__).resolve().parent / "runs",
        help="Root directory containing run folders.",
    )
    parser.add_argument(
        '--use-default-config',
        action='store_true',
        help='Use the checkpoint config instead of the default config.',
    )
    parser.add_argument("--mode", choices=("viz", "eval"), default="viz", help="Run mode.")
    args = parser.parse_args()

    logging.basicConfig(format="%(levelname)s | %(message)s", level=logging.INFO)

    ckpt_dir = args.checkpoint or _find_latest_checkpoint(args.runs_root)
    logging.info("Using checkpoint: %s", ckpt_dir)
    print(ckpt_dir)
    inference_fn = _load_policy(ckpt_dir)

    if args.use_default_config:
        exp_config = ExperimentConfig()
    else:
        print("Using default config from checkpoint")
        with open(ckpt_dir.parent / "run_config.json") as f:
            exp_config = ExperimentConfig.model_validate_json(f.read())
        

    env = Hexcopter3DEnv(config=exp_config.env)
    rand_fn = get_domain_rand_fn_v2(exp_config.env.stage_config.env_rand)

    print(exp_config.env.action_delay_discrete, exp_config.env.action_delay)
    if args.mode == "viz":
        v_rand_fn = functools.partial(rand_fn, rng=jax.random.PRNGKey(random.randint(0, 1 << 16)).reshape(1, 2)) if rand_fn is not None else None
        env = envs.training.wrap_progress(
            env,
            episode_length=exp_config.train.episode_length,
            action_repeat=exp_config.train.action_repeat,
            randomization_fn=v_rand_fn,
        )
        visualize(env, inference_fn, exp_config.train.action_repeat)
    else:
        num_eval_envs = 128
        v_rand_fn = functools.partial(rand_fn, rng=jax.random.split(jax.random.PRNGKey(random.randint(0, 1 << 16)), num_eval_envs).reshape(-1, 2)) if rand_fn is not None else None
        env = envs.training.wrap_progress(
            env,
            episode_length=exp_config.train.episode_length,
            action_repeat=exp_config.train.action_repeat,
            randomization_fn=v_rand_fn,
        )
        evaluate(env, inference_fn, exp_config, num_eval_envs)


if __name__ == "__main__":
    main()

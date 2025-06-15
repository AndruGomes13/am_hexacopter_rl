import functools
import logging
import random
import time
from typing import Dict, Set
import IPython
from environments.hexcopter_t_yaw.config import ExperimentConfig
from environments.hexcopter_t_yaw.observation_models import FullDroneObservation
from environments.hexcopter_t_yaw.state_interfaces import (
    AugmentedEnvState,
    AugmentedPipelineState,
    CurriculumProgressInfo,
    create_parse_sensordata_fn,
)
from brax import envs
from brax.io import mjcf
from environments.hexcopter_t_yaw.hexcopter import Hexcopter3DEnv, get_domain_rand_fn_v2
from environments.hexcopter_t_yaw.env_utils import get_env_xml_path

from pynput import keyboard

import mujoco
import numpy as np
from mujoco import mjx
import mujoco.viewer
import jax.numpy as jp
import jax


import os

np.set_printoptions(precision=3, suppress=True)

# os.environ["JAX_PLATFORM_NAME"] = "cpu"

XML_MODEL_PATH = get_env_xml_path().as_posix()

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


def visualize_control(env: Hexcopter3DEnv, action_repeat: int) -> None:
    """Launch an interactive MuJoCo viewer driven by `inference_fn`."""
    progress = CurriculumProgressInfo.get_default_with_progress(1.0)
    # env.setup_viz_regions(progress)

    model: mujoco.MjModel = env.mj_model
    mjdata = mujoco.MjData(model)

    # JIT compile hot paths
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    rng = jax.random.PRNGKey(random.randint(0, 1 << 16)).reshape(1, 2)
    state: AugmentedEnvState = reset_fn(rng, progress)

    current_keys = _create_key_listener()
    total_reward, local_len = 0.0, 0

    theta_1 = 0
    theta_2 = 0
    with mujoco.viewer.launch_passive(model, mjdata) as viewer:
        viewer.cam.lookat[:] = 0, 0, 1
        viewer.cam.distance = 9
        viewer.cam.elevation = -30

        action = jp.zeros(model.nu, dtype=jp.float32).reshape(1,-1)
        action = action.at[2, 0].set(-1)
        
        while viewer.is_running():
            step_start = time.time()
            if keyboard.Key.enter in current_keys :#or jp.isclose(state.done, 1):
                print("Resetting environment...")
                # _print_done_reason(state.metrics)
                # logging.info("Episode finished | steps=%d reward=%.2f", local_len, total_reward)
                rng = jax.random.PRNGKey(random.randint(0, 1 << 16)).reshape(1, 2)
                state = reset_fn(rng, progress)
                total_reward, local_len = 0.0, 0

            rng = jax.random.PRNGKey(random.randint(0, 1 << 16)).reshape(1, 2)

            # --- Control With Key Press ---
            d = 1

            if keyboard.Key.up in current_keys or keyboard.Key.down in current_keys:
                theta_1 += d if keyboard.Key.up in current_keys else -d
                theta_1 = np.clip(theta_1, 0.0, 180.0)

            if keyboard.Key.right in current_keys or keyboard.Key.left in current_keys:
                theta_2 += d if keyboard.Key.right in current_keys else -d
                theta_2 = np.clip(theta_2, -10, 45.0)
            np.set_printoptions(precision=2, suppress=True)
            # action = jp.array([[0, 0, 0, 0, 0, 0, theta_1, theta_1 + theta_2]]) / 180 * np.pi  # Convert to radians
            # last_ctrl= state.pipeline_state.last_ctrl
            # action = last_ctrl.at[5].set((last_ctrl[5]-last_ctrl[4])/(jp.pi/2))  # Arm 1 pitch
            # action = last_ctrl.at[4].set(last_ctrl[4]/jp.pi)  # Arm 1 pitch
            # print(action)
            # print(theta_1, theta_2)
            # state = step_fn(state, action)
            # qpos=  state.pipeline_state.original_pipeline_state.qpos
            # qpos = qpos.at[2].set(1)  # Adjust height for visualization
            # new_pipeline_orig = state.pipeline_state.original_pipeline_state.replace(qpos=qpos)
            # new_pipeline = state.pipeline_state.replace(original_pipeline_state=new_pipeline_orig)
            # state = state.replace(pipeline_state=new_pipeline)
            
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

def visualize(env: Hexcopter3DEnv) -> None:
    model = mujoco.MjModel.from_xml_path(get_env_xml_path().as_posix())
    # model = env.mj_model
    mjdata = mujoco.MjData(model)
    # mjdata.qpos[3] = 1.0  # Adjust height for visualization
    # mujoco.mj_forward(model, mjdata)
    mujoco.viewer.launch(model, mjdata)

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




def main() -> None:  # noqa: D401
    logging.basicConfig(format="%(levelname)s | %(message)s", level=logging.INFO)
    exp_config = ExperimentConfig()

    env = Hexcopter3DEnv(config=exp_config.env)  
    # visualize(env)

    # exit()
    rand_fn = get_domain_rand_fn_v2(exp_config.env.stage_config.env_rand)

    print(exp_config.env.action_delay_discrete, exp_config.env.action_delay)
    v_rand_fn = functools.partial(rand_fn, rng=jax.random.PRNGKey(random.randint(0, 1 << 16)).reshape(1, 2)) if rand_fn is not None else None
    env = envs.training.wrap_progress(
        env,
        episode_length=exp_config.train.episode_length,
        action_repeat=exp_config.train.action_repeat,
        randomization_fn=v_rand_fn,
    )
    visualize_control(env, exp_config.train.action_repeat)

if __name__ == "__main__":
    main()
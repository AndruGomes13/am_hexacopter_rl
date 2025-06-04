import functools
import logging
import random
import time
from typing import Dict, Set
import IPython
from environments.hexcopter.config import ExperimentConfig
from environments.hexcopter.observation_models import FullDroneObservation
from environments.hexcopter.state_interfaces import (
    AugmentedEnvState,
    AugmentedPipelineState,
    CurriculumProgressInfo,
    create_parse_sensordata_fn,
)
from brax import envs
from brax.io import mjcf
from environments.hexcopter.hexcopter import Hexcopter3DEnv, get_domain_rand_fn_v2
from environments.hexcopter.env_utils import get_env_xml_path

# from models.skydio_x2 import get_x2_3d_xml
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


def visualize(env: Hexcopter3DEnv, action_repeat: int) -> None:
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

    with mujoco.viewer.launch_passive(model, mjdata) as viewer:
        viewer.cam.lookat[:] = 0, 0, 1
        viewer.cam.distance = 9
        viewer.cam.elevation = -30

        while viewer.is_running():
            step_start = time.time()
            if keyboard.Key.enter in current_keys :#or jp.isclose(state.done, 1):
                _print_done_reason(state.metrics)
                logging.info("Episode finished | steps=%d reward=%.2f", local_len, total_reward)
                rng = jax.random.PRNGKey(random.randint(0, 1 << 16)).reshape(1, 2)
                state = reset_fn(rng, progress)
                total_reward, local_len = 0.0, 0

            rng = jax.random.PRNGKey(random.randint(0, 1 << 16)).reshape(1, 2)
            obs = state.obs
            action = jp.zeros(7)
            state = step_fn(state, action)

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



def main_passive(current_keys: Set):
    # Load the MuJoCo model and create the simulation data.
    model = mujoco.MjModel.from_xml_path(XML_MODEL_PATH)
    data = mujoco.MjData(model)
    # data.qpos[1] = 0
    # mujoco.mj_forward(model, data)
    # IPython.embed()
    # exit()
    # --- Setup PID control ---
    # env = DroneTracking3DEnv()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()

        viewer.cam.lookat[:] = 0, 0, 1
        viewer.cam.distance = 9
        viewer.cam.elevation = -30
        A = get_allocation_matrix()
        A_inv = jp.linalg.inv(A)

        while viewer.is_running() and time.time() - start:
            step_start = time.time()
            # --- Control ---
            action = np.array([0, 0, 0, 0])

            s = 0.3
            if keyboard.Key.left in current_keys:
                action[0] = 1
            if keyboard.Key.right in current_keys:
                action[1] = 1
            if keyboard.Key.right in current_keys:
                action[2] = 1
            if keyboard.Key.right in current_keys:
                action[3] = 1

            action = action * 2 * s - 1
            action = jp.array(action)

            data.ctrl = action.flatten()
            print("Action: ", action.flatten())
            mujoco.mj_step(model, data)

            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


def main_regular():
    model = mujoco.MjModel.from_xml_path(XML_MODEL_PATH)
    data = mujoco.MjData(model)
    sys = mjcf.load_model(model)
    data_mjx = mjx.make_data(sys)
    IPython.embed()
    exit()
    mujoco.viewer.launch(model, data)


def add_text(data, viewer, input, t: float = 0):
    # create an invisibale geom and add label on it
    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_LABEL,
        size=np.array([0.2, 0.2, 0.2]),  # label_size
        # pos=data.qpos[:3]
        # + np.array(
        #     [0.0, 0.0, t]
        # ),  # lebel position, here is 1 meter above the root joint
        pos=np.array([0.0, 0.0, t], dtype=float),  # lebel position, here is 1 meter above the root joint
        mat=np.eye(3).flatten(),  # label orientation, here is no rotation
        rgba=np.array([0, 0, 0, 0]),  # invisible
    )
    geom.label = input  # receive string input only
    viewer.user_scn.ngeom += 1


def main_brax(current_keys):
    env = Hexcopter3DEnv()
    jit_reset = env.reset
    jit_step = env.step
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    rng = jax.random.PRNGKey(0)
    state: AugmentedEnvState = jit_reset(rng)

    model = env.mj_model
    mjdata = mujoco.MjData(model)
    i = 0

    # A = get_allocation_matrix()
    # A_inv = jp.linalg.inv(A)

    with mujoco.viewer.launch_passive(model, mjdata) as viewer:
        viewer.cam.lookat[:] = 0, 0, 1
        viewer.cam.distance = 15
        viewer.cam.elevation = -30

        parse_sensor_data = create_parse_sensordata_fn(model)
        collision_checker = create_drone_ball_collision_fn(model)

        while viewer.is_running():
            step_start = time.time()

            # --- Control Part ---
            action = np.array([-1, 0, 0, 0])
            s = 1
            if keyboard.Key.up in current_keys:
                action[0] = 1
            if keyboard.Key.left in current_keys:
                action[1] = 1
            if keyboard.Key.right in current_keys:
                action[2] = 1
            if keyboard.Key.down in current_keys:
                action[3] = 1

            # action = jp.array(A_inv @ action)
            # action = action * 2 * s - 1

            # ------
            # action = jp.array(action_list)
            state = jit_step(state, action)

            obs = DroneOnlyObservation.from_array(state.obs)

            # (2) copy the JAX state back into mjdata so the viewer sees it
            # do an np.array(...) on the jax arrays
            mjdata.qpos[:] = np.array(state.pipeline_state.original_pipeline_state.qpos)
            mjdata.qvel[:] = np.array(state.pipeline_state.original_pipeline_state.qvel)
            mjdata.ctrl[:] = np.array(state.pipeline_state.original_pipeline_state.ctrl)
            viewer.user_scn.ngeom = 0
            add_text(mjdata, viewer, f"Reward: {state.reward}", t=0)
            obs_str_1 = f"Position: {str(obs.drone_imu_position)} Velocity: {obs.drone_imu_velocity}"
            obs_str_2 = f"Orientation: {str(obs.drone_imu_orientation_quat_wxyz)} Ang Velocity: {obs.drone_imu_angular_velocity}"
            obs_str_3 = f"Last Action: {str(obs.last_action)}"
            add_text(mjdata, viewer, obs_str_3, t=0.3)
            add_text(mjdata, viewer, obs_str_2, t=0.6)
            add_text(mjdata, viewer, obs_str_1, t=0.9)
            add_text(mjdata, viewer, f"Done: {state.done}", t=1.2)

            if abs(state.reward) > 1:
                print(state.reward)

            if keyboard.Key.enter in current_keys:
                IPython.embed()
                state = jit_reset(rng)

            if jp.isclose(state.done, 1):
                rng = jax.random.PRNGKey(random.randint(0, 10000))
                state = jit_reset(rng)

            # col = collision_checker(state.pipeline_state.original_pipeline_state)
            # a = jp.where(col, 1, 0)
            # print(a)

            mujoco.mj_forward(model, mjdata)
            # etc. for other fields you need to keep in sync

            viewer.sync()
            time_until_next_step = model.opt.timestep - (time.time() - step_start) + 1 / 100
            if time_until_next_step > 0:
                time.sleep(time_until_next_step * 1)
            i += 1
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
    rand_fn = get_domain_rand_fn_v2(exp_config.env.stage_config.env_rand)

    print(exp_config.env.action_delay_discrete, exp_config.env.action_delay)
    v_rand_fn = functools.partial(rand_fn, rng=jax.random.PRNGKey(random.randint(0, 1 << 16)).reshape(1, 2)) if rand_fn is not None else None
    env = envs.training.wrap_progress(
        env,
        episode_length=exp_config.train.episode_length,
        action_repeat=exp_config.train.action_repeat,
        randomization_fn=v_rand_fn,
    )
    visualize(env, exp_config.train.action_repeat)

if __name__ == "__main__":
    main()
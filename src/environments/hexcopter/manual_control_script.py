import random
import time
from typing import Set
import IPython
from environments.hexcopter.observation_models import FullDroneObservation
from environments.hexcopter.state_interfaces import (
    AugmentedEnvState,
    create_parse_sensordata_fn,
)
from models.rpg_drone import get_allocation_matrix

from brax.io import mjcf

# from controllers.controllers import PIDState, pid_update
# from controllers.drone_control_allocation import (
#     ThrustTorque2D,
#     create_control_allocation_2d_fn,
# )
from environments.hexcopter.drone_tracking_3d import Hexcopter3DEnv
from environments.hexcopter.utils import get_env_xml_path

# from models.skydio_x2 import get_x2_3d_xml
from pynput import keyboard

import mujoco
import glfw
import numpy as np
from mujoco import mjx
from OpenGL.GL import glEnable, GL_MULTISAMPLE
import mujoco.viewer
import jax.numpy as jp
import jax


import os

np.set_printoptions(precision=2, suppress=True)

os.environ["JAX_PLATFORM_NAME"] = "cpu"

XML_MODEL_PATH = "src/environments/drone_over_wall/drone_over_wall.xml"
# XML_MODEL_PATH = get_x2_3d_xml().as_posix()
XML_MODEL_PATH = get_env_xml_path().as_posix()
print(XML_MODEL_PATH)

# Global viewport dimensions
viewport_width, viewport_height = 800, 600


def window_resize_callback(window, width, height):
    global viewport_width, viewport_height
    viewport_width, viewport_height = width, height


def main_glfw():
    # Load the MuJoCo model from our XML file
    model = mujoco.MjModel.from_xml_path(XML_MODEL_PATH)
    data = mujoco.MjData(model)

    # Initialize GLFW and request 4x MSAA
    if not glfw.init():
        raise RuntimeError("Could not initialize GLFW")
    glfw.window_hint(glfw.SAMPLES, 4)

    # Create a window for rendering
    width, height = 800, 600
    window = glfw.create_window(width, height, "MuJoCo Playground", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Could not create GLFW window")
    glfw.make_context_current(window)

    # Enable multi-sampling and VSync
    glEnable(GL_MULTISAMPLE)
    glfw.swap_interval(1)

    # Set up window resize callback to update viewport
    glfw.set_window_size_callback(window, window_resize_callback)

    # Create a MuJoCo rendering context
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

    # Create a camera and scene
    cam = mujoco.MjvCamera()
    scn = mujoco.MjvScene(model, maxgeom=1000)

    # Configure camera parameters
    cam.azimuth = 90  # rotation around z-axis
    cam.elevation = -30  # rotation up/down
    cam.distance = 8.0  # distance from the center
    cam.lookat = np.array([0, 0, 0])

    # Set up scene options (tweak these if needed for better lighting/shadows)
    opt = mujoco.MjvOption()

    # Main rendering loop
    while not glfw.window_should_close(window):
        # Step the simulation
        data.ctrl = np.array([100, 0])
        mujoco.mj_step(model, data)

        # Update scene
        mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)

        # Retrieve the actual framebuffer size
        fb_width, fb_height = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, fb_width, fb_height)

        # Render using the framebuffer dimensions
        mujoco.mjr_render(viewport, scn, context)

        # Swap buffers and poll GLFW events
        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()


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


def setup_keyboard_listener():
    current_keys = set()

    def on_press(key):
        try:
            current_keys.add(key)
        except AttributeError:
            current_keys.add(key)
        # print("Key pressed:", key)

    def on_release(key):
        try:
            current_keys.discard(key)
        except AttributeError:
            current_keys.discard(key)
        # print("Key released:", key)
        if key == keyboard.Key.esc:
            return False  # This stops the listener

    # Start the listener in a separate thread
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    return current_keys


if __name__ == "__main__":
    # main_brax()

    current_keys = setup_keyboard_listener()
    # main_passive(current_keys=current_keys)
    main_brax(current_keys)
    # main_regular()
    # listener.join()

from contextlib import contextmanager
from dataclasses import asdict, replace
from functools import wraps
from pathlib import Path
import time
from typing import Any, Literal, Optional, Tuple, Type
import IPython
from environments.hexcopter.action_models import (
    ActionModel,
    ActionModelType,
    DirectPropellerThrust,
    NormThrustBodyRate,
    PropellerOmegaDynamicsFn,
    PropellerThrustDynamicsFn,
    ThrustTorque,
    action_model_factory,
    get_rotor_omega_dynamics_fn,
    get_thrust_dynamics_fn,
    get_thrust_map_and_inv_map_fn,
)
from copy import deepcopy
from environments.hexcopter.state_interfaces import (
    AugmentedPipelineState,
    AugmentedEnvState,
    CurriculumProgressInfo,
    DomainRandomizationState,
    DoneMetrics,
    # get_default_progress,
    DroneState,
    RewardArgs,
    RewardMetrics,
    create_parse_sensordata_fn,
    SensorData,
    set_drone_state,
)
import brax.base as base
from environments.hexcopter.config import DomainRandomizationConfig, EnvConfig, FloatRange
from environments.hexcopter.observation_models import (
    Observation,
    ActorObservationType,
    actor_observation_model_factory,
    critic_observation_model_factory,
)
from environments.hexcopter.env_utils import get_env_xml_path
import jax.scipy.spatial
import jax.scipy.spatial.transform
from models.hexcopter import (
    get_allocation_matrix,
    get_mass,
    get_max_propeller_thurst,
    ComponentNames
)

import mujoco
import jax
from jax import numpy as jp
from jax.scipy.spatial.transform import Rotation

from brax.envs.base import PipelineEnv, State as EnvState
from brax.mjx.base import State as MjxState
from brax.io import mjcf
from brax.base import System
from mujoco import mjx
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
ENV_XML_PATH = get_env_xml_path()


class Hexcopter3DEnv(PipelineEnv):
    """
    Environment for training a 3D drone.
    Uses parameters from a configuration object.
    """

    def __init__(self, config: Optional[EnvConfig] = None, debug=False, **kwargs):
        """
        Initializes the DroneOverWallEnv environment using ExperimentConfig.

        Args:
            config: The ExperimentConfig object holding environment parameters.
            debug: If True, enables debug information.
            **kwargs: Additional keyword arguments passed to the parent constructor.
        """
        if config is None:
            self.config = EnvConfig()
        else:
            self.config = config

        xml_path = ENV_XML_PATH
        # self.mj_model = mujoco.MjModel.from_xml_path(xml_path.as_posix())
        self.model_spec = mujoco.MjSpec.from_file(xml_path.as_posix())
        opt = self.model_spec.option
        # opt.solver = mujoco.mjtSolver.mjSOL_CG  # Conjugate‐gradient solver
        # opt.iterations = 6  # number of solver iterations
        # opt.ls_iterations = 6  # line‐search iterations
        opt.timestep = self.config.opt_timestep
        self.mj_model = self.model_spec.compile()

        # TODO: Apply Mujoco settings (could also be part of config if needed)
        self.mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG  # type: ignore
        self.mj_model.opt.iterations = 6
        self.mj_model.opt.ls_iterations = 6
        self.mj_model.opt.timestep = self.config.opt_timestep

        sys = mjcf.load_model(self.mj_model)

        super().__init__(
            sys=sys,
            backend=self.config.backend,
            n_frames=self.config.physics_steps_per_control_step,
            debug=debug,
            **kwargs,
        )

        # --- Sensor Data Parser ---
        self.parse_sensor_data = create_parse_sensordata_fn(self.mj_model)
        # --- Observation Model ---
        self.actor_observation_model = actor_observation_model_factory(self.config.actor_observation_model)
        self.actor_observation_model.resolve_fields(self.config.action_history_len, self.config.sensor_data_history_len, self.config.action_delay_discrete)
        self.critic_observation_model = critic_observation_model_factory(self.config.critic_observation_model)
        self.critic_observation_model.resolve_fields(self.config.action_history_len, self.config.sensor_data_history_len, self.config.action_delay_discrete)
        
        
        # --- Action Model ---
        self.action_model = action_model_factory(self.config.action_model)
        self.MAX_PROP_CTRL = get_max_propeller_thurst()
        G = 9.8
        self.hover_prop_norm_action = jp.ones(4) * ((get_mass() * G) / 4) / self.MAX_PROP_CTRL

        # --- Rotor dynamics ---
        self.propeller_thrust_dynamics: PropellerThrustDynamicsFn = get_thrust_dynamics_fn(self.config.rotor_tau, self.config.opt_timestep)
        self.propeller_omega_dynamics: PropellerOmegaDynamicsFn = get_rotor_omega_dynamics_fn(self.config.rotor_tau, self.config.opt_timestep)
        self.thrust_map , self.inverse_thrust_map = get_thrust_map_and_inv_map_fn(jp.array(self.config.thrust_map_coefficients))

        self.dt_per_control_step = self.config.opt_timestep * self.config.physics_steps_per_control_step
        self.dt_per_control_step = 1

    def step(self, state: AugmentedEnvState, action: jp.ndarray) -> AugmentedEnvState:
        keys = jax.random.split(state.pipeline_state.rng, 4)

        action_has_changed = (action != state.pipeline_state.last_action).any()

        prev_pipeline_state_aug: AugmentedPipelineState = state.pipeline_state
        # --- Parse Sensor Data ---
        prev_sensor_data = prev_pipeline_state_aug.sensor_data_realtime

        # --- Un-Normalize Action ---
        current_action_from_policy = action
        # --- Action delay ---
        action_to_apply = prev_pipeline_state_aug.get_n_action_history(self.config.action_delay_discrete - 1) if self.config.action_delay_discrete > 0 else action

        # --- Propeller Dynamics ---
        prop_thrust_request = self.action_model.propeller_action_from_normalized_action(action_to_apply, prev_sensor_data)
        prop_thrust_request = jp.clip(prop_thrust_request, 0.0, self.MAX_PROP_CTRL)

        USE_THRUST_MAP_DYNAMICS = True
        if USE_THRUST_MAP_DYNAMICS:
            omega_request = self.inverse_thrust_map(prop_thrust_request )
            real_omega = self.propeller_omega_dynamics(omega_request, prev_pipeline_state_aug)
            prop_thrust = self.thrust_map(real_omega, prev_pipeline_state_aug)
            
        else:
            prop_thrust = self.propeller_thrust_dynamics(prop_thrust_request, prev_pipeline_state_aug)
            real_omega = jp.zeros_like(prop_thrust)  # This is not used if we don't use the thrust map dynamics

        prop_action_norm = prop_thrust / self.MAX_PROP_CTRL

        # --- Step Simulation ---
        # --- Setup system wiht domain randomization ---
        if self.config.stage_config.env_rand.use_domain_randomization:
            new_sys = state.pipeline_state.domain_rand_state.update_sys_with_rand_state(self.sys)
            with self._swap_sys(new_sys):
                stepped_pipeline_state: MjxState = self.pipeline_step(
                    state.pipeline_state.original_pipeline_state,
                    prop_thrust,
                )  # type: ignore
        else:
            stepped_pipeline_state: MjxState = self.pipeline_step(
                state.pipeline_state.original_pipeline_state,
                prop_thrust,
            )  # type: ignore

        # --- Get sensor data
        current_sensor_data = self.parse_sensor_data(stepped_pipeline_state)
        # --- Get next state ---
        next_pipeline_state_aug: AugmentedPipelineState = prev_pipeline_state_aug.replace(
            original_pipeline_state=stepped_pipeline_state,
            prop_action_norm=prop_action_norm,
            prop_omega =real_omega,
            sensor_data_realtime=current_sensor_data,
            # last_action=action,
            rng=keys[0],
        )

        # Update the history only if it's not a repeated action
        next_pipeline_state_aug = jax.lax.cond(action_has_changed, lambda s: s.update_with_new_action(action), lambda s: s, next_pipeline_state_aug)
        next_pipeline_state_aug = jax.lax.cond(
            action_has_changed, lambda s: s.update_with_new_sensor_data(current_sensor_data), lambda s: s, next_pipeline_state_aug
        )

        # --- Get Observation ---
        actor_observation = self.actor_observation_model.get_observation(next_pipeline_state_aug)
        critic_observation =self.critic_observation_model.get_observation(next_pipeline_state_aug)
        obs = {"actor": actor_observation.to_array(), "critic": critic_observation.to_array()}

        # --- Rewards/Termination ---
        reward_args = RewardArgs(
            prev_pipeline_state=prev_pipeline_state_aug,
            next_pipeline_state=next_pipeline_state_aug,
        )
        total_reward, done, next_pipeline_state_aug, reward_metrics, done_metrics = self.reward(reward_args)

        # --- Zero out metrics if state is done ---
        # NOTE: This is done because, while using repeat_action > 1, the metrics are accumulated over those steps. As such, if done happened at the start,
        # the metrics will keep counting even though the episode should be over.
        reward_metrics, done_metrics = jax.lax.cond(
            state.done,
            lambda: (RewardMetrics.get_reset_metrics(), DoneMetrics.get_reset_metrics()),
            lambda: (reward_metrics, done_metrics),
        )


        metrics = {**state.metrics, **reward_metrics.as_dict(), **done_metrics.as_dict()}
        return state.replace(pipeline_state=next_pipeline_state_aug, obs=obs, reward=total_reward, done=done, metrics=metrics)

    @property
    def action_size(self) -> int:
        return self.action_model.get_action_dim()

    def reward(self, reward_args: RewardArgs):
        next_pipeline_state = reward_args.next_pipeline_state
        next_sensor_data = next_pipeline_state.sensor_data_realtime
        #### Reward Section ####
        p = next_pipeline_state.progress.training_progress

        # --- Survive ---
        reward_drone_survive = self.config.stage_config.reward_drone_survive(p)

        # --- Distance to target ---
        reward_drone_dist_to_target = self._reward_dist_to_target(reward_args, p, "exp")

        # --- Distance to target orientation ---
        reward_drone_ang_from_des = self._reward_drone_ang_from_des(reward_args, p, "exp")

        # --- Velocity ---
        reward_drone_vel = self._reward_drone_vel(reward_args, p, "squared")

        # --- Vertical angle ---
        reward_drone_ang_from_v = self._reward_drone_ang_from_v(reward_args, p, "exp")

        # --- Ang Velocity ---
        reward_drone_ang_vel = (
            self.config.stage_config.reward_drone_ang_vel(p) * (jp.linalg.norm(next_sensor_data.drone_imu_angular_velocity) / (2 * jp.pi)) ** 2
        )

        # --- Yaw Rate ---
        reward_drone_yaw_rate = -self.config.stage_config.reward_drone_yaw_rate(p) * (
            jp.exp(-jp.abs(next_sensor_data.drone_imu_angular_velocity[2])) - 1
        )

        # --- Action ---
        reward_drone_action = self._reward_drone_action(reward_args, p, "squared")
        # --- Action change ---
        reward_drone_action_change = self._reward_drone_action_change(reward_args, p, "squared")

        # --- Action Deviation ---
        reward_drone_action_deviation = self._reward_drone_action_deviation(reward_args, p, "exp")

        #### Termination Section ####
        # --- Drone is outside ---
        drone_outside_cond = self._drone_outside_playzone(reward_args)
        reward_drone_outside_playground = jp.where(drone_outside_cond, self.config.stage_config.reward_drone_outside_playground(p), 0.0)

        # --- Sum rewards ---
        cumulative_reward = (
            reward_drone_survive
            + reward_drone_dist_to_target
            + reward_drone_vel
            + reward_drone_ang_from_v
            + reward_drone_ang_vel
            + reward_drone_yaw_rate
            + reward_drone_action_change
            + reward_drone_action
            + reward_drone_outside_playground
            + reward_drone_action_deviation
            + reward_drone_ang_from_des
        )

    
        drone_done_condition = drone_outside_cond
        done = drone_done_condition 
        done = jp.where(done, 1.0, 0.0)

        reward_metrics = RewardMetrics(
            total=cumulative_reward,
            drone_survive=reward_drone_survive,
            drone_dist_to_target=reward_drone_dist_to_target,
            drone_ang_from_v=reward_drone_ang_from_v,
            drone_vel=reward_drone_vel,
            drone_ang_vel=reward_drone_ang_vel,
            drone_yaw_rate=reward_drone_yaw_rate,
            drone_action_change=reward_drone_action_change,
            drone_action=reward_drone_action,
            drone_outside_playground=reward_drone_outside_playground,
            drone_action_deviation=reward_drone_action_deviation,
            drone_ang_from_des=reward_drone_ang_from_des
        )

        done_metrics = DoneMetrics(
            done=done,
            drone_outside_playground=jp.where(drone_outside_cond, 1.0, 0.0),
        )

        return cumulative_reward, done, next_pipeline_state, reward_metrics, done_metrics

    def reset(self, rng: jp.ndarray) -> AugmentedEnvState:
        progress = CurriculumProgressInfo.get_default()
        return self.reset_with_progress(rng, progress)

    def reset_with_progress(self, rng: jp.ndarray, curriculum_info: CurriculumProgressInfo):
        # NOTE: This reset overides the reset of the domain randomization. This means that if this function uses any sys specific data for setting the starting state, all these will only see the default sys parameters.
        # Any sys specific reset parameters could be added to the standard reset function since it is called at the start. However, then some merging of the states has to made since this result normally overides the pipeline parameters variable.
        # --- Curriculum based sampling ---
        rng_drone_state, rng_domain_rand, rng_goal_orientation , rng_state = jax.random.split(rng, 4)
        p = curriculum_info.training_progress
        drone_state = self._random_drone_state_with_progress(rng_drone_state, p)


        qpos0, qvel0 = self.sys.qpos0, jp.zeros(self.sys.nv)
        qpos, qvel = set_drone_state(qpos0, qvel0, drone_state)

        domain_rand_state = self._get_domain_rand_state(rng_domain_rand, self.sys, p)

        # --- Standard boiler plate ---
        prop_action_norm = jp.zeros(self.action_size)
        pipeline_state: MjxState = self.pipeline_init(qpos, qvel)  # type: ignore
        prop_omega = jp.zeros(self.action_size)

        # --- Get sensor data ---
        sensor_data = self.parse_sensor_data(pipeline_state)

        # --- Get state ---
        sensor_data_history = jax.tree.map(lambda x: jp.broadcast_to(x, ((self.config.sensor_data_history_len,) + x.shape)), sensor_data)

        # --- Get Target orientation ---
        target_orientation_xyzw = self._random_quaternion_in_cone(self.config.stage_config.reset_drone_ori_target(p=p), rng_goal_orientation)

        augmented_pipeline_state = AugmentedPipelineState(
            original_pipeline_state=pipeline_state,
            prop_action_norm=prop_action_norm,
            prop_omega=prop_omega,
            progress=curriculum_info,
            rng=rng_state,
            domain_rand_state=domain_rand_state,
            action_history=jp.zeros((self.config.action_buffer_size, self.action_size)),
            sensor_data_realtime=sensor_data,
            sensor_data_history=sensor_data_history,
            target_orientation_xyzw = target_orientation_xyzw
        )

        
        # --- Get observation ---
        actor_observation = self.actor_observation_model.get_observation(augmented_pipeline_state)
        critic_observation =self.critic_observation_model.get_observation(augmented_pipeline_state)
        obs = {"actor": actor_observation.to_array(), "critic": critic_observation.to_array()}

        # --- Other ---
        reward, done = jp.zeros(2)
        reward_metrics = RewardMetrics.get_reset_metrics().as_dict()
        done_metrics = DoneMetrics.get_reset_metrics().as_dict()
        metrics = {**reward_metrics, **done_metrics}

        return EnvState(
            augmented_pipeline_state,  # type: ignore
            obs,
            reward,
            done,
            metrics,
        )

    # --- Reward functions ---
    def _reward_dist_to_target(self, reward_args: RewardArgs, p: jp.ndarray, weight_type: Literal["squared", "exp"] = "squared"):
        dist = self._drone_distance_to_target(reward_args)
        if weight_type == "squared":
            return self.config.stage_config.reward_drone_dist_to_target(p) * dist**2
        elif weight_type == "exp":
            return -(jp.exp(-dist * 5) - 1) * self.config.stage_config.reward_drone_dist_to_target(p)
        else:
            raise ValueError(f"Invalid weight type {weight_type}")

    def _reward_drone_vel(self, reward_args: RewardArgs, p: jp.ndarray, weight_type: Literal["squared", "exp"] = "squared"):
        next_sensor_data = reward_args.next_pipeline_state.sensor_data_realtime
        drone_vel = jp.linalg.norm(next_sensor_data.drone_imu_velocity)
        if weight_type == "squared":
            return self.config.stage_config.reward_drone_vel(p) * drone_vel**2
        elif weight_type == "exp":
            return -self.config.stage_config.reward_drone_vel(p) * (jp.exp(-drone_vel) - 1)
        else:
            raise ValueError(f"Invalid weight type {weight_type}")

    def _reward_drone_ang_from_v(self, reward_args: RewardArgs, p: jp.ndarray, weight_type: Literal["squared", "exp"] = "squared"):
        angle_from_v = self._angle_from_vertical(reward_args)
        if weight_type == "squared":
            return self.config.stage_config.reward_drone_ang_from_v(p) * angle_from_v**2
        elif weight_type == "exp":
            return -self.config.stage_config.reward_drone_ang_from_v(p) * (jp.exp(-angle_from_v / (jp.pi / 2)) - 1)
        else:
            raise ValueError(f"Invalid weight type {weight_type}")

    def _reward_drone_ang_from_des(self, reward_args: RewardArgs, p: jp.ndarray, weight_type: Literal["squared", "exp"] = "squared"):
        angle_from_des = self._ang_diff_from_target(reward_args)
        if weight_type == "squared":
            return self.config.stage_config.reward_drone_ang_from_des(p) * angle_from_des**2
        elif weight_type == "exp":
            return -self.config.stage_config.reward_drone_ang_from_des(p) * (jp.exp(-angle_from_des / (jp.pi/6)) - 1)
        else:
            raise ValueError(f"Invalid weight type {weight_type}")


    def _reward_drone_action(self, reward_args: RewardArgs, p: jp.ndarray, weight_type: Literal["squared", "exp"] = "squared"):
        action = reward_args.next_pipeline_state.prop_action_norm  # - self.hover_prop_norm_action
        if weight_type == "squared":
            return self.config.stage_config.reward_drone_action(p) * jp.sum((action) ** 2) / 4

        elif weight_type == "exp":
            return -self.config.stage_config.reward_drone_action(p) * (jp.sum(jp.exp(-jp.abs(action))) / 4 - 1)
        else:
            raise ValueError(f"Invalid weight type {weight_type}")

    def _reward_drone_action_change(self, reward_args: RewardArgs, p: jp.ndarray, weight_type: Literal["squared", "exp"] = "squared"):
        if issubclass(self.action_model, (NormThrustBodyRate, ThrustTorque)):
            abs_delta_action = jp.abs(reward_args.next_pipeline_state.last_action - reward_args.prev_pipeline_state.last_action)
        else:
            abs_delta_action = jp.abs(reward_args.next_pipeline_state.prop_action_norm - reward_args.prev_pipeline_state.prop_action_norm)

        if weight_type == "squared":
            return self.config.stage_config.reward_drone_action_change(p) * jp.sum((abs_delta_action) ** 2) / 4
        elif weight_type == "exp":
            # return -self.config.stage_config.reward_drone_action_change(p) * jp.exp(-jp.sum(abs_delta_action))
            return -self.config.stage_config.reward_drone_action_change(p) * (jp.sum(jp.exp(-abs_delta_action)) / 4 - 1)
        else:
            raise ValueError(f"Invalid weight type {weight_type}")

    def _reward_drone_action_deviation(self, reward_args: RewardArgs, p: jp.ndarray, weight_type: Literal["squared", "exp"] = "squared"):
        action = reward_args.next_pipeline_state.prop_action_norm
        mean_action = jp.mean(action)
        action_deviation = action - mean_action
        return self.config.stage_config.reward_drone_action_deviation(p) * jp.sum((action_deviation) ** 2)

    # --- State Reset Helper functions ---
    # --- Domain Randomization
    def _get_domain_rand_state(self, rng: jp.ndarray, sys: System, p: jp.ndarray) -> DomainRandomizationState:
        key_density, key_viscosity, key_thrust_map = jax.random.split(rng, 3)
        new_density = sys.density * jax.random.uniform(
            key_density, minval=self.config.stage_config.env_rand.density_mult(p)[0], maxval=self.config.stage_config.env_rand.density_mult(p)[1]
        )
        new_viscosity = sys.viscosity * jax.random.uniform(
            key_viscosity,
            minval=self.config.stage_config.env_rand.viscosity_mult(p)[0],
            maxval=self.config.stage_config.env_rand.viscosity_mult(p)[1],
        )
        
        thrust_map_coefficients =  jp.array(self.config.thrust_map_coefficients) * jax.random.uniform(key_thrust_map, minval=self.config.stage_config.env_rand.thrust_map_coefficients_mult(p)[0], maxval=self.config.stage_config.env_rand.thrust_map_coefficients_mult(p)[1])

        return DomainRandomizationState(density=new_density, viscosity=new_viscosity, thrust_map_coefficients=thrust_map_coefficients)

    # --- Drone
    def _random_drone_state_with_progress(self, rng: jp.ndarray, p) -> DroneState:
        keys = jax.random.split(rng, 4)

        # --- Orientation
        max_angle = self.config.stage_config.reset_drone_orientation_rand(p)

        orientation_xyzw = self._random_quaternion_in_cone(
            max_angle=max_angle,
            rng=keys[0],
        )
        orientation_wxyz = orientation_xyzw[jp.array((3, 0, 1, 2))]

        # --- Ang velocity
        max_ang_vel = self.config.stage_config.reset_drone_ang_vel_rand(p)
        ang_vel = self._random_ang_vel(keys[1], max_ang_vel)

        # --- Position
        pos_rand = self.config.stage_config.reset_drone_pos_rand(p)
        rand_pos = jax.random.uniform(
            key=keys[2],
            shape=pos_rand.shape,
            minval=-pos_rand,
            maxval=pos_rand,
        )
        pos = jp.array(self.config.stage_config.reset_drone_pos_start) + rand_pos

        # --- Lin Vel
        vel_rand = self.config.stage_config.reset_drone_vel_rand(p)
        rand_vel = jax.random.uniform(
            key=keys[3],
            shape=vel_rand.shape,
            minval=-vel_rand,
            maxval=vel_rand,
        )
        vel = rand_vel

        return DroneState(pos, vel, orientation_wxyz, ang_vel)

    # --- Sampling Helper Functions
    def _sample_unit_vector_in_cone(self, rng, max_angle: float):
        key_z, key_φ = jax.random.split(rng, 2)

        cosθ = jp.cos(max_angle)
        z = jax.random.uniform(key_z, minval=cosθ, maxval=1.0)
        φ: jax.Array = jax.random.uniform(key_φ, minval=0.0, maxval=2 * jp.pi)
        r = jp.sqrt(1 - z**2)
        dir = jp.stack([r * jp.cos(φ), r * jp.sin(φ), z])

        return dir

    def _random_quaternion_in_cone(self, max_angle: jp.ndarray, rng: jp.ndarray) -> jp.ndarray:
        key_vec, key_yaw = jax.random.split(rng, 2)

        dir = self._sample_unit_vector_in_cone(key_vec, max_angle)

        # 3) compute tilt quaternion: rotate [0,0,1] → dir
        z_axis = jp.array([0.0, 0.0, 1.0])
        axis = jp.cross(z_axis, dir)
        dot = jp.clip(jp.dot(z_axis, dir), -1.0, 1.0)
        angle = jp.arccos(dot)

        norm = jp.linalg.norm(axis)
        axis = jp.where(
            norm < 1e-6,
            jp.array([1.0, 0.0, 0.0]),  # fallback if dir ≈ ±Z
            axis / norm,
        )

        q_tilt = Rotation.from_rotvec(axis * angle)

        # 4) sample yaw about the *new* body‑Z
        yaw = jax.random.uniform(key_yaw, minval=0.0, maxval=2 * jp.pi)
        z_body = q_tilt.apply(z_axis)  # body Z in world frame
        q_yaw = Rotation.from_rotvec(z_body * yaw)

        # 5) combine: yaw * tilt
        q_cone = q_yaw * q_tilt

        return q_cone.as_quat()  # [x, y, z, w]

    def _random_ang_vel(self, rng: jp.ndarray, max_ang_vel: float):
        axis = jax.random.normal(rng, (3,))
        axis = axis / jp.linalg.norm(axis) * max_ang_vel
        return axis
    
    # --- Helper functions ---
    def _angle_from_vertical(self, reward_args: RewardArgs):
        next_sensor_data = reward_args.next_pipeline_state.sensor_data_realtime
        orientation_xyzw = next_sensor_data.drone_imu_orientation_quat_wxyz[jp.array([1, 2, 3, 0])]
        rot = jax.scipy.spatial.transform.Rotation.from_quat(orientation_xyzw)
        rotated_z = rot.apply(jp.array([0, 0, 1]))
        vertical = jp.array([0, 0, 1])
        cos_theta = jp.clip(jp.dot(rotated_z, vertical), -1.0, 1.0)
        ang_from_vertical = jp.arccos(cos_theta)
        return ang_from_vertical
    
    def _ang_diff_from_target(self, reward_args: RewardArgs):
        next_sensor_data = reward_args.next_pipeline_state.sensor_data_realtime
        orientation_xyzw = next_sensor_data.drone_imu_orientation_quat_wxyz[jp.array([1, 2, 3, 0])]

        return self._ang_diff(orientation_xyzw, reward_args.next_pipeline_state.target_orientation_xyzw)
    
    def _ang_diff(self, orientation_quat_xyzw_1:jp.ndarray, orientation_quat_xyzw_2:jp.ndarray):
        current_rot = jax.scipy.spatial.transform.Rotation.from_quat(orientation_quat_xyzw_1)
        desired_rot = jax.scipy.spatial.transform.Rotation.from_quat(orientation_quat_xyzw_2)
        diff = current_rot * desired_rot.inv()
        return diff.magnitude()

    def _drone_outside_playzone(self, reward_args: RewardArgs):
        next_sensor_data = reward_args.next_pipeline_state.sensor_data_realtime
        p = reward_args.next_pipeline_state.progress.training_progress
        drone_outside_playzone_min = jp.array(self.config.stage_config.zones_drone_playzone_min(p))
        drone_outside_playzone_max = jp.array(self.config.stage_config.zones_drone_playzone_max(p))
        drone_outside_playzone_cond = jp.logical_or(
            (next_sensor_data.drone_imu_position < drone_outside_playzone_min).any(),
            (next_sensor_data.drone_imu_position > drone_outside_playzone_max).any(),
        )
        return drone_outside_playzone_cond

    def _drone_distance_to_target(self, reward_args: RewardArgs):
        next_sensor_data = reward_args.next_pipeline_state.sensor_data_realtime
        # drone_target_position = jp.array(self.config.stage_config.reset_drone_pos_start)  # NOTE: MIGHT ADD DIFFERENT TARGET IN THE FUTURE
        drone_target_position = jp.array(self.config.stage_config.reward_drone_target_position)
        dist_to_drone_target_position = jp.linalg.norm(drone_target_position - next_sensor_data.drone_imu_position)
        return dist_to_drone_target_position

    def _sample_goal_plane_height(self, rng):
        return jax.random.uniform(rng, minval=0.2, maxval=1.5)

    # ---- Helper Tracer Functions ---
    @contextmanager
    def _swap_sys(self, new_sys):
        old_sys = self.sys
        self.sys = new_sys
        try:
            yield
        finally:
            self.sys = old_sys

# --- Domain Randomization Function ---
def _tree_build_diff(a, b, *, atol=1e-8):
    dict_a = asdict(a)
    dict_b = asdict(b)
    diffs_struct = {}

    def get_diff_struct(x, y):
        # nested dict → recurse
        if isinstance(x, dict) and isinstance(y, dict):
            dict_diff_struct = {}
            for k in x:
                leaf_struct = get_diff_struct( x[k], y.get(k))
                if leaf_struct is not None:
                    dict_diff_struct[k] = leaf_struct       
            if dict_diff_struct:
                return dict_diff_struct
            
        # array → allclose
        elif isinstance(x, (jp.ndarray, np.ndarray)) :
            if not jp.allclose(x, y, atol=atol):
                return []
        # other → regular !=
        else:
            if x != y:
                return []
        
        return None

    diffs_struct:dict = get_diff_struct(dict_a, dict_b)
    diffs_struct.pop("mj_model", None) # Manually remove the mj_model key as it is not needed in the diff struct
    return diffs_struct


def _append_to_diff_struct(diff_struct, model):
    """ Find the values in the model that are described in the diff_struct and append them to the diff_struct"""
    for key, value in diff_struct.items():
        if isinstance(value, dict):
            _append_to_diff_struct(value, getattr(model, key))
        else:
            diff_struct[key].append(getattr(model, key))
    return diff_struct


def _get_diff_struct_in_axes(diff_struct):
    """Get the diff_struct in the axes format"""
    axes_diff_struct = {}
    for key, value in diff_struct.items():
        if isinstance(value, dict):
            axes_diff_struct[key] = _get_diff_struct_in_axes(value)
        else:
            axes_diff_struct[key] = 0
    return axes_diff_struct


def _stack_diff_structs_lists(diff_structs):
    """Stack the diff_structs lists into a single list"""
    stacked_diff_struct = {}
    for key, value in diff_structs.items():
        if isinstance(value, dict):
            stacked_diff_struct[key] = _stack_diff_structs_lists(value)
        else:
            stacked_diff_struct[key] = jp.stack(value)
    return stacked_diff_struct


def _recursive_replace(obj, diffs):
    """
    Recursively replaces fields on a flax.struct.dataclass obj.
    """
    for field, val in diffs.items():
        if isinstance(val, dict):
            # first update the nested dataclass
            subobj = getattr(obj, field)
            val = _recursive_replace(subobj, val)
        obj = obj.replace(**{field: val})
    return obj


def get_domain_rand_fn_v2(domain_rand_config: DomainRandomizationConfig):
    """Returns a domain randomization function if enabled in config, otherwise returns None."""
    if domain_rand_config.use_domain_randomization is False:
        return None
    def randomize_env(sys: base.System, rng: jp.ndarray) -> Tuple[base.System, jp.ndarray]:
        """Randomizes the System's physical properties.

        Args:
            sys: The Brax system to randomize
            rng: Random key for batch processing

        Returns:
            Tuple of (randomized system, vmap in_axes specification)
        """
        print("--- Starting domain randomization ---")
        ts = time.time()

        assert rng.ndim  == 2, "rng should be a 2D array with shape (num_envs, rng_size)"
        num_envs = rng.shape[0]
        print("Number of environments: ", num_envs)

        diff_struct = None # Placeholder for the diff struct (to be filled with the differences between the original and randomized system)
        in_axes_diff = None # Placeholder for the in_axes specification (to be filled with the in_axes of the diff struct)

        model = deepcopy(sys.mj_model)
        data = mujoco.MjData(model)
        original_drone_mass = sys.mj_model.body(ComponentNames.BODY).mass 
        original_actuator_1_gear = sys.mj_model.actuator(ComponentNames.ACTUATOR_PROP_1).gear
        original_actuator_2_gear = sys.mj_model.actuator(ComponentNames.ACTUATOR_PROP_2).gear 
        original_actuator_3_gear = sys.mj_model.actuator(ComponentNames.ACTUATOR_PROP_3).gear 
        original_actuator_4_gear = sys.mj_model.actuator(ComponentNames.ACTUATOR_PROP_4).gear 
        
        for i in range(num_envs):
            # Modify the model parameters
            rng_i = rng[i]  # Get the random key for this environment
            rng_motor_gear_variation, rng_drone_mass, rng_contact = jax.random.split(rng_i, 3)
            drone_mass_mult_range = domain_rand_config.drone_mass_mult_range
            motor_gear_variation_range = jp.array(domain_rand_config.gear_variation_range)

            model.body(ComponentNames.BODY).mass = original_drone_mass * jax.random.uniform(
                key=rng_drone_mass,
                minval=drone_mass_mult_range[0],
                maxval=drone_mass_mult_range[1],
            )

            rng_motor_gear_1, rng_motor_gear_2, rng_motor_gear_3, rng_motor_gear_4 = jax.random.split(rng_motor_gear_variation, 4)
            model.actuator(ComponentNames.ACTUATOR_PROP_1).gear = original_actuator_1_gear + jax.random.uniform(
                key=rng_motor_gear_1,
                shape=motor_gear_variation_range.shape,
                minval=-motor_gear_variation_range,
                maxval=motor_gear_variation_range,
            )
            model.actuator(ComponentNames.ACTUATOR_PROP_2).gear = original_actuator_2_gear + jax.random.uniform(
                key=rng_motor_gear_2,
                shape=motor_gear_variation_range.shape,
                minval=-motor_gear_variation_range,
                maxval=motor_gear_variation_range,
            )
            model.actuator(ComponentNames.ACTUATOR_PROP_3).gear = original_actuator_3_gear + jax.random.uniform(
                key=rng_motor_gear_3,
                shape=motor_gear_variation_range.shape,
                minval=-motor_gear_variation_range,
                maxval=motor_gear_variation_range,
            )
            model.actuator(ComponentNames.ACTUATOR_PROP_4).gear = original_actuator_4_gear + jax.random.uniform(
                key=rng_motor_gear_4,
                shape=motor_gear_variation_range.shape,
                minval=-motor_gear_variation_range,
                maxval=motor_gear_variation_range,
            )

            model.geom(ComponentNames.GEOM_PADDLE).solref[1] = jax.random.uniform(
                key=rng_contact,
                minval=domain_rand_config.contact_solref_damping_range[0],
                maxval=domain_rand_config.contact_solref_damping_range[1],
            )
            
            mujoco.mj_setConst(model, data)  # Set constant values in the model
            jax_model = mjcf.load_model(model)

            if diff_struct is None:
                # Initialize the diff_struct with the original model parameters
                diff_struct = _tree_build_diff(sys, jax_model)
                in_axes_diff = _get_diff_struct_in_axes(diff_struct)
            # Append the differences to the diff_struct
            
            _append_to_diff_struct(diff_struct, jax_model)
            

        stacked_diff_struct = _stack_diff_structs_lists(diff_struct)
        randomized_sys = _recursive_replace(sys, stacked_diff_struct)
        print("Ended randomization")

        # Create in_axes specification for vmap
        in_axes_none = jax.tree_util.tree_map(lambda _: None, sys)
        in_axes = _recursive_replace(in_axes_none, in_axes_diff)

        print("Time to create randomized envs: ", time.time() - ts)
        return randomized_sys, in_axes

    return randomize_env

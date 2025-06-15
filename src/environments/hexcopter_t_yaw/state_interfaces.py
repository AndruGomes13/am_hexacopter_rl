from dataclasses import dataclass, fields
from typing import Callable, NamedTuple, Self, Tuple
from environments.hexcopter_t_yaw.env_utils import get_env_xml_path
from environments.hexcopter_t_yaw.env_utils import TrajectoryState
import jax

# from models.skydio_x2 import (
#     X2ComponentNames,
# )
from models.hexcopter_t_yaw import (
    ComponentNames,
)
from models.hexcopter_t_yaw import JointSlices
import mujoco
from jax import numpy as jp

from brax.envs.base import State as EnvState
from brax.mjx.base import State as MjxState

from brax.envs.wrappers.training import CurriculumProgressInfo
from mujoco import mjx
from flax import struct
import brax.base as base
from utils.mujoco_utils import get_qpos_qvel_slice_from_free_joint_body


ENV_XML_PATH = get_env_xml_path().as_posix()
model = mujoco.MjModel.from_xml_path(ENV_XML_PATH)


# --- Pipeline State Dataclasses/interfaces ---
# @dataclass(frozen=True)
class SensorData(NamedTuple):
    # --- Drone ---
    drone_imu_position: jp.ndarray
    drone_imu_orientation_quat_wxyz: jp.ndarray
    drone_imu_velocity: jp.ndarray
    # drone_imu_body_rate: jp.ndarray
    drone_imu_angular_velocity: jp.ndarray

    # --- Arm ---
    end_effector_position: jp.ndarray
    end_effector_orientation_quat_wxyz: jp.ndarray


@dataclass(frozen=True)
class QPos:
    pass


@dataclass(frozen=True)
class QVel:
    pass


def create_parse_sensordata_fn(
    mj_model: mujoco.MjModel,
) -> Callable[[mjx.Data], SensorData]:
    def get_sensor_indices(mj_model: mujoco.MjModel, sensor_name: str) -> jp.ndarray:
        start = mj_model.sensor(sensor_name).adr.item()
        dim = mj_model.sensor(sensor_name).dim.item()
        return jp.arange(start, start + dim)

    # --- Drone ---
    drone_pos_sensor_idx = get_sensor_indices(mj_model, ComponentNames.SENSOR_IMU_POS)
    drone_quat_wxyz_sensor_idx = get_sensor_indices(mj_model, ComponentNames.SENSOR_IMU_QUAT)
    drone_linvel_sensor_idx = get_sensor_indices(mj_model, ComponentNames.SENSOR_IMU_LIN_VEL)
    drone_angvel_sensor_idx = get_sensor_indices(mj_model, ComponentNames.SENSOR_IMU_GYRO)
    # drone_angvel_sensor_idx = get_sensor_indices(mj_model, X2ComponentNames.SENSOR_IMU_ANG_VEL)
    # --- Arm ---
    end_effector_pos_sensor_idx = get_sensor_indices(mj_model, ComponentNames.SENSOR_END_EFFECTOR_POS)
    end_effector_quat_wxyz_sensor_idx = get_sensor_indices(mj_model, ComponentNames.SENSOR_END_EFFECTOR_ORIENTATION)

    def parse_sensordata(data: mjx.Data) -> SensorData:
        # --- Drone ---
        drone_position = data.sensordata[drone_pos_sensor_idx]
        drone_orientation_quat_wxyz = data.sensordata[drone_quat_wxyz_sensor_idx]
        drone_velocity = data.sensordata[drone_linvel_sensor_idx]
        drone_angular_vel = data.sensordata[drone_angvel_sensor_idx]
        # --- Arm ---
        end_effector_position = data.sensordata[end_effector_pos_sensor_idx]
        end_effector_orientation_quat_wxyz = data.sensordata[end_effector_quat_wxyz_sensor_idx]

        return SensorData(
            drone_imu_position=drone_position,
            drone_imu_angular_velocity=drone_angular_vel,
            drone_imu_orientation_quat_wxyz=drone_orientation_quat_wxyz,
            drone_imu_velocity=drone_velocity,
            end_effector_position=end_effector_position,
            end_effector_orientation_quat_wxyz=end_effector_orientation_quat_wxyz,
        )

    return parse_sensordata


# --- State Domain Randomization ---
@struct.dataclass
class DomainRandomizationState:
    density: jp.ndarray
    viscosity: jp.ndarray
    thrust_map_coefficients: jp.ndarray

    def update_sys_with_rand_state(self, sys: base.System) -> base.System:
        new_opt = sys.opt.tree_replace({"density": self.density, "viscosity": self.viscosity})
        randomized_sys = sys.tree_replace({"density": self.density, "viscosity": self.viscosity, "opt": new_opt})

        return randomized_sys


# --- State Augmentation ---
@struct.dataclass
class AugmentedPipelineState:
    # --- Original Sim State ---
    original_pipeline_state: MjxState

    # --- Curriculum ----
    progress: CurriculumProgressInfo

    # --- Sensor Data ---
    # NOTE: Updates every timestep.
    sensor_data_realtime: SensorData
    # NOTE: Updates only at action change (which is not at every timestep if repeat_action > 1)
    sensor_data_history: SensorData

    # --- Action ---
    # NOTE: Updates only at action change (which is not at every timestep if repeat_action > 1)
    action_history: jp.ndarray
    last_ctrl: jp.ndarray

    # --- Objective Trajectory ---
    trajectory: TrajectoryState

    # --- Randomization Related ---
    domain_rand_state: DomainRandomizationState
    rng: jp.ndarray

    @property
    def last_action(self) -> jp.ndarray:
        return self._get_last(self.action_history)

    @property
    def last_sensor_data(self) -> SensorData:
        return self._get_last(self.sensor_data_history)

    def get_n_action_history(self, n: int) -> jp.ndarray:
        return self._get_n(self.action_history, n)

    def update_with_new_action(self, action: jp.ndarray) -> Self:
        assert action.size == self.action_history.shape[1]
        updated = self._update_history(self.action_history, action)
        return self.replace(action_history=updated)

    def update_with_new_sensor_data(self, sensor_data: SensorData) -> Self:
        updated = self._update_history(self.sensor_data_history, sensor_data)
        return self.replace(sensor_data_history=updated)

    # --- Helper Methods ---
    def _update_history(self, history, new_data):
        shifted = jax.tree.map(self._shift, history)
        updated = jax.tree.map(lambda old, nxt: old.at[0].set(nxt), shifted, new_data)
        return updated

    def _get_last(self, history):
        return self._get_n(history, 0)
    
    def _get_n(self, history, n: int):
        """Returns the n elements of the history."""
        return jax.tree.map(lambda x: x[n], history)

    def _shift(self, arr):
        return arr.at[1:].set(arr[:-1])


@struct.dataclass
class AugmentedEnvState(EnvState):
    pipeline_state: AugmentedPipelineState


# @dataclass(frozen=True)
@struct.dataclass
class RewardArgs:
    prev_pipeline_state: AugmentedPipelineState
    next_pipeline_state: AugmentedPipelineState


# --- Drone State related functionality ---
@struct.dataclass
class DroneState:
    position: jp.ndarray
    velocity: jp.ndarray
    yaw: jp.ndarray
    yaw_rate: jp.ndarray


def get_drone_qpos_qvel(drone_state: DroneState) -> Tuple[jp.ndarray, jp.ndarray]:
    """
    Returns (qpos, qvel)
    """
    qpos = jp.concatenate((drone_state.position, drone_state.orientation_wxyz))
    qvel = jp.concatenate((drone_state.velocity, drone_state.angular_velocity))

    return qpos, qvel


def set_drone_state(current_qpos: jp.ndarray, current_qvel: jp.ndarray, drone_state: DroneState) -> Tuple[jp.ndarray, jp.ndarray]:
    """Returns (new_qpos, new_qvel)"""
    pos_ids = jp.concatenate([JointSlices.JOINT_X_QPOS.value, JointSlices.JOINT_Y_QPOS.value, JointSlices.JOINT_Z_QPOS.value])
    vel_ids = jp.concatenate([JointSlices.JOINT_X_QVEL.value, JointSlices.JOINT_Y_QVEL.value, JointSlices.JOINT_Z_QVEL.value])
    new_qpos = current_qpos.at[pos_ids].set(drone_state.position)
    new_qpos = new_qpos.at[JointSlices.JOINT_YAW_QPOS.value].set(drone_state.yaw)

    # Set qvel
    new_qvel = current_qvel.at[vel_ids].set(drone_state.velocity)
    new_qvel = current_qvel.at[JointSlices.JOINT_YAW_QVEL.value].set(drone_state.yaw_rate)

    return new_qpos, new_qvel

def set_arm_state(
    current_qpos: jp.ndarray,
    current_qvel: jp.ndarray,
    arm_pitch_1: float,
    arm_pitch_2: float,
) -> Tuple[jp.ndarray, jp.ndarray]:
    """Returns (new_qpos, new_qvel)"""
    # Set arm pitch 1 and 2
    new_qpos = current_qpos.at[JointSlices.JOINT_ARM_PITCH_1_QPOS.value].set(arm_pitch_1)
    new_qpos = new_qpos.at[JointSlices.JOINT_ARM_PITCH_2_QPOS.value].set(arm_pitch_2)

    # Set arm pitch 1 and 2 velocities to zero
    new_qvel = current_qvel.at[JointSlices.JOINT_ARM_PITCH_1_QVEL.value].set(0.0)
    new_qvel = new_qvel.at[JointSlices.JOINT_ARM_PITCH_2_QVEL.value].set(0.0)

    return new_qpos, new_qvel

# --- Metrics classes ---
@struct.dataclass
class RewardMetrics:
    total: jp.ndarray
    drone_survive: jp.ndarray
    drone_dist_to_target: jp.ndarray
    drone_ang_from_v: jp.ndarray
    drone_ang_from_des: jp.ndarray
    drone_vel: jp.ndarray
    drone_ang_vel: jp.ndarray
    drone_action_change: jp.ndarray
    drone_action: jp.ndarray

    drone_outside_playground: jp.ndarray

    end_effector_dist_to_target: jp.ndarray

    @classmethod
    def get_reset_metrics(cls) -> Self:
        return cls(**{f.name: 0.0 for f in fields(cls)})

    def as_dict(self) -> dict[str, jp.ndarray]:
        """Flatten into a dict of scalar/array metrics, with each key prefixed by 'rewards_'."""
        return {f"rewards/{f.name}": getattr(self, f.name) for f in fields(self)}


@struct.dataclass
class DoneMetrics:
    done: jp.ndarray
    drone_outside_playground: jp.ndarray

    @classmethod
    def get_reset_metrics(cls) -> Self:
        return cls(**{f.name: 0.0 for f in fields(cls)})

    def as_dict(self) -> dict[str, jp.ndarray]:
        """Flatten into a dict of scalar/array metrics, with each key prefixed by 'done_'."""
        return {f"done/{f.name}": getattr(self, f.name) for f in fields(self)}

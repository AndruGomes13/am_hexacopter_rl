from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, fields
from enum import StrEnum
import functools
import math
import sys
from typing import Annotated, ClassVar, Optional, Self, Type, get_args, get_origin
import IPython
from environments.hexcopter_t_yaw.env_utils import sample_trajectory
from environments.hexcopter_t_yaw.state_interfaces import (
    AugmentedPipelineState,
)
from jax import numpy as jp
import jax
from jax.scipy.spatial.transform import Rotation
import numpy as np
import pytest


class ActorObservationType(StrEnum):
    FULL = "full"
    FULL_HIST = "full_hist"

class CriticObservationType(StrEnum):
    FULL = "full"



Shape = tuple[int, ...]  # alias for clarity

# --- Setup general observation "machinery" ---
# NOTE: We followed this method with __FIELDS__, etc vs a pytree approach because we want the serialization to work on both jit and non-jit code.
# The tree would need to know the arrays shapes to de-serialize but it doesn't know unless you tell it explicitely, for example with a sample observation.


def observation_dataclass(cls):
    """Decorator that makes the obs dataclasses and builds the __FIELDS__"""
    derived: list[tuple[str, Shape]] = []
    for f in fields(cls):
        anno = cls.__annotations__[f.name]
        if get_origin(anno) is Annotated:
            base, shape = get_args(anno)
            if base is jp.ndarray:
                derived.append((f.name, shape))

    cls.__FIELD_SPECS__ = tuple(derived)

    return cls

@dataclass(frozen=True)
class Observation(ABC):
    __FIELD_SPECS__: ClassVar[tuple[tuple[str, tuple[int | str, ...]]]] = ()
    __FIELDS__: ClassVar[tuple[tuple[str, Shape]]] = ()
    ACTION_HISTORY_LEN: ClassVar[int] = 1
    ACTION_BUFFER_LEN: ClassVar[int] = 1
    ACTION_SIZE: ClassVar[int] = 1
    SENSOR_DATA_HISTORY_LEN: ClassVar[int] = 1
    __resolved__: ClassVar[bool] = False

    @classmethod
    @abstractmethod
    def get_observation(cls, state: AugmentedPipelineState) -> Self:
        pass

    def to_array(self) -> jp.ndarray:
        """Flattens the observation into a 1D array using the static field specification."""
        if not self.__class__.__resolved__:
            raise RuntimeError("You must call .resolve_fields(...) before using to_array()")

        arrays = []
        for field_name, _ in self.__FIELDS__:
            arrays.append(getattr(self, field_name).ravel())
        return jp.concatenate(arrays, axis=0)

    @classmethod
    def from_array(cls, arr: jp.ndarray):
        """Rebuilds an observation from the 1D array using the static field specification."""
        if not cls.__resolved__:
            raise RuntimeError("You must call .resolve_fields(...) before using from_array()")

        out = {}
        idx = 0
        for field_name, shape in cls.__FIELDS__:
            size = math.prod(shape)  # Compute the total size for this field.
            out[field_name] = arr[idx : idx + size].reshape(shape)
            idx += size

        assert idx == arr.size, f"got {arr.size} but expected {idx}"
        return cls(**out)

    @classmethod
    def resolve_fields(cls, action_size: Optional[int] = None, action_history_len: Optional[int] = None, sensor_data_history_len: Optional[int] = None, action_buffer_len :Optional[int] = None):
        if action_history_len is not None:
            cls.ACTION_HISTORY_LEN = action_history_len
        if sensor_data_history_len is not None:
            cls.SENSOR_DATA_HISTORY_LEN = sensor_data_history_len
        if action_buffer_len is not None:
            cls.ACTION_BUFFER_LEN = action_buffer_len
        if action_size is not None:
            cls.ACTION_SIZE = action_size

        resolved = []
        for name, spec in cls.__FIELD_SPECS__:
            # turn any string dims into ints
            shape = tuple(getattr(cls, dim) if isinstance(dim, str) else dim for dim in spec)
            resolved.append((name, shape))
        cls.__FIELDS__ = tuple(resolved)
        cls.__resolved__ = True

        # wrap get_observation to auto‐validate
        orig = cls.get_observation.__func__  # unwrap the classmethod

        @functools.wraps(orig)
        def _wrapped(cls, state):
            obs = orig(cls, state)
            cls._validate(obs)
            return obs

        cls.get_observation = classmethod(_wrapped)

    @classmethod
    def _validate(cls, obs: Self):
        if not cls.__resolved__:
            raise RuntimeError("Must call resolve_fields(...) before using get_observation")
        for name, expected in cls.__FIELDS__:
            actual = getattr(obs, name).shape
            assert actual == expected, f"{cls.__name__}.{name} shape mismatch: got {actual}, expected {expected}"

    @classmethod
    def generate_random(cls, key: jp.ndarray) -> Self:
        if not cls.__resolved__:
            raise RuntimeError("Must call resolve_fields(...) before using get_observation")

        keys = jax.random.split(key, len(cls.__FIELDS__))

        random_attr = {}
        for (field_name, shape), k in zip(cls.__FIELDS__, keys):
            random_attr[field_name] = jax.random.normal(k, shape)

        return cls(**random_attr)

    def __eq__(self, other: Self) -> bool:
        for field_name, shape in self.__FIELDS__:
            self_value: jp.ndarray = getattr(self, field_name)
            other_value: jp.ndarray = getattr(other, field_name)

            # Check shape
            if self_value.shape != other_value.shape:
                return False

            # Check type
            if self_value.dtype != other_value.dtype:
                return False

            # Check if equal
            if not jp.allclose(self_value, other_value):
                return False

        return True


# --- Actor Observation Models ---
@observation_dataclass
@dataclass(frozen=True, eq=False)
class FullDroneObservation(Observation):
    # --- Drone ---
    drone_imu_position: Annotated[jp.ndarray, (3,)]
    drone_imu_orientation_quat_wxyz: Annotated[jp.ndarray, (4,)]
    drone_imu_velocity: Annotated[jp.ndarray, (3,)]
    # drone_imu_body_rate: jp.ndarray
    drone_imu_angular_velocity: Annotated[jp.ndarray, (3,)]


    # --- Last action ---
    # last_action: Annotated[jp.ndarray, (6,)]  # TODO: This expects action of size 4. Make this dynamic.

    @classmethod
    def get_observation(cls, state: AugmentedPipelineState) -> Self:
        sensor_data = state.sensor_data_realtime

        return cls(
            drone_imu_position=sensor_data.drone_imu_position,
            drone_imu_orientation_quat_wxyz=sensor_data.drone_imu_orientation_quat_wxyz,
            drone_imu_velocity=sensor_data.drone_imu_velocity,
            drone_imu_angular_velocity=sensor_data.drone_imu_angular_velocity,
            # last_action=state.last_action,
        )


@observation_dataclass
@dataclass(frozen=True, eq=False)
class ActorFullObservationWithHistory(Observation):
    # --- Drone ---
    drone_imu_position: Annotated[jp.ndarray, (3,)]
    drone_imu_orientation_quat_wxyz: Annotated[jp.ndarray, (4,)]
    # drone_imu_orientation_matrix: Annotated[jp.ndarray, (3, 3)]
    drone_imu_velocity: Annotated[jp.ndarray, (3,)]
    # drone_imu_body_rate: jp.ndarray
    drone_imu_angular_velocity: Annotated[jp.ndarray, (3,)]

    # --- Target Orientation ---

    # --- Last action ---
    # last_action: Annotated[jp.ndarray, (4,)]  # TODO: This expects action of size 4. Make this dynamic.
    # last_prop_action_norm: Annotated[jp.ndarray, (4,)]
    action_history: Annotated[jp.ndarray, ("ACTION_HISTORY_LEN", "ACTION_SIZE")]

    trajectory_forward: Annotated[jp.ndarray, (5,3)]

    @classmethod
    def get_observation(cls, state: AugmentedPipelineState) -> Self:
        sensor_data = state.sensor_data_realtime
        # Orientation 3x3
        # orientation_matrix = Rotation.from_quat(sensor_data.drone_imu_orientation_quat_wxyz).as_matrix()
        trajectory_forward = sample_trajectory(state.trajectory, state.original_pipeline_state.time, 0.2, 5)[0] - sensor_data.drone_imu_position
        return cls(
            drone_imu_position=sensor_data.drone_imu_position,
            drone_imu_orientation_quat_wxyz=sensor_data.drone_imu_orientation_quat_wxyz,
            # drone_imu_orientation_matrix=orientation_matrix,
            drone_imu_velocity=sensor_data.drone_imu_velocity,
            drone_imu_angular_velocity=sensor_data.drone_imu_angular_velocity,
            # last_prop_action_norm=state.prop_action_norm,
            action_history=state.action_history[:cls.ACTION_HISTORY_LEN],
            trajectory_forward = trajectory_forward
        )


def actor_observation_model_factory(observation_model_type: ActorObservationType) -> type[Observation]:
    if observation_model_type == ActorObservationType.FULL:
        return FullDroneObservation
    elif observation_model_type == ActorObservationType.FULL_HIST:
        return ActorFullObservationWithHistory
    else:
        raise ValueError(f"Not valid observation model: {observation_model_type}")

# --- Critic Observation Models ---
@observation_dataclass
@dataclass(frozen=True, eq=False)
class CriticFullObservationWithHistory(Observation):
    # --- Drone ---
    drone_imu_position: Annotated[jp.ndarray, (3,)]
    drone_imu_orientation_quat_wxyz: Annotated[jp.ndarray, (4,)]
    # drone_imu_orientation_matrix: Annotated[jp.ndarray, (3, 3)]
    drone_imu_velocity: Annotated[jp.ndarray, (3,)]
    # drone_imu_body_rate: jp.ndarray
    drone_imu_angular_velocity: Annotated[jp.ndarray, (3,)]

    end_effector_position: Annotated[jp.ndarray, (3,)]  # Position of the end effector in world coordinates
    trajectory_forward: Annotated[jp.ndarray, (5,3)]

    # --- Last action ---
    # last_action: Annotated[jp.ndarray, (4,)]  # TODO: This expects action of size 4. Make this dynamic.
    action_history: Annotated[jp.ndarray, ("ACTION_BUFFER_LEN", "ACTION_SIZE")]

    @classmethod
    def get_observation(cls, state: AugmentedPipelineState) -> Self:
        sensor_data = state.sensor_data_realtime
       
        # Orientation 3x3
        # orientation_matrix = Rotation.from_quat(sensor_data.drone_imu_orientation_quat_wxyz).as_matrix()

        trajectory_forward = sample_trajectory(state.trajectory, state.original_pipeline_state.time, 0.2, 5)[0] - sensor_data.drone_imu_position
        
        return cls(
            drone_imu_position=sensor_data.drone_imu_position,
            drone_imu_orientation_quat_wxyz=sensor_data.drone_imu_orientation_quat_wxyz,
            # drone_imu_orientation_matrix=orientation_matrix,
            drone_imu_velocity=sensor_data.drone_imu_velocity,
            drone_imu_angular_velocity=sensor_data.drone_imu_angular_velocity,
            action_history=state.action_history,
            end_effector_position=sensor_data.end_effector_position,
            trajectory_forward = trajectory_forward
        )

def critic_observation_model_factory(observation_model_type: CriticObservationType) -> type[Observation]:
    if observation_model_type == CriticObservationType.FULL:
        return CriticFullObservationWithHistory
    else:
        raise ValueError(f"Not valid observation model: {observation_model_type}")


# --- Serialization Tools ---
@contextmanager
def reset_fields(obs_cls):
    old_state = (obs_cls.__resolved__, obs_cls.__FIELDS__)
    try:
        yield
    finally:
        obs_cls.__resolved__, obs_cls.__FIELDS__ = old_state


@pytest.mark.parametrize("obs_cls", [FullDroneObservation, ActorFullObservationWithHistory,CriticFullObservationWithHistory])
@pytest.mark.parametrize(
    "action_hist,sensor_hist",
    [
        (1, 1),
        (3, 2),
        (5, 7),  # add edge cases: 1, large, etc.
    ],
)
def test_roundtrip(obs_cls, action_hist, sensor_hist):
    with reset_fields(obs_cls):
        key = jax.random.PRNGKey(0)
        obs_cls.resolve_fields(action_hist, sensor_hist)
        original = obs_cls.generate_random(key)

        flat = original.to_array()
        restored = obs_cls.from_array(flat)

        assert restored == original


if __name__ == "__main__":
    # Runs Serialization Tests
    sys.exit(pytest.main([__file__, "-q"]))

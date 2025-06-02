"""
Sensor functions.
"""
from __future__ import annotations
import jax as jax
from jax import numpy as jp
import mujoco as mujoco
from mujoco.mjx._src import math
from mujoco.mjx._src import ray
from mujoco.mjx._src import smooth
from mujoco.mjx._src import support
from mujoco.mjx._src.types import Data
from mujoco.mjx._src.types import DisableBit
from mujoco.mjx._src.types import Model
from mujoco.mjx._src.types import ObjType
from mujoco.mjx._src.types import SensorType
import numpy as np
__all__ = ['Data', 'DisableBit', 'Model', 'ObjType', 'SensorType', 'apply_cutoff', 'jax', 'jp', 'math', 'mujoco', 'np', 'ray', 'sensor_acc', 'sensor_pos', 'sensor_vel', 'smooth', 'support']
def apply_cutoff(sensor: jax.Array, cutoff: jax.Array, data_type: int) -> jax.Array:
    """
    Clip sensor to cutoff value.
    """
def sensor_acc(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data) -> mujoco.mjx._src.types.Data:
    """
    Compute acceleration/force-dependent sensors values.
    """
def sensor_pos(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data) -> mujoco.mjx._src.types.Data:
    """
    Compute position-dependent sensors values.
    """
def sensor_vel(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data) -> mujoco.mjx._src.types.Data:
    """
    Compute velocity-dependent sensors values.
    """

"""
Passive forces.
"""
from __future__ import annotations
import jax as jax
from jax import numpy as jp
from mujoco.mjx._src import math
from mujoco.mjx._src import scan
from mujoco.mjx._src import support
import mujoco.mjx._src.types
from mujoco.mjx._src.types import Data
from mujoco.mjx._src.types import DisableBit
from mujoco.mjx._src.types import JointType
from mujoco.mjx._src.types import Model
__all__ = ['Data', 'DisableBit', 'JointType', 'Model', 'jax', 'jp', 'math', 'passive', 'scan', 'support']
def _fluid(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data) -> jax.Array:
    """
    Applies body-level viscosity, lift and drag.
    """
def _gravcomp(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data) -> jax.Array:
    """
    Applies body-level gravity compensation.
    """
def _inertia_box_fluid_model(m: mujoco.mjx._src.types.Model, inertia: jax.Array, mass: jax.Array, root_com: jax.Array, xipos: jax.Array, ximat: jax.Array, cvel: jax.Array) -> typing.Tuple[jax.Array, jax.Array]:
    """
    Fluid forces based on inertia-box approximation.
    """
def _spring_damper(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data) -> jax.Array:
    """
    Applies joint level spring and damping forces.
    """
def passive(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data) -> mujoco.mjx._src.types.Data:
    """
    Adds all passive forces.
    """

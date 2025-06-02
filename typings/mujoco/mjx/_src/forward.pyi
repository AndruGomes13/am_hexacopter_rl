"""
Forward step functions.
"""
from __future__ import annotations
import functools as functools
import jax as jax
from jax import numpy as jp
import mujoco as mujoco
from mujoco.mjx._src import collision_driver
from mujoco.mjx._src import constraint
from mujoco.mjx._src import math
from mujoco.mjx._src import passive
from mujoco.mjx._src import scan
from mujoco.mjx._src import sensor
from mujoco.mjx._src import smooth
from mujoco.mjx._src import solver
from mujoco.mjx._src import support
from mujoco.mjx._src.types import BiasType
from mujoco.mjx._src.types import Data
from mujoco.mjx._src.types import DisableBit
from mujoco.mjx._src.types import DynType
from mujoco.mjx._src.types import GainType
from mujoco.mjx._src.types import IntegratorType
from mujoco.mjx._src.types import JointType
from mujoco.mjx._src.types import Model
import numpy as np
import numpy
__all__ = ['BiasType', 'Data', 'DisableBit', 'DynType', 'GainType', 'IntegratorType', 'JointType', 'Model', 'collision_driver', 'constraint', 'euler', 'forward', 'functools', 'fwd_acceleration', 'fwd_actuation', 'fwd_position', 'fwd_velocity', 'implicit', 'jax', 'jp', 'math', 'mujoco', 'named_scope', 'np', 'passive', 'rungekutta4', 'scan', 'sensor', 'smooth', 'solver', 'step', 'support']
def _advance(*args, **kwargs) -> mujoco.mjx._src.types.Data:
    """
    Advance state and time given activation derivatives and acceleration.
    """
def _integrate_pos(*args, **kwargs) -> jax.Array:
    """
    Integrate position given velocity.
    """
def _next_activation(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data, act_dot: jax.Array) -> jax.Array:
    """
    Returns the next act given the current act_dot, after clamping.
    """
def euler(*args, **kwargs) -> mujoco.mjx._src.types.Data:
    """
    Euler integrator, semi-implicit in velocity.
    """
def forward(*args, **kwargs) -> mujoco.mjx._src.types.Data:
    """
    Forward dynamics.
    """
def fwd_acceleration(*args, **kwargs) -> mujoco.mjx._src.types.Data:
    """
    Add up all non-constraint forces, compute qacc_smooth.
    """
def fwd_actuation(*args, **kwargs) -> mujoco.mjx._src.types.Data:
    """
    Actuation-dependent computations.
    """
def fwd_position(*args, **kwargs) -> mujoco.mjx._src.types.Data:
    """
    Position-dependent computations.
    """
def fwd_velocity(*args, **kwargs) -> mujoco.mjx._src.types.Data:
    """
    Velocity-dependent computations.
    """
def implicit(*args, **kwargs) -> mujoco.mjx._src.types.Data:
    """
    Integrates fully implicit in velocity.
    """
def named_scope(fn, name: str = ''):
    ...
def rungekutta4(*args, **kwargs) -> mujoco.mjx._src.types.Data:
    """
    Runge-Kutta explicit order 4 integrator.
    """
def step(*args, **kwargs) -> mujoco.mjx._src.types.Data:
    """
    Advance simulation.
    """
_RK4_A: numpy.ndarray  # value = array([[0.5, 0. , 0. ],...
_RK4_B: numpy.ndarray  # value = array([0.16666667, 0.33333333, 0.33333333, 0.16666667])

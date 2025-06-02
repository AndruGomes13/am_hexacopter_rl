"""
Core smooth dynamics functions.
"""
from __future__ import annotations
import jax as jax
from jax import numpy as jp
import mujoco as mujoco
from mujoco.mjx._src import math
from mujoco.mjx._src import scan
from mujoco.mjx._src import support
from mujoco.mjx._src.types import CamLightType
from mujoco.mjx._src.types import Data
from mujoco.mjx._src.types import DisableBit
from mujoco.mjx._src.types import EqType
from mujoco.mjx._src.types import JointType
from mujoco.mjx._src.types import Model
from mujoco.mjx._src.types import TrnType
from mujoco.mjx._src.types import WrapType
import numpy as np
import numpy
__all__ = ['CamLightType', 'Data', 'DisableBit', 'EqType', 'JointType', 'Model', 'TrnType', 'WrapType', 'camlight', 'com_pos', 'com_vel', 'crb', 'factor_m', 'jax', 'jp', 'kinematics', 'math', 'mujoco', 'np', 'rne', 'rne_postconstraint', 'scan', 'solve_m', 'subtree_vel', 'support', 'tendon', 'transmission']
def _site_dof_mask(m: mujoco.mjx._src.types.Model) -> numpy.ndarray:
    """
    Creates a dof mask for site transmissions.
    """
def camlight(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data) -> mujoco.mjx._src.types.Data:
    """
    Computes camera and light positions and orientations.
    """
def com_pos(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data) -> mujoco.mjx._src.types.Data:
    """
    Maps inertias and motion dofs to global frame centered at subtree-CoM.
    """
def com_vel(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data) -> mujoco.mjx._src.types.Data:
    """
    Computes cvel, cdof_dot.
    """
def crb(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data) -> mujoco.mjx._src.types.Data:
    """
    Runs composite rigid body inertia algorithm.
    """
def factor_m(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data) -> mujoco.mjx._src.types.Data:
    """
    Gets factorizaton of inertia-like matrix M, assumed spd.
    """
def kinematics(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data) -> mujoco.mjx._src.types.Data:
    """
    Converts position/velocity from generalized coordinates to maximal.
    """
def rne(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data) -> mujoco.mjx._src.types.Data:
    """
    Computes inverse dynamics using the recursive Newton-Euler algorithm.
    """
def rne_postconstraint(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data) -> mujoco.mjx._src.types.Data:
    """
    RNE with complete data: compute cacc, cfrc_ext, cfrc_int.
    """
def solve_m(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data, x: jax.Array) -> jax.Array:
    """
    Computes sparse backsubstitution:  x = inv(L'*D*L)*y .
    """
def subtree_vel(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data) -> mujoco.mjx._src.types.Data:
    """
    Subtree linear velocity and angular momentum.
    """
def tendon(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data) -> mujoco.mjx._src.types.Data:
    """
    Computes tendon lengths and moments.
    """
def transmission(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data) -> mujoco.mjx._src.types.Data:
    """
    Computes actuator/transmission lengths and moments.
    """

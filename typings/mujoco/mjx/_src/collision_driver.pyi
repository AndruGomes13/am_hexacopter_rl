"""
Runs collision checking for all geoms in a Model.

To do this, collision_driver builds a collision function table, and then runs
the collision functions serially on the parameters in the table.

For example, if a Model has three geoms:

geom   |   type
---------------
1      | sphere
2      | capsule
3      | sphere

collision_driver organizes it into these functions and runs them:

function       | geom pair
--------------------------
sphere_sphere  | (1, 3)
sphere_capsule | (1, 2), (2, 3)


Besides collision function, function tables are keyed on mesh id and condim,
in order to guarantee static shapes for contacts and jacobians.
"""
from __future__ import annotations
import itertools as itertools
import jax as jax
from jax import numpy as jp
import mujoco as mujoco
from mujoco.mjx._src.collision_types import FunctionKey
from mujoco.mjx._src import support
from mujoco.mjx._src.types import Contact
from mujoco.mjx._src.types import Data
from mujoco.mjx._src.types import DisableBit
from mujoco.mjx._src.types import GeomType
from mujoco.mjx._src.types import Model
import numpy as np
import numpy
import os as os
__all__ = ['Contact', 'Data', 'DisableBit', 'FunctionKey', 'GeomType', 'Model', 'collision', 'geom_pairs', 'has_collision_fn', 'itertools', 'jax', 'jp', 'make_condim', 'mujoco', 'np', 'os', 'support']
def _contact_groups(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data) -> typing.Dict[mujoco.mjx._src.collision_types.FunctionKey, mujoco.mjx._src.types.Contact]:
    """
    Returns contact groups to check for collisions.
    
      Contacts are grouped the same way as _geom_groups.  Only one contact is
      emitted per geom pair, even if the collision function emits multiple contacts.
    
      Args:
        m: MJX model
        d: MJX data
    
      Returns:
        a dict where the key is the grouping and value is a Contact
      
    """
def _geom_groups(m: typing.Union[mujoco.mjx._src.types.Model, mujoco._structs.MjModel]) -> typing.Dict[mujoco.mjx._src.collision_types.FunctionKey, typing.List[typing.Tuple[int, int, int]]]:
    """
    Returns geom pairs to check for collision grouped by collision function.
    
      The grouping consists of:
        - The collision function to run, which is determined by geom types
        - For mesh geoms, convex functions are run for each distinct mesh in the
          model, because the convex functions expect static mesh size. If a sphere
          collides with a cube and a tetrahedron, sphere_convex is called twice.
        - The condim of the collision. This ensures that the size of the resulting
          constraint jacobian is determined at compile time.
    
      Args:
        m: a MuJoCo or MJX model
    
      Returns:
        a dict with grouping key and values geom1, geom2, pair index
      
    """
def _numeric(m: typing.Union[mujoco.mjx._src.types.Model, mujoco._structs.MjModel], name: str) -> int:
    ...
def collision(m: mujoco.mjx._src.types.Model, d: mujoco.mjx._src.types.Data) -> mujoco.mjx._src.types.Data:
    """
    Collides geometries.
    """
def geom_pairs(m: typing.Union[mujoco.mjx._src.types.Model, mujoco._structs.MjModel]) -> typing.Iterator[typing.Tuple[int, int, int]]:
    """
    Yields geom pairs to check for collisions.
    
      Args:
        m: a MuJoCo or MJX model
    
      Yields:
        geom1, geom2, and pair index if defined in <pair> (else -1)
      
    """
def has_collision_fn(t1: mujoco.mjx._src.types.GeomType, t2: mujoco.mjx._src.types.GeomType) -> bool:
    """
    Returns True if a collision function exists for a pair of geom types.
    """
def make_condim(m: typing.Union[mujoco.mjx._src.types.Model, mujoco._structs.MjModel]) -> numpy.ndarray:
    """
    Returns the dims of the contacts for a Model.
    """
_COLLISION_FUNC: dict  # value = {(<GeomType.PLANE: 0>, <GeomType.SPHERE: 2>): <function collider.<locals>.wrapper.<locals>.collide at 0x113d9b920>, (<GeomType.PLANE: 0>, <GeomType.CAPSULE: 3>): <function collider.<locals>.wrapper.<locals>.collide at 0x113d9ba60>, (<GeomType.PLANE: 0>, <GeomType.BOX: 6>): <function collider.<locals>.wrapper.<locals>.collide at 0x113d9a2a0>, (<GeomType.PLANE: 0>, <GeomType.ELLIPSOID: 4>): <function collider.<locals>.wrapper.<locals>.collide at 0x113d9bba0>, (<GeomType.PLANE: 0>, <GeomType.CYLINDER: 5>): <function collider.<locals>.wrapper.<locals>.collide at 0x113d9bce0>, (<GeomType.PLANE: 0>, <GeomType.MESH: 7>): <function collider.<locals>.wrapper.<locals>.collide at 0x113d9a2a0>, (<GeomType.HFIELD: 1>, <GeomType.SPHERE: 2>): <function collider.<locals>.wrapper.<locals>.collide at 0x113d9b240>, (<GeomType.HFIELD: 1>, <GeomType.CAPSULE: 3>): <function collider.<locals>.wrapper.<locals>.collide at 0x113d9b380>, (<GeomType.HFIELD: 1>, <GeomType.BOX: 6>): <function collider.<locals>.wrapper.<locals>.collide at 0x113d9b4c0>, (<GeomType.HFIELD: 1>, <GeomType.MESH: 7>): <function collider.<locals>.wrapper.<locals>.collide at 0x113d9b4c0>, (<GeomType.SPHERE: 2>, <GeomType.SPHERE: 2>): <function collider.<locals>.wrapper.<locals>.collide at 0x113d9bec0>, (<GeomType.SPHERE: 2>, <GeomType.CAPSULE: 3>): <function collider.<locals>.wrapper.<locals>.collide at 0x113da0040>, (<GeomType.SPHERE: 2>, <GeomType.CYLINDER: 5>): <function collider.<locals>.wrapper.<locals>.collide at 0x113da1300>, (<GeomType.SPHERE: 2>, <GeomType.ELLIPSOID: 4>): <function collider.<locals>.wrapper.<locals>.collide at 0x113da11c0>, (<GeomType.SPHERE: 2>, <GeomType.BOX: 6>): <function collider.<locals>.wrapper.<locals>.collide at 0x113d9a480>, (<GeomType.SPHERE: 2>, <GeomType.MESH: 7>): <function collider.<locals>.wrapper.<locals>.collide at 0x113d9a480>, (<GeomType.CAPSULE: 3>, <GeomType.CAPSULE: 3>): <function collider.<locals>.wrapper.<locals>.collide at 0x113da0180>, (<GeomType.CAPSULE: 3>, <GeomType.BOX: 6>): <function collider.<locals>.wrapper.<locals>.collide at 0x113d9a660>, (<GeomType.CAPSULE: 3>, <GeomType.ELLIPSOID: 4>): <function collider.<locals>.wrapper.<locals>.collide at 0x113da1440>, (<GeomType.CAPSULE: 3>, <GeomType.CYLINDER: 5>): <function collider.<locals>.wrapper.<locals>.collide at 0x113da1580>, (<GeomType.CAPSULE: 3>, <GeomType.MESH: 7>): <function collider.<locals>.wrapper.<locals>.collide at 0x113d9a660>, (<GeomType.ELLIPSOID: 4>, <GeomType.ELLIPSOID: 4>): <function collider.<locals>.wrapper.<locals>.collide at 0x113da16c0>, (<GeomType.ELLIPSOID: 4>, <GeomType.CYLINDER: 5>): <function collider.<locals>.wrapper.<locals>.collide at 0x113da1800>, (<GeomType.CYLINDER: 5>, <GeomType.CYLINDER: 5>): <function collider.<locals>.wrapper.<locals>.collide at 0x113da1940>, (<GeomType.BOX: 6>, <GeomType.BOX: 6>): <function collider.<locals>.wrapper.<locals>.collide at 0x113d9af20>, (<GeomType.BOX: 6>, <GeomType.MESH: 7>): <function collider.<locals>.wrapper.<locals>.collide at 0x113d9b060>, (<GeomType.MESH: 7>, <GeomType.MESH: 7>): <function collider.<locals>.wrapper.<locals>.collide at 0x113d9b060>}
_GEOM_NO_BROADPHASE: set  # value = {<GeomType.PLANE: 0>, <GeomType.HFIELD: 1>}

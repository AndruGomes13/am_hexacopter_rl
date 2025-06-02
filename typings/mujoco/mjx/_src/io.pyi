"""
Functions to initialize, load, or save data.
"""
from __future__ import annotations
import copy as copy
import jax as jax
from jax import numpy as jp
import mujoco as mujoco
from mujoco.mjx._src import collision_driver
from mujoco.mjx._src import constraint
from mujoco.mjx._src import mesh
from mujoco.mjx._src import support
from mujoco.mjx._src import types
import numpy as np
import numpy
import scipy as scipy
__all__ = ['collision_driver', 'constraint', 'copy', 'get_data', 'get_data_into', 'jax', 'jp', 'make_data', 'mesh', 'mujoco', 'np', 'put_data', 'put_model', 'scipy', 'support', 'types']
def _get_contact(c: mujoco._structs._MjContactList, cx: mujoco.mjx._src.types.Contact):
    """
    Converts mjx.Contact to mujoco._structs._MjContactList.
    """
def _make_contact(c: mujoco._structs._MjContactList, dim: numpy.ndarray, efc_address: numpy.ndarray) -> typing.Tuple[mujoco.mjx._src.types.Contact, numpy.ndarray]:
    """
    Converts mujoco.structs._MjContactList into mjx.Contact.
    """
def _make_option(o: mujoco._structs.MjOption, _full_compat: bool = False) -> mujoco.mjx._src.types.Option:
    """
    Returns mjx.Option given mujoco.MjOption.
    """
def _make_statistic(s: mujoco._structs.MjStatistic) -> mujoco.mjx._src.types.Statistic:
    """
    Puts mujoco.MjStatistic onto a device, resulting in mjx.Statistic.
    """
def _strip_weak_type(tree):
    ...
def get_data(m: mujoco._structs.MjModel, d: mujoco.mjx._src.types.Data) -> typing.Union[mujoco._structs.MjData, typing.List[mujoco._structs.MjData]]:
    """
    Gets mjx.Data from a device, resulting in mujoco.MjData or List[MjData].
    """
def get_data_into(result: typing.Union[mujoco._structs.MjData, typing.List[mujoco._structs.MjData]], m: mujoco._structs.MjModel, d: mujoco.mjx._src.types.Data):
    """
    Gets mjx.Data from a device into an existing mujoco.MjData or list.
    """
def make_data(m: typing.Union[mujoco.mjx._src.types.Model, mujoco._structs.MjModel], device = None, _full_compat: bool = False) -> mujoco.mjx._src.types.Data:
    """
    Allocate and initialize Data.
    
      Args:
        m: the model to use
        device: which device to use - if unspecified picks the default device
        _full_compat: create all MjData fields on device irrespective of MJX support
          This is an experimental feature.  Avoid using it for now. If using this
          flag, also use _full_compat for put_model.
    
      Returns:
        an initialized mjx.Data placed on device
      
    """
def put_data(m: mujoco._structs.MjModel, d: mujoco._structs.MjData, device = None, _full_compat: bool = False) -> mujoco.mjx._src.types.Data:
    """
    Puts mujoco.MjData onto a device, resulting in mjx.Data.
    
      Args:
        m: the model to use
        d: the data to put on device
        device: which device to use - if unspecified picks the default device
        _full_compat: put all MjModel fields onto device irrespective of MJX support
          This is an experimental feature.  Avoid using it for now. If using this
          flag, also use _full_compat for put_model.
    
      Returns:
        an mjx.Data placed on device
      
    """
def put_model(m: mujoco._structs.MjModel, device = None, _full_compat: bool = False) -> mujoco.mjx._src.types.Model:
    """
    Puts mujoco.MjModel onto a device, resulting in mjx.Model.
    
      Args:
        m: the model to put onto device
        device: which device to use - if unspecified picks the default device
        _full_compat: put all MjModel fields onto device irrespective of MJX support
          This is an experimental feature.  Avoid using it for now.
    
      Returns:
        an mjx.Model placed on device
      
    """

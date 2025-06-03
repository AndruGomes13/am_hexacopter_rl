from abc import abstractmethod, ABC
from enum import StrEnum
from typing import Callable, Protocol, Type
from environments.hexcopter.state_interfaces import AugmentedPipelineState, SensorData
from models.hexcopter import get_allocation_matrix, get_inertia_matrix, get_max_propeller_thurst, get_thrust_torque_limits, get_mass
import jax.numpy as jp

G = 9.81


class ActionModelType(StrEnum):
    DIRECT_PROP = "direct_prop"
    THRUST_TORQUE = "thrust_torque"
    NORM_THRUST_BODY_RATE = "norm_thrust_body_rate"


class ActionModel(ABC):
    @classmethod
    @abstractmethod
    def propeller_action_from_normalized_action(cls, normalized_action: jp.ndarray, sensor_data: SensorData) -> jp.ndarray:
        pass

    @classmethod
    @abstractmethod
    def get_action_dim(cls) -> int:
        pass


class DirectPropellerThrust(ActionModel):
    MAX_PROP_CTRL_ACTION = get_max_propeller_thurst()

    @classmethod
    def propeller_action_from_normalized_action(cls, normalized_action: jp.ndarray, sensor_data: SensorData) -> jp.ndarray:
        prop_action = ((normalized_action + 1.0) * 0.5) * cls.MAX_PROP_CTRL_ACTION
        return prop_action

    @classmethod
    def get_action_dim(cls) -> int:
        return 4


class ThrustTorque(ActionModel):
    ALLOCATION_MATRIX_INV = jp.linalg.pinv(get_allocation_matrix())
    ACTUATION_LIMITS = get_thrust_torque_limits()
    MAX_PROP_THRUST = get_max_propeller_thurst()
    MIN_PROP_THRUST = 0

    @classmethod
    def propeller_action_from_normalized_action(cls, normalized_action: jp.ndarray, sensor_data: SensorData) -> jp.ndarray:
        scaled_action = (normalized_action + 1) * 0.5
        scaled_action = cls.ACTUATION_LIMITS[0] + (cls.ACTUATION_LIMITS[1] - cls.ACTUATION_LIMITS[0]) * scaled_action
        prop_action = cls.propeller_mixing(scaled_action)
        return prop_action

    @classmethod
    def get_action_dim(cls) -> int:
        return 4

    @classmethod
    def propeller_mixing(cls, thrust_torque_command: jp.ndarray):
        prop_raw = cls.ALLOCATION_MATRIX_INV @ thrust_torque_command
        return prop_raw
        shift = jp.maximum(cls.MIN_PROP_THRUST - jp.min(prop_raw), 0)
        scale = jp.minimum(1, (cls.MAX_PROP_THRUST - cls.MIN_PROP_THRUST) / (jp.max(prop_raw) + shift - cls.MIN_PROP_THRUST))
        prop_tmp = prop_raw + shift
        prop = cls.MIN_PROP_THRUST + scale * (prop_tmp - cls.MIN_PROP_THRUST)
        return prop


class NormThrustBodyRate(ActionModel):
    INERTIA_MATRIX = get_inertia_matrix()
    DEG = jp.pi / 180
    MASS = get_mass()

    # --- Compute scaling values ---
    min_thurst_torque, max_thurst_torque = get_thrust_torque_limits()
    fx, fy, fz = max_thurst_torque[:3]
    wx = 600 * DEG
    wy = wx
    wz = 250 * DEG

    ANG_VEL_SCALLING = jp.array([wx, wy, wz])
    MIN_CTRL = jp.concatenate([min_thurst_torque[:3],-ANG_VEL_SCALLING])
    MAX_CTRL = jp.concatenate([max_thurst_torque[:3],ANG_VEL_SCALLING])

    # --- Controller params ---
    Kp = jp.array([20.0, 20.0, 41.0])

    @classmethod
    def propeller_action_from_normalized_action(cls, normalized_action: jp.ndarray, sensor_data: SensorData) -> jp.ndarray:
        w = sensor_data.drone_imu_angular_velocity
        unormalized_action = cls.unormalize_actions(normalized_action)
        error_body_rate = unormalized_action[3:] - w
        desired_ang_acc = cls.Kp * error_body_rate
        desired_torque = cls.INERTIA_MATRIX @ desired_ang_acc + jp.cross(w, cls.INERTIA_MATRIX @ w)

        desired_thrust_torque = jp.concatenate((unormalized_action[:3], desired_torque))
        prop_action = ThrustTorque.propeller_mixing(desired_thrust_torque)

        return prop_action

    @classmethod
    def get_action_dim(cls) -> int:
        return 6
    
    @classmethod
    def unormalize_actions(cls, normalized_action: jp.ndarray):
        return cls.MIN_CTRL + ((normalized_action + 1) / 2) * (cls.MAX_CTRL - cls.MIN_CTRL)

    


def action_model_factory(action_model_type: ActionModelType) -> Type[ActionModel]:
    if action_model_type == ActionModelType.DIRECT_PROP:
        return DirectPropellerThrust
    elif action_model_type == ActionModelType.THRUST_TORQUE:
        return ThrustTorque
    elif action_model_type == ActionModelType.NORM_THRUST_BODY_RATE:
        return NormThrustBodyRate
    else:
        raise ValueError(f"Not valid action model: {action_model_type}")


# Rotor Dynamics
class PropellerThrustDynamicsFn(Protocol):
    def __call__(self, propeller_thrust_request: jp.ndarray, state: AugmentedPipelineState) -> jp.ndarray: ...

class PropellerOmegaDynamicsFn(Protocol):
    def __call__(self, propeller_omega_request: jp.ndarray, state: AugmentedPipelineState) -> jp.ndarray: ...

class ThrustMapFn(Protocol):
    def __call__(self, propeller_omega: jp.ndarray, state: AugmentedPipelineState) -> jp.ndarray: ...

class InverseThrustMapFn(Protocol):
    def __call__(self, propeller_thrust_request: jp.ndarray) -> jp.ndarray: ...

# TODO: Might be useful to add a level of indirection from force -> omega -> force (can simulate model mismatch)
def get_thrust_dynamics_fn(rotor_tau: float, timestep: float) -> PropellerThrustDynamicsFn:
    TAU = rotor_tau
    T_STEP = timestep
    ALPHA = jp.exp(-T_STEP / TAU)

    def propeller_dynamics(propeller_thrust_request: jp.ndarray, state: AugmentedPipelineState) -> jp.ndarray:
        norm_propeller_thrust_request = propeller_thrust_request / get_max_propeller_thurst()

        new_norm_propeller_thurst = ALPHA * state.prop_action_norm + (1 - ALPHA) * norm_propeller_thrust_request

        return new_norm_propeller_thurst * get_max_propeller_thurst()

    return propeller_dynamics

def get_rotor_omega_dynamics_fn(rotor_tau: float, timestep: float) -> PropellerOmegaDynamicsFn:
    TAU = rotor_tau
    T_STEP = timestep
    ALPHA = jp.exp(-T_STEP / TAU)

    def propeller_dynamics(propeller_omega_request: jp.ndarray, state: AugmentedPipelineState) -> jp.ndarray:

        new_propeller_omega = ALPHA * state.prop_omega + (1 - ALPHA) * propeller_omega_request

        return new_propeller_omega 

    return propeller_dynamics


def get_thrust_map_and_inv_map_fn(thrust_map_coefficients: jp.ndarray,) -> tuple[ThrustMapFn, InverseThrustMapFn]:
    def thrust_map(propeller_omega: jp.ndarray, state: AugmentedPipelineState) -> jp.ndarray:
        """ Maps propeller omega to propeller thrust."""
        return jp.polyval(state.domain_rand_state.thrust_map_coefficients, propeller_omega)

    def inverse_thrust_map(propeller_thrust_request: jp.ndarray) -> jp.ndarray:
        """ Maps propeller thrust request to propeller omega request."""
        p = thrust_map_coefficients
        a, b, c= p[0], p[1], p[2] - propeller_thrust_request
        
        # Solve the quadratic equation a*x^2 + b*x + c = 0 #NOTE: Assume omega is always positive
        discriminant = b**2 - 4*a*c
        omega = (-b + jp.sqrt(discriminant)) / (2 * a)
        return omega
    
    return thrust_map, inverse_thrust_map

        
    

if __name__ == "__main__":
    s = SensorData(1, 1, 1, 1, 1, 1)
    p = DirectPropellerThrust.propeller_action_from_normalized_action(jp.array([1, 0, 0, 1]), s)
    print(p)

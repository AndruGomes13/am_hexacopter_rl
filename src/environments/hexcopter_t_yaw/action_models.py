from abc import abstractmethod, ABC
from enum import StrEnum
from typing import Callable, Protocol, Type
from environments.hexcopter_t_yaw.state_interfaces import AugmentedPipelineState, SensorData
from models.hexcopter_t_yaw import get_inertia_matrix, get_max_propeller_thurst, get_mass, CtrlIds
import jax.numpy as jp

G = 9.81


class ActionModelType(StrEnum):
    F_YAW_JOINT_POS = "force_yaw"
    F_WZ_JOINT_POS = "force_wz" 


class PropellerActionModel(ABC):
    @classmethod
    @abstractmethod
    def propeller_action_from_normalized_action(cls, normalized_action: jp.ndarray, sensor_data: SensorData) -> jp.ndarray:
        pass

    @classmethod
    @abstractmethod
    def get_action_dim(cls) -> int:
        pass


class ActionModel(ABC):
    @classmethod
    @abstractmethod
    def mujoco_ctrl_from_normalized_action(cls, normalized_action: jp.ndarray, state: AugmentedPipelineState) -> jp.ndarray:
        pass

    @classmethod
    @abstractmethod
    def get_action_dim(cls) -> int:
        pass

# ---------    

class ForceWzJointPos(ActionModel):
    F_LIMITS_SYM = jp.array([15, 15, 15, 10])  # Force limits for x, y, z, and wz
    PITCH_1_LIMITS = jp.array([0, 180])  # Pitch limits for arm pitch 1
    PITCH_2_LIMITS = jp.array([-10, 45])  # Pitch limits for arm pitch 2
    
    @classmethod
    def mujoco_ctrl_from_normalized_action(cls, normalized_action: jp.ndarray, state: AugmentedPipelineState) -> jp.ndarray:
        base_normalized_action = normalized_action[:4]
        arm_1_pitch = normalized_action[4]
        arm_2_pitch = normalized_action[5]

        # Scale the action
        base_action = base_normalized_action * cls.F_LIMITS_SYM
        arm_1_pitch = cls.PITCH_1_LIMITS[0] + (arm_1_pitch + 1) * 0.5 * (cls.PITCH_1_LIMITS[1] - cls.PITCH_1_LIMITS[0])
        arm_2_pitch = cls.PITCH_2_LIMITS[0] + (arm_2_pitch + 1) * 0.5 * (cls.PITCH_2_LIMITS[1] - cls.PITCH_2_LIMITS[0])

        # Clamp arm 2 pitch
        arm_2_pitch = jp.clip(arm_2_pitch, -arm_1_pitch)

        # The pitch 2 is relative to pitch 1, so we add them together
        arm_2_joint = arm_2_pitch + arm_1_pitch # NOTE: The pitch of arm 2 is absolute. Thus the joint angle is the sum of both pitches.

        # Clamp the arm pitch delta
        arm_pitch_1_delta = arm_1_pitch - state.last_ctrl[4]
        arm_2_joint_delta = arm_2_joint - state.last_ctrl[5]
        arm_pitch_1_delta_clipped = jp.clip(arm_pitch_1_delta, -5, 5)
        arm_2_joint_delta_clipped = jp.clip(arm_2_joint_delta, -5, 5)
        arm_1_pitch = state.last_ctrl[4] + arm_pitch_1_delta_clipped
        arm_2_joint = state.last_ctrl[5] + arm_2_joint_delta_clipped

        # Create the control vector
        ctrl = jp.zeros(cls.get_action_dim(), dtype=jp.float32)
        ctrl = populate_ctrl_array(
            base_action[:3],
            base_action[3],  # wz
            arm_1_pitch,
            arm_2_joint,
            ctrl
        )
        return ctrl

    @classmethod
    def get_action_dim(cls) -> int:
        return 6



def action_model_factory(action_model_type: ActionModelType) -> Type[ActionModel]:
    if action_model_type == ActionModelType.F_WZ_JOINT_POS:
        return ForceWzJointPos
    else:
        raise ValueError(f"Not valid action model: {action_model_type}")


# --- Control Array Population ---

def populate_ctrl_array(
    f: jp.ndarray,
    wz: float,
    arm_pitch_1: float,
    arm_pitch_2: float,
    ctrl: jp.ndarray = jp.zeros(6, dtype=jp.float32),
) -> jp.ndarray:
    """
    Populate the control array with the action values.
    """
    ctrl = ctrl.at[jp.array([CtrlIds.ACTUATOR_X.value, CtrlIds.ACTUATOR_Y.value, CtrlIds.ACTUATOR_Z.value])].set(f)
    ctrl = ctrl.at[CtrlIds.ACTUATOR_WZ.value].set(wz)
    ctrl = ctrl.at[CtrlIds.ACTUATOR_ARM_PITCH_1.value].set(arm_pitch_1)
    ctrl = ctrl.at[CtrlIds.ACTUATOR_ARM_PITCH_2.value].set(arm_pitch_2)
    return ctrl
 

if __name__ == "__main__":
    s = SensorData(1, 1, 1, 1, 1, 1)
    p = DirectPropellerThrust.propeller_action_from_normalized_action(jp.array([1, 0, 0, 1]), s)
    print(p)

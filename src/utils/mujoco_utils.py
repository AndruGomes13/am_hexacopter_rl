from typing import Tuple
import mujoco
import jax.numpy as jp

# --- Joint related ---
def get_qpos_qvel_slice_from_free_joint_body(body_name: str, model : mujoco.MjModel) -> Tuple[jp.ndarray, jp.ndarray]:
    """
    This function returns the qpos and qvel slices for a body with a free joint. This is specific to the input model.

    TODO: Generalize to multiple joints type + bodies with multiple joints
    """
    body = model.body(body_name)
    jnt_id = body.jntadr[0] # If a body has a free joint, it should only have one joint.
    if model.joint(jnt_id).type != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(f"Body {body_name} does not have a free joint. Joint type: {model.joint(jnt_id).type}")
    
    return get_qpos_qvel_slice_from_joint_id(jnt_id, model)
   
def get_qpos_qvel_slice_from_joint_id(joint_id: str | int, model: mujoco.MjModel) -> Tuple[jp.ndarray, jp.ndarray]:
    """
    This function returns the qpos and qvel slices for a joint. This is specific to the input model.

    It accepts either a joint name (str) or a joint ID (int).
    The joint ID is the index of the joint in the model's joint array.
    """
    joint = model.joint(joint_id)
    jnt_id = joint.id
    qpos_start_id = model.jnt_qposadr[jnt_id]
    qvel_start_id = model.jnt_dofadr[jnt_id]

    if joint.type == mujoco.mjtJoint.mjJNT_FREE:
        qpos_slice = jp.arange(qpos_start_id, qpos_start_id + 7)  # 7 for free joints
        qvel_slice = jp.arange(qvel_start_id, qvel_start_id + 6)  # 6 for free joints
    elif joint.type == mujoco.mjtJoint.mjJNT_BALL:
        qpos_slice = jp.arange(qpos_start_id, qpos_start_id + 4)
        qvel_slice = jp.arange(qvel_start_id, qvel_start_id + 3)
    elif joint.type == mujoco.mjtJoint.mjJNT_HINGE or joint.type == mujoco.mjtJoint.mjJNT_SLIDE:
        qpos_slice = jp.arange(qpos_start_id, qpos_start_id + 1)
        qvel_slice = jp.arange(qvel_start_id, qvel_start_id + 1)
    else:
        raise ValueError(f"Unsupported joint type: {joint.type}")
    

    return qpos_slice, qvel_slice

def get_body_joint_ids(body_name: str, model: mujoco.MjModel) -> list[int]:
    """
    Returns a list of joint names that are available for a given body.
    """
    body = model.body(body_name)
    joint_start = body.jntadr[0]
    joint_num = body.jntnum[0]
    joint_ids = list(range(joint_start, joint_start + joint_num))
    return joint_ids

# --- Actuator related ---
def get_actuator_id_from_actuator_name(actuator_name: str, model: mujoco.MjModel) -> jp.ndarray:
    """
    Returns the index of the actuator for a given actuator name. This is the same index as the MjData.ctrl.
    """
    actuator = model.actuator(actuator_name)
    return actuator.id
    
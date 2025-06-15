import itertools
from pathlib import Path
from enum import Enum, StrEnum
import tempfile
from typing import Tuple
import IPython
import jax.numpy as jp
from jinja2 import Template
import scipy
import scipy.spatial
from scipy.spatial.transform import Rotation
import mujoco
import numpy as np
from utils.mujoco_utils import get_qpos_qvel_slice_from_joint_id

BASE_DIR = Path(__file__).resolve().parent


def _get_hex_template_xml_path() -> Path:
    XML_PATH = BASE_DIR / "hex_template.xml"
    return XML_PATH


def get_hex_xml_path() -> Path:
    variables: dict[str, Path] = {"base_dir": BASE_DIR,
                                  "mesh_dir": BASE_DIR / "meshes",}

    rendered_xml = Template(_get_hex_template_xml_path().read_text()).render(variables)
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".xml", prefix="rendered_", delete=False) as tmp:
        tmp.write(rendered_xml)
        tmp.flush()
        tmp_path = Path(tmp.name)
    return tmp_path


def get_hex_xml_str() -> str:
    with open(get_hex_xml_path()) as f:
        xml = f.read()
    return xml

model_spec = mujoco.MjSpec.from_file(get_hex_xml_path().as_posix())
model = model_spec.compile()
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

class ComponentNames(StrEnum):
    BODY = "hex"
    # --- Sensors body
    SENSOR_IMU_QUAT = "body_quat"
    SENSOR_IMU_POS = "body_pos"
    SENSOR_IMU_GYRO = "body_gyro"
    SENSOR_IMU_ACC = "body_linacc"
    SENSOR_IMU_LIN_VEL = "body_linvel"
    SENSOR_IMU_ANG_VEL = "body_angvel"
    # --- Sensors arm
    SENSOR_END_EFFECTOR_POS = "end_effector_pos"
    SENSOR_END_EFFECTOR_ORIENTATION = "end_effector_quat"

    # --- Joints and actuators
    ACTUATOR_X = "fx"
    ACTUATOR_Y = "fy"
    ACTUATOR_Z = "fz"
    ACTUATOR_WZ = "wz"
    ACTUATOR_ARM_PITCH_1 = "arm_pitch_1"
    ACTUATOR_ARM_PITCH_2 = "arm_pitch_2" 

    JOINT_X = "root_x"
    JOINT_Y = "root_y"
    JOINT_Z = "root_z"
    JOINT_YAW = "root_yaw"
    JOINT_ARM_PITCH_1 = "arm_link1_pitch_joint"  
    JOINT_ARM_PITCH_2 = "arm_link2_pitch_joint" 

class CtrlIds(Enum):
    # --- Joints and actuators
    ACTUATOR_X = model.actuator(ComponentNames.ACTUATOR_X).id
    ACTUATOR_Y = model.actuator(ComponentNames.ACTUATOR_Y).id
    ACTUATOR_Z = model.actuator(ComponentNames.ACTUATOR_Z).id
    ACTUATOR_WZ = model.actuator(ComponentNames.ACTUATOR_WZ).id
    ACTUATOR_ARM_PITCH_1 = model.actuator(ComponentNames.ACTUATOR_ARM_PITCH_1).id
    ACTUATOR_ARM_PITCH_2 = model.actuator(ComponentNames.ACTUATOR_ARM_PITCH_2).id

class JointSlices(Enum):
    JOINT_X_QPOS = get_qpos_qvel_slice_from_joint_id(ComponentNames.JOINT_X.value, model)[0]
    JOINT_X_QVEL = get_qpos_qvel_slice_from_joint_id(ComponentNames.JOINT_X.value, model)[1]
    JOINT_Y_QPOS = get_qpos_qvel_slice_from_joint_id(ComponentNames.JOINT_Y.value, model)[0]
    JOINT_Y_QVEL = get_qpos_qvel_slice_from_joint_id(ComponentNames.JOINT_Y.value, model)[1]
    JOINT_Z_QPOS = get_qpos_qvel_slice_from_joint_id(ComponentNames.JOINT_Z.value, model)[0]
    JOINT_Z_QVEL = get_qpos_qvel_slice_from_joint_id(ComponentNames.JOINT_Z.value, model)[1]
    JOINT_YAW_QPOS = get_qpos_qvel_slice_from_joint_id(ComponentNames.JOINT_YAW.value, model)[0]
    JOINT_YAW_QVEL = get_qpos_qvel_slice_from_joint_id(ComponentNames.JOINT_YAW.value, model)[1]
    JOINT_ARM_PITCH_1_QPOS = get_qpos_qvel_slice_from_joint_id(ComponentNames.JOINT_ARM_PITCH_1.value, model)[0]
    JOINT_ARM_PITCH_1_QVEL = get_qpos_qvel_slice_from_joint_id(ComponentNames.JOINT_ARM_PITCH_1.value, model)[1]
    JOINT_ARM_PITCH_2_QPOS = get_qpos_qvel_slice_from_joint_id(ComponentNames.JOINT_ARM_PITCH_2.value, model)[0]
    JOINT_ARM_PITCH_2_QVEL = get_qpos_qvel_slice_from_joint_id(ComponentNames.JOINT_ARM_PITCH_2.value, model)[1]



def get_inertia_matrix() -> jp.ndarray:
    x2_body = model.body(ComponentNames.BODY)
    I_diag = x2_body.inertia
    iq = x2_body.iquat

    R_i = scipy.spatial.transform.Rotation.from_quat(iq, scalar_first=True)
    I_body_frame = R_i.as_matrix() @ np.diag(I_diag) @ R_i.as_matrix().T
    return jp.array(I_body_frame)


def get_mass() -> float:
    x2_body = model.body(ComponentNames.BODY)
    mass = x2_body.mass[0]
    return mass


def get_max_propeller_thurst() -> float:
    return float(model.actuator_ctrlrange[0, 1])


if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)

    print(np.rad2deg(-0.523698))
    IPython.embed()
    
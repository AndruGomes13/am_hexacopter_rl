import itertools
from pathlib import Path
from enum import StrEnum
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

BASE_DIR = Path(__file__).resolve().parent


def _get_source_x2_3d_xml_path() -> Path:
    XML_PATH = BASE_DIR / "hex_template.xml"
    return XML_PATH


def get_x2_3d_xml_path() -> Path:
    variables: dict[str, Path] = {"base_dir": BASE_DIR}

    rendered_xml = Template(_get_source_x2_3d_xml_path().read_text()).render(variables)
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".xml", prefix="rendered_", delete=False) as tmp:
        tmp.write(rendered_xml)
        tmp.flush()
        tmp_path = Path(tmp.name)
    return tmp_path


def get_x2_3d_xml_str() -> str:
    with open(get_x2_3d_xml_path()) as f:
        xml = f.read()
    return xml


class ComponentNames(StrEnum):
    SENSOR_IMU_QUAT = "body_quat"
    SENSOR_IMU_POS = "body_pos"
    SENSOR_IMU_GYRO = "body_gyro"
    SENSOR_IMU_ACC = "body_linacc"
    SENSOR_IMU_LIN_VEL = "body_linvel"
    SENSOR_IMU_ANG_VEL = "body_angvel"
    ACTUATOR_PROP_1 = "thrust1"
    ACTUATOR_PROP_2 = "thrust2"
    ACTUATOR_PROP_3 = "thrust3"
    ACTUATOR_PROP_4 = "thrust4"
    ACTUATOR_PROP_5 = "thrust5"
    ACTUATOR_PROP_6 = "thrust6"
    SITE_PROP_1 = "thrust1"
    SITE_PROP_2 = "thrust2"
    SITE_PROP_3 = "thrust3"
    SITE_PROP_4 = "thrust4"
    SITE_PROP_5 = "thrust5"
    SITE_PROP_6 = "thrust6"
    BODY = "hex"
    GEOM_PADDLE = "paddle"


model_spec = mujoco.MjSpec.from_file(get_x2_3d_xml_path().as_posix())
model = model_spec.compile()
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)


def get_allocation_matrix() -> jp.ndarray:
    """
    Returns the allocation matrix -> Maps actuation level to thrust and torques
    """
    x2_body = model.body(ComponentNames.BODY)
    x2_center_of_mass = x2_body.ipos

    # --- Get Propellor thrust vectors, lever arms and torques
    def get_actuator_contribution_bf(actuator_name: str, site_name: str) -> Tuple[np.ndarray, np.ndarray]:
        site = model.site(site_name)
        actuator = model.actuator(actuator_name)

        actuator_pos = site.pos
        actuator_quat_wxyz = site.quat
        actuator_rot = Rotation.from_quat(actuator_quat_wxyz, scalar_first=True)

        actuator_gear = actuator.gear
        thrust_vec = actuator_rot.apply(actuator_gear[:3])
        drag_torque_vec = actuator_rot.apply(actuator_gear[3:])
        lever_arm = actuator_pos - x2_center_of_mass
        thrust_torque_vec = jp.cross(lever_arm, thrust_vec)
        total_torque = thrust_torque_vec + drag_torque_vec

        return thrust_vec, total_torque

    f1, t1 = get_actuator_contribution_bf(ComponentNames.ACTUATOR_PROP_1, ComponentNames.SITE_PROP_1)
    f2, t2 = get_actuator_contribution_bf(ComponentNames.ACTUATOR_PROP_2, ComponentNames.SITE_PROP_2)
    f3, t3 = get_actuator_contribution_bf(ComponentNames.ACTUATOR_PROP_3, ComponentNames.SITE_PROP_3)
    f4, t4 = get_actuator_contribution_bf(ComponentNames.ACTUATOR_PROP_4, ComponentNames.SITE_PROP_4)
    f5, t5 = get_actuator_contribution_bf(ComponentNames.ACTUATOR_PROP_5, ComponentNames.SITE_PROP_5)
    f6, t6 = get_actuator_contribution_bf(ComponentNames.ACTUATOR_PROP_6, ComponentNames.SITE_PROP_6)

    full_allocation_matrix = jp.array(
    [
        [f1[0], f2[0], f3[0], f4[0], f5[0], f6[0]],
        [f1[1], f2[1], f3[1], f4[1], f5[1], f6[1]],
        [f1[2], f2[2], f3[2], f4[2], f5[2], f6[2]],
        [t1[0], t2[0], t3[0], t4[0], t5[0], t6[0]],
        [t1[1], t2[1], t3[1], t4[1], t5[1], t6[1]],
        [t1[2], t2[2], t3[2], t4[2], t5[2], t6[2]],
    ]
)
    return full_allocation_matrix#[2:, :]


def get_allocation_matrix_2D() -> jp.ndarray:
    allocation_matrix = get_allocation_matrix()
    allocation_matrix_2d = allocation_matrix[[0, 2], :]
    return jp.array(allocation_matrix_2d)


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


def get_thrust_torque_limits():
    combinations = np.array(list(itertools.product([0, 1], repeat=6)))
    propeller_acts = combinations * get_max_propeller_thurst()
    A = get_allocation_matrix()
    min_act = np.zeros(6)
    max_act = np.zeros(6)
    for p in propeller_acts:
        act = A @ np.array(p)
        min_act = np.minimum(min_act, act)
        max_act = np.maximum(max_act, act)

    return min_act * 0.9, max_act * 0.9


if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)
    alloc = get_allocation_matrix()
    alloc_inv = np.linalg.inv(alloc)
    print(alloc_inv)
    print(np.rad2deg(-0.523698))
    
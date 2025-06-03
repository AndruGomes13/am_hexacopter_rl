from enum import Enum
from pathlib import Path
import tempfile
from typing import Tuple

import mujoco
import jax.numpy as jp
# from models import skydio_x2 as model
from models import hexcopter as model
from jinja2 import Template

BASE_DIR = Path(__file__).resolve().parent


# --- Environment XML related ---
def get_env_xml_path() -> Path:
    """
    Loads and parses the xml.

    """
    variables = {
        "hex_path": model.get_x2_3d_xml_path(),
    }
    xml_template = (BASE_DIR / "scene.xml").read_text()
    rendered_xml = Template(xml_template).render(variables)
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".xml", prefix="rendered_", delete=False) as tmp:
        tmp.write(rendered_xml)
        tmp.flush()
        tmp_path = Path(tmp.name)
    return tmp_path

if __name__ == "__main__":
    from brax.io import mjcf
    import mujoco
    import IPython

    mj_model = mujoco.MjModel.from_xml_path(get_env_xml_path().as_posix())
    sys = mjcf.load_model(mj_model)
    IPython.embed()

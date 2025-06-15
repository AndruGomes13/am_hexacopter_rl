from enum import Enum
from pathlib import Path
import tempfile
from typing import NamedTuple, Tuple

import mujoco
import jax.numpy as jp
import jax
# from models import skydio_x2 as model
from models import hexcopter_t_yaw as model
from jinja2 import Template

BASE_DIR = Path(__file__).resolve().parent


# --- Environment XML related ---
def get_env_xml_path() -> Path:
    """
    Loads and parses the xml.

    """
    variables = {
        "hex_path": model.get_hex_xml_path(),
    }
    xml_template = (BASE_DIR / "scene.xml").read_text()
    rendered_xml = Template(xml_template).render(variables)
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".xml", prefix="rendered_", delete=False) as tmp:
        tmp.write(rendered_xml)
        tmp.flush()
        tmp_path = Path(tmp.name)
    return tmp_path


# --- Trajectory generation ---
class Spline1D(NamedTuple):
    t_wp: jp.ndarray      # shape (N,)
    a: jp.ndarray         # shape (N-1,)
    b: jp.ndarray         # shape (N-1,)
    c: jp.ndarray         # shape (N-1,)
    d: jp.ndarray         # shape (N-1,)
    h: jp.ndarray         # shape (N-1,)

class TrajectoryState(NamedTuple):
    key: jp.ndarray       
    splines: Spline1D      # x-axis spline
    spliney: Spline1D      # y-axis spline
    splinez: Spline1D      # z-axis spline
    duration: float

def _compute_natural_cubic_spline(t_wp: jp.ndarray,
                                  y_wp: jp.ndarray) -> Spline1D:
    N = t_wp.shape[0]
    h = t_wp[1:] - t_wp[:-1]                            # (N-1,)
    # build alpha
    dy = y_wp[1:] - y_wp[:-1]
    alpha = jp.zeros(N)
    alpha = alpha.at[1:-1].set(
        3*(dy[1:]/h[1:] - dy[:-1]/h[:-1])
    )
    # build tridiagonal matrix A
    main = jp.concatenate([jp.array([1.]), 2*(h[:-1] + h[1:]), jp.array([1.])])
    off  = h
    A = jp.diag(main) + jp.diag(off, 1) + jp.diag(off, -1)
    # solve for second derivatives m
    m = jp.linalg.solve(A, alpha)                      # (N,)
    # compute coefficients for each segment
    a = y_wp[:-1]
    b = dy/h - (h*(2*m[:-1] + m[1:]))/6
    c = m[:-1]/2
    d = (m[1:] - m[:-1])/(6*h)
    return Spline1D(t_wp=t_wp, a=a, b=b, c=c, d=d, h=h)

def init_trajectory(key: jp.ndarray,
                    num_waypoints: int = 6,
                    duration: float = 3.0,
                    workspace_limits: dict = None) -> TrajectoryState:
    if workspace_limits is None:
        workspace_limits = {'x': (-1.0, 1.0),
                            'y': (-1.0, 1.0),
                            'z': ( 0.3, 1.5)}
    # 1) sample waypoints times
    t_wp = jp.linspace(0., duration, num_waypoints)
    # 2) sample random waypoints in each axis
    key, *subkeys = jax.random.split(key, 4)
    xs = jax.random.uniform(subkeys[0], (num_waypoints,),
                        minval=workspace_limits['x'][0],
                        maxval=workspace_limits['x'][1])
    ys = jax.random.uniform(subkeys[1], (num_waypoints,),
                        minval=workspace_limits['y'][0],
                        maxval=workspace_limits['y'][1])
    zs = jax.random.uniform(subkeys[2], (num_waypoints,),
                        minval=workspace_limits['z'][0],
                        maxval=workspace_limits['z'][1])
    # 3) build splines per axis
    splinex = _compute_natural_cubic_spline(t_wp, xs)
    spliney = _compute_natural_cubic_spline(t_wp, ys)
    splinez = _compute_natural_cubic_spline(t_wp, zs)
    return TrajectoryState(key=key,
                           splines=splinex,
                           spliney=spliney,
                           splinez=splinez,
                           duration=duration)

def eval_spline(spline: Spline1D, t_vals: jp.ndarray):
        # find segment indices
        idx = jp.clip(jp.searchsorted(spline.t_wp, t_vals) - 1,
                       0, spline.a.shape[0]-1)
        s = t_vals - spline.t_wp[idx]
        p = spline.a[idx] + spline.b[idx]*s + spline.c[idx]*s**2 + spline.d[idx]*s**3
        v =            spline.b[idx]   + 2*spline.c[idx]*s + 3*spline.d[idx]*s**2
        acc =                    2*spline.c[idx] + 6*spline.d[idx]*s
        return p, v, acc

def sample_trajectory(state: TrajectoryState,
                      t_start: float = 0.0,
                      preview_dt: float = 0.2,
                      K: int = 5):
    """
    control_dt  : how much we advance state.t each tick
    preview_dt  : spacing between way-points in the preview window
    K           : number of preview points
    """
    # --- 1) compute the future sample times ---
    # build the look-ahead grid
    times = t_start + jp.arange(K) * preview_dt
    # clamp to avoid splining outside [0,duration]
    times = jp.minimum(times, state.duration)

    # --- 2) evaluate splines at 'times' exactly as before ---
    px, vx, ax = eval_spline(state.splines, times)
    py, vy, ay = eval_spline(state.spliney, times)
    pz, vz, az = eval_spline(state.splinez, times)

    positions     = jp.stack([px, py, pz], axis=1)
    velocities    = jp.stack([vx, vy, vz], axis=1)
    accelerations = jp.stack([ax, ay, az], axis=1)

    # --- 3) advance the internal clock by control_dt ---
    return positions, velocities, accelerations


if __name__ == "__main__":
    from brax.io import mjcf
    import mujoco
    import IPython

    mj_model = mujoco.MjModel.from_xml_path(get_env_xml_path().as_posix())
    sys = mjcf.load_model(mj_model)
    IPython.embed()

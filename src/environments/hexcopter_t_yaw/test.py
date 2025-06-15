import jax
import jax.numpy as jnp
from jax import random
from typing import NamedTuple

class Spline1D(NamedTuple):
    t_wp: jnp.ndarray      # shape (N,)
    a: jnp.ndarray         # shape (N-1,)
    b: jnp.ndarray         # shape (N-1,)
    c: jnp.ndarray         # shape (N-1,)
    d: jnp.ndarray         # shape (N-1,)
    h: jnp.ndarray         # shape (N-1,)

class TrajectoryState(NamedTuple):
    key: jnp.ndarray       
    splines: Spline1D      # x-axis spline
    spliney: Spline1D      # y-axis spline
    splinez: Spline1D      # z-axis spline
    duration: float

def _compute_natural_cubic_spline(t_wp: jnp.ndarray,
                                  y_wp: jnp.ndarray) -> Spline1D:
    N = t_wp.shape[0]
    h = t_wp[1:] - t_wp[:-1]                            # (N-1,)
    # build alpha
    dy = y_wp[1:] - y_wp[:-1]
    alpha = jnp.zeros(N)
    alpha = alpha.at[1:-1].set(
        3*(dy[1:]/h[1:] - dy[:-1]/h[:-1])
    )
    # build tridiagonal matrix A
    main = jnp.concatenate([jnp.array([1.]), 2*(h[:-1] + h[1:]), jnp.array([1.])])
    off  = h
    A = jnp.diag(main) + jnp.diag(off, 1) + jnp.diag(off, -1)
    # solve for second derivatives m
    m = jnp.linalg.solve(A, alpha)                      # (N,)
    # compute coefficients for each segment
    a = y_wp[:-1]
    b = dy/h - (h*(2*m[:-1] + m[1:]))/6
    c = m[:-1]/2
    d = (m[1:] - m[:-1])/(6*h)
    return Spline1D(t_wp=t_wp, a=a, b=b, c=c, d=d, h=h)

def init_trajectory(key: jnp.ndarray,
                    num_waypoints: int = 6,
                    duration: float = 3.0,
                    workspace_limits: dict = None) -> TrajectoryState:
    if workspace_limits is None:
        workspace_limits = {'x': (-0.5, 0.5),
                            'y': (-0.5, 0.5),
                            'z': ( 0.2, 1.0)}
    # 1) sample waypoints times
    t_wp = jnp.linspace(0., duration, num_waypoints)
    # 2) sample random waypoints in each axis
    key, *subkeys = random.split(key, 4)
    xs = random.uniform(subkeys[0], (num_waypoints,),
                        minval=workspace_limits['x'][0],
                        maxval=workspace_limits['x'][1])
    ys = random.uniform(subkeys[1], (num_waypoints,),
                        minval=workspace_limits['y'][0],
                        maxval=workspace_limits['y'][1])
    zs = random.uniform(subkeys[2], (num_waypoints,),
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

def eval_spline(spline: Spline1D, t_vals: jnp.ndarray):
        # find segment indices
        idx = jnp.clip(jnp.searchsorted(spline.t_wp, t_vals) - 1,
                       0, spline.a.shape[0]-1)
        s = t_vals - spline.t_wp[idx]
        p = spline.a[idx] + spline.b[idx]*s + spline.c[idx]*s**2 + spline.d[idx]*s**3
        v =            spline.b[idx]   + 2*spline.c[idx]*s + 3*spline.d[idx]*s**2
        acc =                    2*spline.c[idx] + 6*spline.d[idx]*s
        return p, v, acc

def sample_trajectory(state: TrajectoryState,
                      t: float = 0.0,
                      preview_dt: float = 0.2,
                      K: int = 5):
    """
    control_dt  : how much we advance state.t each tick
    preview_dt  : spacing between way-points in the preview window
    K           : number of preview points
    """
    # --- 1) compute the future sample times ---
    # get current time along the trajectory
    t0 = t
    # build the look-ahead grid
    times = t0 + jnp.arange(K) * preview_dt
    # clamp to avoid splining outside [0,duration]
    times = jnp.minimum(times, state.duration)

    # --- 2) evaluate splines at 'times' exactly as before ---
    px, vx, ax = eval_spline(state.splines, times)
    py, vy, ay = eval_spline(state.spliney, times)
    pz, vz, az = eval_spline(state.splinez, times)

    positions     = jnp.stack([px, py, pz], axis=1)
    velocities    = jnp.stack([vx, vy, vz], axis=1)
    accelerations = jnp.stack([ax, ay, az], axis=1)

    # --- 3) advance the internal clock by control_dt ---
    return positions, velocities, accelerations


if __name__ == "__main__":
    key = random.PRNGKey(0)
    state = init_trajectory(key, num_waypoints=20, duration=60.0)
    # in your training loop:
    pos, vel, acc = sample_trajectory(state, t=0, preview_dt=0.1, K=10)
    print(pos.shape)

    # for _ in range(1):              # 10 control ticks
    # # print(pos)
    # # Plot the trajectory
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 6))
    # print(pos[:, 0])
    # plt.plot(pos[:, 0], pos[:, 1], label='Trajectory (XY plane)')
    # plt.scatter(pos[0, 0], pos[0, 1], color='green', label='Start', s=1)
    # plt.scatter(pos[-1, 0], pos[-1, 1], color='red', label='End', s=1)
    # plt.xlabel('X Position')
    # plt.ylabel('Y Position')
    # plt.title('Sampled Trajectory')
    # plt.legend()
    # plt.grid()
    # plt.show()
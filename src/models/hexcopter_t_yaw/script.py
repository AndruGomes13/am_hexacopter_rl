from models.hexcopter import get_allocation_matrix, get_mass, get_max_propeller_thurst
import numpy as np
import plotly.graph_objects as go

def generate_cone_vectors(num_theta=20, num_phi=20, cone_half_angle_deg=30):
    # Convert cone angle to radians
    cone_half_angle_rad = np.radians(cone_half_angle_deg)

    # Theta: azimuthal angle in [0, 2pi]
    theta = np.linspace(0, 2 * np.pi, num_theta, endpoint=False)

    # Phi: inclination from z-axis, in [0, cone_half_angle]
    phi = np.linspace(0, cone_half_angle_rad, num_phi)

    # Create grid of spherical coordinates
    theta_grid, phi_grid = np.meshgrid(theta, phi)

    # Convert to Cartesian coordinates on the unit sphere
    x = np.sin(phi_grid) * np.cos(theta_grid)
    y = np.sin(phi_grid) * np.sin(theta_grid)
    z = np.cos(phi_grid)

    # Stack into (N, 3) vectors
    vectors = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    return vectors

def angle_from_vertical_deg(vecs):
    unit_z = np.array([0, 0, 1])
    norms = np.linalg.norm(vecs[:, :3], axis=1)
    dots = vecs[:, :3] @ unit_z
    cos_theta = dots / norms
    angles_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return np.degrees(angles_rad)

if __name__ == "__main__":
    alloc = get_allocation_matrix()
    alloc_inv = np.linalg.inv(alloc)

    vectors = generate_cone_vectors(cone_half_angle_deg=30, num_theta=100, num_phi=100) 
    vectors = (np.concatenate([vectors, np.zeros_like(vectors)], axis = 1) * get_mass() * 9.8)
    prop_cmd = alloc_inv@vectors.T

    prop_cmd_bool = (prop_cmd > 0) & (prop_cmd < get_max_propeller_thurst())
    is_green = np.all(prop_cmd_bool, axis=0)

    angles_deg = angle_from_vertical_deg(vectors)
    green_angles = angles_deg[is_green]
    red_angles = angles_deg[~is_green]


 

    # Separate green and red vectors
    green = vectors[is_green]
    red = vectors[~is_green]

    fig = go.Figure()


    fig.add_trace(go.Scatter3d(
        x=green[:, 0], y=green[:, 1], z=green[:, 2],
        mode='markers',
        marker=dict(size=3, color='green'),
        name='Feasible',
        text=[f"{a:.1f}° from vertical" for a in green_angles],
        hoverinfo='text'
    ))

    fig.add_trace(go.Scatter3d(
        x=red[:, 0], y=red[:, 1], z=red[:, 2],
        mode='markers',
        marker=dict(size=3, color='red'),
        name='Infeasible',
        text=[f"{a:.1f}° from vertical" for a in red_angles],
        hoverinfo='text'
    ))

    fig.update_layout(
        title="Feasible Thrust Directions (Hover Shows Angle)",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data"
        ),
        showlegend=True
    )
    fig.show(renderer='browser')
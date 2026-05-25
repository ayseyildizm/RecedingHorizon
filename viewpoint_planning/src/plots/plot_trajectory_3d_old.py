"""
3D Trajectory Plot — Robot path with target and occluders.

Shows the actual viewpoints the robot visited in 3D space, plus the target
and occluding objects. Useful for the Methods section of the thesis to
illustrate how the planner navigates around occlusions.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


COLORS = {
    'RH-NBV':   '#2ca02c',
    'Gradient': '#1f77b4',
    'PSO':      '#ff7f0e',
    'Random':   '#d62728',
}


# Occluder positions per scenario (from viewpoint_planning.py spawn_* methods)
OCCLUDERS = {
    'complex': [
        ([0.73, -0.25, 0.95], 'box'),
        ([0.50, -0.22, 1.00], 'bar'),
        ([0.60, -0.32, 1.30], 'box'),
    ],
    'easy':    [([0.65, -0.30, 1.10], 'box')],
    'hard':    [([0.60, -0.25, 1.10], 'box')],
    'bottom':  [([0.60, -0.25, 1.00], 'box')],   # adjust if different
    'none':    [],
}


def _add_box(ax, center, size=(0.05, 0.05, 0.05), color='gray', alpha=0.4):
    """Draw a small wireframe box at center."""
    cx, cy, cz = center
    sx, sy, sz = size
    # 8 corners
    corners = np.array([
        [cx-sx, cy-sy, cz-sz], [cx+sx, cy-sy, cz-sz],
        [cx+sx, cy+sy, cz-sz], [cx-sx, cy+sy, cz-sz],
        [cx-sx, cy-sy, cz+sz], [cx+sx, cy-sy, cz+sz],
        [cx+sx, cy+sy, cz+sz], [cx-sx, cy+sy, cz+sz],
    ])
    faces = [
        [corners[0], corners[1], corners[2], corners[3]],
        [corners[4], corners[5], corners[6], corners[7]],
        [corners[0], corners[1], corners[5], corners[4]],
        [corners[2], corners[3], corners[7], corners[6]],
        [corners[1], corners[2], corners[6], corners[5]],
        [corners[0], corners[3], corners[7], corners[4]],
    ]
    ax.add_collection3d(Poly3DCollection(
        faces, facecolors=color, edgecolors='black',
        linewidths=0.5, alpha=alpha,
    ))


def plot_3d_trajectory(
    trails,
    target_position,
    occlusion_type='complex',
    save_path=None,
    title='3D Robot Trajectory',
):
    """
    Plot 3D robot trajectories with target and occluders.

    Args:
        trails: dict of {method_name: list_of_xyz_positions}
                e.g. {'RH-NBV': vp.trail_rh, 'PSO': vp.trail_pso}
        target_position: [x, y, z] of target (e.g. vp.target_position)
        occlusion_type: 'complex', 'easy', 'hard', 'bottom', 'none'
        save_path: optional PNG path
        title: figure title
    """
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each method's trajectory
    for method, trail in trails.items():
        trail = np.asarray(trail)
        if trail.ndim != 2 or len(trail) < 2:
            continue

        color = COLORS.get(method, 'black')

        # Path line
        ax.plot(
            trail[:, 0], trail[:, 1], trail[:, 2],
            color=color, linewidth=2, alpha=0.7,
            label=f'{method} ({len(trail)} steps)',
        )

        # Waypoints
        ax.scatter(
            trail[:, 0], trail[:, 1], trail[:, 2],
            color=color, s=40, alpha=0.9, edgecolors='black', linewidths=0.5,
        )

        # Start (square) and end (triangle)
        ax.scatter(*trail[0],  color=color, s=200, marker='s',
                   edgecolors='black', linewidths=2, zorder=5,
                   label=f'{method} start')
        ax.scatter(*trail[-1], color=color, s=200, marker='^',
                   edgecolors='black', linewidths=2, zorder=5,
                   label=f'{method} end')

    # Target (red star)
    target_position = np.asarray(target_position)
    ax.scatter(
        *target_position, color='red', s=400, marker='*',
        edgecolors='darkred', linewidths=2, zorder=6, label='Target',
    )

    # Occluders
    for occ_pos, occ_type in OCCLUDERS.get(occlusion_type, []):
        if occ_type == 'box':
            _add_box(ax, occ_pos, size=(0.04, 0.04, 0.04), color='gray')
        else:  # bar
            _add_box(ax, occ_pos, size=(0.02, 0.02, 0.10), color='gray')

    ax.set_xlabel('X (m)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=11, fontweight='bold')
    ax.set_zlabel('Z (m)', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Equal-ish aspect
    ax.view_init(elev=20, azim=-60)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig, ax

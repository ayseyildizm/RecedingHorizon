"""
Reconstruction comparison plot.
Left: RH-NBV reconstructed target voxels (blue).
Right: Ground truth mesh (red).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_reconstruction_comparison(
    target_voxels,
    mesh_coordinates,
    save_path=None,
    method_label='RH-NBV',
    elev=20,
    azim=-60,
):
    """
    Side-by-side 3D scatter: reconstructed voxels vs ground truth mesh.

    Args:
        target_voxels    : (N, 3) numpy array — vp.rh_planner.target_voxels
        mesh_coordinates : (M, 3) numpy array — vp.rh_planner.mesh_coordinates
        save_path        : PNG output path (optional)
        method_label     : label shown in subplot title
        elev, azim       : 3D view angle
    """
    # Guard: skip if no voxels detected yet
    if (target_voxels is None
            or not isinstance(target_voxels, np.ndarray)
            or target_voxels.ndim < 2
            or len(target_voxels) == 0):
        print("Reconstruction plot skipped: no target voxels detected.")
        return None, None

    mesh_arr   = np.asarray(mesh_coordinates)
    voxel_arr  = np.asarray(target_voxels)

    fig = plt.figure(figsize=(14, 6))
    fig.suptitle("RH-NBV Reconstruction vs. Ground Truth",
                 fontsize=13, fontweight='bold', y=1.01)

    # ── Left: reconstructed voxels ───────────────────────────────────
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.view_init(elev=elev, azim=azim)
    ax1.scatter(
        voxel_arr[:, 0], voxel_arr[:, 1], voxel_arr[:, 2],
        c='blue', s=1, alpha=0.6, rasterized=True,
    )
    ax1.set_title(
        f"Target Voxels Point Cloud\n({method_label}: {len(voxel_arr):,} voxels)",
        fontsize=11, fontweight='bold',
    )
    ax1.set_xlabel('X (m)', fontsize=9)
    ax1.set_ylabel('Y (m)', fontsize=9)
    ax1.set_zlabel('Z (m)', fontsize=9)

    # ── Right: ground truth mesh ─────────────────────────────────────
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.view_init(elev=elev, azim=azim)
    ax2.scatter(
        mesh_arr[:, 0], mesh_arr[:, 1], mesh_arr[:, 2],
        c='red', s=1, alpha=0.6, rasterized=True,
    )
    ax2.set_title(
        f"Mesh Coordinates Point Cloud\n(Ground truth: {len(mesh_arr):,} points)",
        fontsize=11, fontweight='bold',
    )
    ax2.set_xlabel('X (m)', fontsize=9)
    ax2.set_ylabel('Y (m)', fontsize=9)
    ax2.set_zlabel('Z (m)', fontsize=9)

    # Equal aspect on both axes using mesh bounds
    all_pts = mesh_arr
    mid     = all_pts.mean(axis=0)
    span    = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2
    span    = max(span, 0.10)

    for ax in [ax1, ax2]:
        ax.set_xlim(mid[0] - span, mid[0] + span)
        ax.set_ylim(mid[1] - span, mid[1] + span)
        ax.set_zlim(mid[2] - span, mid[2] + span)
        ax.set_box_aspect([1, 1, 1])

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close(fig)

    return fig, (ax1, ax2)

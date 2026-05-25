"""
rh_planner_box.py
-----------------
Thin subclass of RHPlanner with calculate_F1() corrected for the simple
box target used in sanity-check experiments.

WHY A SEPARATE FILE?
--------------------
rh_planner.py hardcodes  `half = 0.002`  (2 mm) inside calculate_F1().
This works fine for the Stanford Bunny because that mesh has thousands of
vertices packed tightly together (~2 mm apart after scaling).

For the box sanity-check target we use a coarser 10×10 grid per face:
    mesh spacing = BOX_SIZE / (N - 1) = 0.15 / 9 ≈ 16.7 mm

With half=0.002 a voxel must land within 2 mm of a mesh point to count
as a match — almost impossible at 16.7 mm spacing, so F1 stays near zero
even when the box is fully reconstructed.

FIX: override calculate_F1() to use
    half = BOX_MESH_HALF  (default 10 mm, slightly tighter than spacing)

Everything else in RHPlanner is unchanged.

USAGE in viewpoint_planning_box.py
-----------------------------------
    from viewpoint_planners.rh_planner_box import RHPlannerBox
    ...
    self.rh_planner = RHPlannerBox(
        ...
        voxel_size=np.array([0.010]),   # match BOX_MESH_HALF
        ...
    )
"""

import numpy as np
from scipy.spatial import KDTree

from viewpoint_planners.rh_planner import RHPlanner


# Half-tolerance for F1 matching.
# Must satisfy:  BOX_MESH_HALF  <  mesh_spacing / 2  ×  (1 + margin)
# mesh_spacing ≈ 0.15 / 9 = 0.01667 m  →  half < 0.0083 m × 1.2 ≈ 0.010 m
BOX_MESH_HALF = 0.010   # 10 mm


class RHPlannerBox(RHPlanner):
    """
    RHPlanner subclass for the box sanity-check experiment.

    Only calculate_F1() is overridden — the matching tolerance `half` is
    set to BOX_MESH_HALF (10 mm) instead of the hardcoded 2 mm in the
    parent class.
    """

    def calculate_F1(self, occluder_positions=None):
        """
        Same logic as RHPlanner.calculate_F1() but with
        half = BOX_MESH_HALF instead of 0.002.

        occluder_positions: list of (center, half_size) tuples — voxels
        inside known occluders are excluded before F1 computation.
        """
        voxel_points, _, _ = self.get_occupied_points()
        if len(voxel_points) == 0:
            self.target_voxels = np.zeros((0, 3))
            return 0.0, 0.0, 0.0

        # Mask out known occluder voxels (they are false positives)
        if occluder_positions:
            keep = np.ones(len(voxel_points), dtype=bool)
            for center, half_size in occluder_positions:
                c = np.array(center)
                h = np.array(half_size)
                in_occ = np.all(np.abs(voxel_points - c) <= h, axis=1)
                keep &= ~in_occ
            voxel_points = voxel_points[keep]

        if len(voxel_points) == 0:
            self.target_voxels = np.zeros((0, 3))
            return 0.0, 0.0, 0.0

        self.target_voxels = voxel_points

        mesh_tree  = KDTree(self.mesh_coordinates)
        voxel_tree = KDTree(voxel_points)

        half   = BOX_MESH_HALF                   # ← key fix (was 0.002)
        radius = half * np.sqrt(3)

        # Precision: how many reconstructed voxels match a mesh point?
        nr_correct = 0
        for voxel in voxel_points:
            for idx in mesh_tree.query_ball_point(voxel, r=radius):
                coord = self.mesh_coordinates[idx]
                if all(abs(voxel[d] - coord[d]) <= half for d in range(3)):
                    nr_correct += 1
                    break

        # Recall: how many mesh points are covered by a voxel?
        nr_recalled = 0
        for coord in self.mesh_coordinates:
            for idx in voxel_tree.query_ball_point(coord, r=radius):
                voxel = voxel_points[idx]
                if all(abs(voxel[d] - coord[d]) <= half for d in range(3)):
                    nr_recalled += 1
                    break

        precision = nr_correct  / len(voxel_points)
        recall    = nr_recalled / len(self.mesh_coordinates)
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0 else 0.0
        )
        return f1, recall, precision

    def set_occluded_mesh_points(self):
        """Same as parent but uses BOX_MESH_HALF tolerance (was 0.002)."""
        voxel_points, _, _ = self.get_occupied_points()
        half   = BOX_MESH_HALF
        radius = half * np.sqrt(3)
        if len(voxel_points) == 0:
            self.occluded_mesh_points = self.mesh_coordinates.copy()
            return
        voxel_tree = KDTree(voxel_points)
        unseen = []
        for coord in self.mesh_coordinates:
            idxs    = voxel_tree.query_ball_point(coord, r=radius)
            covered = any(
                all(abs(voxel_points[i][d] - coord[d]) <= half for d in range(3))
                for i in idxs
            )
            if not covered:
                unseen.append(coord)
        self.occluded_mesh_points = np.array(unseen) if unseen else np.zeros((0, 3))
        print(
            f"[RHPlannerBox] Occluded after view 0: "
            f"{len(self.occluded_mesh_points)}/{len(self.mesh_coordinates)} "
            f"({100*len(self.occluded_mesh_points)/len(self.mesh_coordinates):.1f}%)"
        )

    def compute_occluded_recall(self) -> float:
        """Same as parent but uses BOX_MESH_HALF tolerance (was 0.002)."""
        if self.occluded_mesh_points is None or len(self.occluded_mesh_points) == 0:
            return 0.0
        voxel_points, _, _ = self.get_occupied_points()
        if len(voxel_points) == 0:
            return 0.0
        voxel_tree = KDTree(voxel_points)
        half       = BOX_MESH_HALF
        radius     = half * np.sqrt(3)
        recovered  = 0
        for coord in self.occluded_mesh_points:
            idxs = voxel_tree.query_ball_point(coord, r=radius)
            if any(
                all(abs(voxel_points[i][d] - coord[d]) <= half for d in range(3))
                for i in idxs
            ):
                recovered += 1
        return recovered / len(self.occluded_mesh_points)

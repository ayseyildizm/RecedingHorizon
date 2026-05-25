import math
import torch
import time

import numpy as np
import rospy
from scipy.spatial import KDTree

from abb_control.arm_control_client import ArmControlClient
from perception.perceiver import Perceiver
from viewpoint_planners.rh_planner_box import RHPlannerBox
from viewpoint_planners.viewpoint_sampler import ViewpointSampler
from utils.py_utils import numpy_to_pose
from utils.sdf_spawner import SDFSpawner


# ---------------------------------------------------------------------------
# Target object geometry
# ---------------------------------------------------------------------------
# A simple 15 cm cube placed at the same position as the bunny was.
# Adjust BOX_SIZE if you use a different object in Gazebo.
#
# In your Gazebo world file, spawn a red box like this:
#
#   <model name="target_box">
#     <pose>0.5 -0.4 1.0 0 0 0</pose>
#     <static>0</static>
#     <link name="link">
#       <gravity>0</gravity>
#       <collision name="collision">
#         <geometry><box><size>0.15 0.15 0.15</size></box></geometry>
#       </collision>
#       <visual name="visual">
#         <geometry><box><size>0.15 0.15 0.15</size></box></geometry>
#         <material>
#           <ambient>0.8 0.1 0.1 1</ambient>   <!-- red for color segmentation -->
#           <diffuse>0.8 0.1 0.1 1</diffuse>
#         </material>
#       </visual>
#     </link>
#   </model>
#
# The color segmentation in perceiver.py already looks for red (HSV 0-10),
# so no changes are needed there.
# ---------------------------------------------------------------------------
BOX_CENTER = np.array([0.5, -0.10, 1.075])   # base at z=1.0, half-size=0.075
BOX_SIZE   = np.array([0.15, 0.15, 0.15])   # [x, y, z] in metres
SURFACE_SAMPLES_PER_FACE = 10               # n×n grid per face  (6×100 = 600 pts)


class ViewpointPlanningBox:
    """
    Viewpoint planning pipeline — identical to ViewpointPlanning but uses a
    simple box as the target object instead of the Stanford Bunny mesh.

    Differences from viewpoint_planning.py
    ---------------------------------------
    * get_mesh_coordinates() generates box surface points analytically —
      no .dae file needed, no XML parsing.
    * grid_size is smaller (box is ~15 cm, bunny was ~30 cm).
    * r_max is tightened slightly to keep the camera close to the small target.
    * target_position / BOX_CENTER reflect the box pose (z centre = 1.075).
    * Class is renamed ViewpointPlanningBox so both files can be imported
      simultaneously without conflict.
    """

    def __init__(self, lr=None):
        # ROS
        self.arm_control = ArmControlClient()
        self.perceiver = Perceiver()
        self.viewpoint_sampler = ViewpointSampler()
        self.sdf_spawner = SDFSpawner()

        self.lr = lr

        # Box centre — note z=1.075 (bottom at 1.0, top at 1.15)
        self.target_position = BOX_CENTER.copy()

        self.config()
        self.mesh_coordinates, self.mesh_tree = self.get_mesh_coordinates()

        start_time = time.time()
        self.rh_planner = RHPlannerBox(
            grid_size=self.grid_size,
            grid_center=self.grid_center,
            image_size=self.image_size,
            intrinsics=self.intrinsics,
            start_pose=self.camera_pose,
            target_params=self.target_position,
            num_samples=1,
            mesh_coordinates=self.mesh_coordinates,
            mesh_tree=self.mesh_tree,
            horizon=3,
            num_candidates=10,
            lambda_cost=2.0,
            step_size=0.065,
            bias_ratio=0.7,
            discount=0.85,
            r_min=0.15,
            r_max=0.35,
            occlusion_bonus=2.0,
            stagnation_patience=4,
            voxel_size=np.array([0.010]),   # 10 mm — matches box mesh spacing
            robot_reach_bounds=np.array([[0.30, -0.25, 0.97], [0.65, 0.40, 1.25]]),
        )
        init_time_rh = time.time() - start_time

        # Metric arrays — same structure as original
        self.losses_rh              = np.array([0.0])
        self.cumulative_time_rh     = np.array([init_time_rh])
        self.coverages_rh           = np.array([0.0])
        self.trail_rh               = [self.camera_pose[:3].copy()]
        self.trajectory_distance_rh = np.array([0.0])
        self.recall_rh              = np.array([0.0])
        self.precision_rh           = np.array([0.0])
        self.f1_rh                  = np.array([0.0])
        self.ray_calls_rh           = np.array([0])
        self.sigma_rh               = np.array([0.0])
        self.occluded_recall_rh     = np.array([0.0])

        print(
            f"[RH-Box] K={self.rh_planner.num_candidates}, "
            f"H={self.rh_planner.horizon} -> "
            f"evals/iter={self.rh_planner.num_candidates * self.rh_planner.horizon}"
        )
        print(
            f"[RH-Box] Target: box {BOX_SIZE*100} cm  "
            f"centre={BOX_CENTER}  "
            f"mesh_pts={len(self.mesh_coordinates)}"
        )

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    def config(self):
        """Configure scene, robot start pose, voxel grid, and camera intrinsics."""
        # Uncomment exactly one scenario:
        self.spawn_no_occlusion()
        #self.spawn_wall_occlusion()    # sanity check — back face hidden, max 83.3%
        # self.spawn_easy_occlusion()
        # self.spawn_hard_occlusion()
        # self.spawn_extreme_occlusion()
        # self.spawn_complex_occlusion()

        self.camera_pose = self.viewpoint_sampler.predefine_start_pose(
            self.target_position
        )
        if self.arm_control:
            self.arm_control.move_arm_to_pose(numpy_to_pose(self.camera_pose))

        # Grid is tighter than bunny — box is ~15 cm, keep some margin
        self.grid_size   = np.array([0.25, 0.25, 0.25])
        self.grid_center = self.target_position

        camera_info      = self.perceiver.get_camera_info()
        self.image_size  = np.array([camera_info.width, camera_info.height])
        self.intrinsics  = np.array(camera_info.K).reshape(3, 3)

    # ------------------------------------------------------------------
    # Occlusion scenarios
    # (positions re-calibrated for the box target at [0.5, -0.4, 1.075])
    # ------------------------------------------------------------------
    def spawn_no_occlusion(self):
        """Control case: no occluding object spawned."""
        pass

    def spawn_wall_occlusion(self):
        """
        Sanity-check occlusion: thin wall (box.sdf = 0.3×0.03×0.3 m) flush
        against the back face of the target box (y+ side).

        Geometry
        --------
        box.sdf size         : 0.3 (x) × 0.03 (y) × 0.3 (z)
        box.sdf half-depth   : 0.015 m

        Target box centre    : x=0.5   y=-0.400  z=1.075
        Target back face     : y = -0.400 + 0.075 = -0.325
        Wall centre          : y = -0.325 + 0.015 = -0.310

        spawn_box() subtracts 0.024 from z internally (Gazebo offset),
        so we pass z = 1.075 + 0.024 = 1.099 to compensate.

        Expected result
        ---------------
        Back face (1/6) permanently hidden  →  max observable ≈ 83.3 %
        Coverage > 83 % → mesh/voxel misalignment bug
        Coverage < 60 % → planner not exploring enough
        """
        wall_center = np.array([0.5, -0.310, 1.075 + 0.024])
        self.sdf_spawner.spawn_box(wall_center, 99)

    def spawn_easy_occlusion(self):
        """Single box, offset slightly to the side of the target."""
        self.sdf_spawner.spawn_box(np.array([0.65, -0.30, 1.075]), 1)

    def spawn_hard_occlusion(self):
        """Single box, closer and more centred in front of the target."""
        self.sdf_spawner.spawn_box(np.array([0.60, -0.25, 1.075]), 1)

    def spawn_extreme_occlusion(self):
        """Two stacked boxes aligned in front of the target."""
        self.sdf_spawner.spawn_box(np.array([0.60, -0.30, 1.075]), 1)
        self.sdf_spawner.spawn_box(np.array([0.60, -0.30, 1.175]), 2)

    def spawn_complex_occlusion(self):
        """Three-object occlusion setup (same layout as bunny experiment)."""
        self.sdf_spawner.spawn_box(np.array([0.73, -0.25, 0.95]),  1)
        self.sdf_spawner.spawn_bar(np.array([0.50, -0.22, 1.00]),  2)
        self.sdf_spawner.spawn_box(np.array([0.60, -0.32, 1.30]),  3)

    # ------------------------------------------------------------------
    # RH execution — identical to original run_rh()
    # ------------------------------------------------------------------
    def run_rh(self):
        """Run one Receding Horizon NBV iteration and log metrics."""
        start_time = time.time()
        current_coverage = float(self.coverages_rh[-1]) if len(self.coverages_rh) > 0 else 0.0
        self.camera_pose, loss, n_evals = self.rh_planner.rh_view(
            current_coverage=current_coverage
        )

        self.camera_pose[:3] = self._clamp_to_rh_bounds(self.camera_pose[:3])
        self.trail_rh.append(self.camera_pose[:3].copy())
        self.losses_rh = np.append(self.losses_rh, loss)

        is_success = self.arm_control.move_arm_to_pose(numpy_to_pose(self.camera_pose))
        rospy.sleep(1.0)

        if is_success:
            depth_image, _, semantics = self.perceiver.run()
            coverage = self.rh_planner.update_voxel_grid(
                depth_image, semantics, self.camera_pose
            )
            if self.rh_planner.occluded_mesh_points is None:
                self.rh_planner.set_occluded_mesh_points()
            self.coverages_rh = np.append(self.coverages_rh, coverage)
            self.trajectory_distance_rh = np.append(
                self.trajectory_distance_rh,
                self.trajectory_distance_rh[-1] + self._last_step_distance(),
            )
        else:
            # Auto-shrink reach bounds on arm failure
            failed_pos = self.camera_pose[:3]
            if self.rh_planner.robot_reach_bounds is not None:
                bounds = self.rh_planner.robot_reach_bounds.cpu().numpy().copy()
                for dim in range(3):
                    if failed_pos[dim] < bounds[0][dim] + 0.02:
                        bounds[0][dim] = min(failed_pos[dim] + 0.03, bounds[1][dim] - 0.05)
                    if failed_pos[dim] > bounds[1][dim] - 0.02:
                        bounds[1][dim] = max(failed_pos[dim] - 0.03, bounds[0][dim] + 0.05)
                self.rh_planner.robot_reach_bounds = torch.tensor(
                    bounds, dtype=torch.float32, device=self.rh_planner.device
                )
                print(f"[Box-VP] Reach bounds tightened: y=[{bounds[0][1]:.3f},{bounds[1][1]:.3f}]")
            # Reset to last good position
            if len(self.trail_rh) >= 2:
                self.rh_planner.current_pos = torch.tensor(
                    self.trail_rh[-2], dtype=torch.float32,
                    device=self.rh_planner.device,
                )
            coverage = self.coverages_rh[-1]
            self.coverages_rh = np.append(self.coverages_rh, coverage)
            self.trajectory_distance_rh = np.append(
                self.trajectory_distance_rh, self.trajectory_distance_rh[-1]
            )

        iter_time = time.time() - start_time
        self.cumulative_time_rh = np.append(
            self.cumulative_time_rh, self.cumulative_time_rh[-1] + iter_time
        )

        f1, recall, precision = self.rh_planner.calculate_F1()
        self.f1_rh        = np.append(self.f1_rh, f1)
        self.recall_rh    = np.append(self.recall_rh, recall)
        self.precision_rh = np.append(self.precision_rh, precision)
        self.ray_calls_rh = np.append(self.ray_calls_rh, self.rh_planner.ray_trace_count)

        sigma     = self.rh_planner.compute_sigma()
        self.sigma_rh = np.append(self.sigma_rh, sigma)

        occ_recall = self.rh_planner.compute_occluded_recall()
        self.occluded_recall_rh = np.append(self.occluded_recall_rh, occ_recall)

        total_occluded = (
            len(self.rh_planner.occluded_mesh_points)
            if self.rh_planner.occluded_mesh_points is not None
            else 0
        )
        recovered = int(occ_recall * total_occluded) if total_occluded > 0 else 0

        print(
            f"[RH-Box] coverage={coverage:.4f} | loss={loss:.4f} | "
            f"F1={f1:.4f} | recall={recall:.4f} | "
            f"precision={precision:.4f} | occ_recall={occ_recall:.4f} | evals={n_evals}"
        )
        print(
            f"         Occluded recall: {occ_recall*100:.1f}% "
            f"({recovered}/{total_occluded} mesh points recovered)"
        )

        return coverage, loss, f1, recall, precision, n_evals

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _clamp_to_rh_bounds(self, position):
        bounds = self.rh_planner.camera_bounds.detach().cpu().numpy()
        return np.clip(position, bounds[0], bounds[1])

    def _last_step_distance(self):
        p1 = self.trail_rh[-2]
        p2 = self.trail_rh[-1]
        return math.sqrt(sum((p2[i] - p1[i]) ** 2 for i in range(3)))

    def get_rh_metrics(self):
        """Return all RH metrics collected so far."""
        return {
            "losses":               self.losses_rh,
            "cumulative_time":      self.cumulative_time_rh,
            "coverages":            self.coverages_rh,
            "trajectory_distance":  self.trajectory_distance_rh,
            "f1":                   self.f1_rh,
            "recall":               self.recall_rh,
            "precision":            self.precision_rh,
            "ray_calls":            self.ray_calls_rh,
            "trail":                np.array(self.trail_rh),
            "sigma":                self.sigma_rh,
            "occluded_recall":      self.occluded_recall_rh,
        }

    # ------------------------------------------------------------------
    # Mesh coordinates — analytical box surface, no .dae file needed
    # ------------------------------------------------------------------
    def get_mesh_coordinates(self):
        """
        Generate ground-truth surface points for a box analytically.

        The box has dimensions BOX_SIZE centred at BOX_CENTER.
        Points are sampled on an n×n grid on each of the 6 faces,
        then deduplicated so edge/corner points are not double-counted.

        This replaces the COLLADA-parsing used for the bunny and is
        much easier to verify visually: you should see a clean cube
        in any 3D plot of mesh_coordinates.
        """
        cx, cy, cz = BOX_CENTER
        hx, hy, hz = BOX_SIZE / 2.0          # half-extents
        n = SURFACE_SAMPLES_PER_FACE

        lin_x = np.linspace(cx - hx, cx + hx, n)
        lin_y = np.linspace(cy - hy, cy + hy, n)
        lin_z = np.linspace(cz - hz, cz + hz, n)

        pts = []
        # +X / -X faces  (left / right)
        for y in lin_y:
            for z in lin_z:
                pts.append([cx + hx, y, z])
                pts.append([cx - hx, y, z])
        # +Y / -Y faces  (back / front)
        for x in lin_x:
            for z in lin_z:
                pts.append([x, cy + hy, z])   # back face ✓
                pts.append([x, cy - hy, z])   # front face ✓
        # +Z / -Z faces  (top / bottom)
        for x in lin_x:
            for y in lin_y:
                pts.append([x, y, cz + hz])
                pts.append([x, y, cz - hz])

        mesh_coordinates = np.unique(np.array(pts, dtype=np.float64), axis=0)
        mesh_tree = KDTree(mesh_coordinates)

        back_face_y = round(cy + hy, 4)
        print(
            f"[RH-Box] Box mesh: {len(mesh_coordinates)} surface points  "
            f"(6 faces, no occlusion)\n"
            f"         Theoretical max coverage = 100 %"
        )
        return mesh_coordinates, mesh_tree

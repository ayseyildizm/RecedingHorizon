"""
SamplingNBV baseline planner.

This is the "local sampling" baseline used in Burusa et al. (ICRA 2024) and
introduced in Burusa et al. (arXiv 2306.09801, IROS-W 2023). At each step it
samples K candidate viewpoints inside a local sphere around the current camera
pose, evaluates the Burusa semantic information gain at each, and picks the
candidate with the highest gain. It is a 1-step (myopic) planner --- the
primary baseline against which both GradientNBV and our tree-based RH-NBV
are compared.

Compatibility:
    Same public interface as RHPlanner / GradientNBVPlanner so viewpoint_planning.py
    can drive it identically. The "view" entry point is sampling_view() to
    mirror gradient's next_best_view() and RH's rh_view() naming.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import KDTree

from scene_representation.voxel_grid import VoxelGrid
from utils.rviz_visualizer import RvizVisualizer
from utils.py_utils import numpy_to_pose_array
from utils.torch_utils import look_at_rotation, transform_from_rotation_translation


class SamplingPlanner:
    """Local-sphere sampling NBV (Burusa et al. 2023)."""

    def __init__(
        self,
        start_pose: np.ndarray,
        mesh_coordinates: np.ndarray,
        mesh_tree,
        grid_size: np.ndarray = np.array([0.3, 0.3, 0.3]),
        voxel_size: np.ndarray = np.array([0.002]),
        grid_center: np.ndarray = np.array([0.5, -0.4, 1.1]),
        image_size: np.ndarray = np.array([600, 450]),
        intrinsics: np.ndarray = np.array([
            [685.5028076171875, 0.0, 485.35955810546875],
            [0.0, 685.6409912109375, 270.7330627441406],
            [0.0, 0.0, 1.0],
        ]),
        num_pts_per_ray: int = 128,
        num_features: int = 4,
        num_samples: int = 1,
        target_params: np.ndarray = np.array([0.5, -0.4, 1.1]),
        # SamplingNBV params (Burusa 2023 default: K=10, sphere radius=0.1)
        num_candidates: int = 10,
        sample_radius: float = 0.1,
        rng_seed: int = 42,
    ) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        grid_size_t = torch.tensor(grid_size, dtype=torch.float32, device=self.device)
        voxel_size_t = torch.tensor(voxel_size, dtype=torch.float32, device=self.device)
        grid_center_t = torch.tensor(grid_center, dtype=torch.float32, device=self.device)

        self.target_params = torch.tensor(
            target_params, dtype=torch.float32, device=self.device
        )

        self.camera_bounds = torch.tensor(
            [
                [start_pose[0] - 0.2, start_pose[1] - 0.1, start_pose[2] - 0.15],
                [start_pose[0] + 0.2, start_pose[1] + 0.1, start_pose[2] + 0.15],
            ],
            dtype=torch.float32,
            device=self.device,
        )

        self.voxel_grid = VoxelGrid(
            grid_size=grid_size_t,
            voxel_size=voxel_size_t,
            grid_center=grid_center_t,
            width=image_size[0],
            height=image_size[1],
            fx=intrinsics[0, 0],
            fy=intrinsics[1, 1],
            cx=intrinsics[0, 2],
            cy=intrinsics[1, 2],
            num_pts_per_ray=num_pts_per_ray,
            num_features=num_features,
            target_params=self.target_params,
            device=self.device,
        )

        self.num_samples = num_samples
        self.rviz_visualizer = RvizVisualizer()
        self.mesh_coordinates = mesh_coordinates
        self.mesh_tree = mesh_tree

        self.num_candidates = int(num_candidates)
        self.sample_radius = float(sample_radius)

        self.current_pos = torch.tensor(
            start_pose[:3], dtype=torch.float32, device=self.device
        )
        self.ray_trace_count = 0
        self.rng_seed = int(rng_seed)
        np.random.seed(self.rng_seed)
        torch.manual_seed(self.rng_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.rng_seed)

        self.target_voxels = np.array(0)

    # ------------------------------------------------------------------
    # Burusa semantic information gain (identical to RHPlanner.compute_gain)
    # ------------------------------------------------------------------

    def compute_gain(self, camera_pos: torch.Tensor) -> float:
        self.ray_trace_count += 1
        quat = look_at_rotation(camera_pos, self.target_params)
        transforms = transform_from_rotation_translation(
            quat[None, :], camera_pos[None, :]
        )
        t_vals = self.voxel_grid.t_vals.clone()
        ray_origins, ray_directions, _ = (
            self.voxel_grid.ray_sampler.ray_origins_directions(transforms=transforms)
        )
        ray_points = (
            ray_directions[:, :, None, :] * t_vals[None, :, None]
            + ray_origins[:, :, None, :]
        ).view(-1, 3)
        ray_points_nor = self.voxel_grid.normalize_3d_coordinate(ray_points)
        ray_points_nor = ray_points_nor.view(1, -1, 1, 1, 3).repeat(2, 1, 1, 1, 1)
        grid = self.voxel_grid.voxel_grid[None, ..., 1:3].permute(4, 0, 1, 2, 3)
        occ_sem = F.grid_sample(grid, ray_points_nor, align_corners=True)
        occ_sem = occ_sem.view(2, -1, self.voxel_grid.num_pts_per_ray)
        occ_sem = occ_sem.clamp(self.voxel_grid.eps, 1.0 - self.voxel_grid.eps)
        opacities = torch.sigmoid(1e7 * (occ_sem[0, ...] - 0.51))
        transmittance = self.voxel_grid.shifted_cumprod(1.0 - opacities)
        ray_gains = transmittance * self.voxel_grid.entropy(occ_sem[1, ...])
        return torch.log(torch.mean(ray_gains) + self.voxel_grid.eps).item()

    # ------------------------------------------------------------------
    # sampling_view: sample K, evaluate each, return the best
    # ------------------------------------------------------------------

    def sampling_view(self) -> tuple:
        self.ray_trace_count = 0
        best_pos = None
        best_gain = -np.inf

        for _ in range(self.num_candidates):
            # Uniform random direction, uniform radius in [0, sample_radius]
            direction = torch.randn(3, device=self.device)
            direction = direction / (torch.norm(direction) + 1e-6)
            radius = self.sample_radius * torch.rand(1, device=self.device).item()
            cand = self.current_pos + direction * radius
            cand = torch.clamp(cand, self.camera_bounds[0], self.camera_bounds[1])
            gain = self.compute_gain(cand)
            if gain > best_gain:
                best_gain = gain
                best_pos = cand

        self.current_pos = best_pos.clone()
        quat = look_at_rotation(best_pos, self.target_params)
        viewpoint = np.zeros(7)
        viewpoint[:3] = best_pos.detach().cpu().numpy()
        viewpoint[3:] = quat.detach().cpu().numpy()
        return viewpoint, -best_gain, self.ray_trace_count

    # ------------------------------------------------------------------
    # Standard interface (update_voxel_grid, visualize, calculate_F1)
    # ------------------------------------------------------------------

    def update_voxel_grid(
        self, depth_image: np.ndarray, semantics: torch.Tensor, viewpoint: np.ndarray
    ) -> float:
        depth_image = torch.tensor(depth_image, dtype=torch.float32, device=self.device)
        position = torch.tensor(viewpoint[:3], dtype=torch.float32, device=self.device)
        orientation = torch.tensor(viewpoint[3:], dtype=torch.float32, device=self.device)
        transform = transform_from_rotation_translation(
            orientation[None, :], position[None, :]
        )
        coverage = self.voxel_grid.insert_depth_and_semantics(
            depth_image, semantics, transform
        )
        if coverage is not None:
            coverage = coverage.cpu().numpy()
        self.current_pos = position.clone()
        return coverage

    def get_occupied_points(self):
        v, s, c = self.voxel_grid.get_occupied_points()
        return v.cpu().numpy(), s.cpu().numpy(), c.cpu().numpy()

    def visualize(self):
        voxels, sem_scores, sem_ids = self.get_occupied_points()
        self.rviz_visualizer.visualize_voxels(voxels, sem_scores, sem_ids)
        tgt = self.target_params.detach().cpu().numpy()
        rois = np.array([[*tgt, 1.0, 0.0, 0.0, 0.0]])
        self.rviz_visualizer.visualize_rois(numpy_to_pose_array(rois))
        self.rviz_visualizer.visualize_camera_bounds(self.camera_bounds.cpu().numpy())

    def calculate_F1(self):
        voxels, _, sem_ids = self.get_occupied_points()
        target_voxels = np.array(
            [v for i, v in enumerate(voxels) if sem_ids[i] == 0]
        )
        self.target_voxels = target_voxels
        if len(target_voxels) == 0:
            return 0, 0, 0
        mesh_tree = KDTree(self.mesh_coordinates)
        voxel_tree = KDTree(target_voxels)
        half = 0.002
        search_r = np.sqrt(3 * (half ** 2))
        n_correct = 0
        for v in target_voxels:
            for idx in mesh_tree.query_ball_point(v, r=search_r):
                c = self.mesh_coordinates[idx]
                if (
                    abs(v[0] - c[0]) <= half
                    and abs(v[1] - c[1]) <= half
                    and abs(v[2] - c[2]) <= half
                ):
                    n_correct += 1
                    break
        n_recalled = 0
        for c in self.mesh_coordinates:
            for idx in voxel_tree.query_ball_point(c, r=search_r):
                v = target_voxels[idx]
                if (
                    abs(v[0] - c[0]) <= half
                    and abs(v[1] - c[1]) <= half
                    and abs(v[2] - c[2]) <= half
                ):
                    n_recalled += 1
                    break
        precision = n_correct / len(target_voxels)
        recall = n_recalled / len(self.mesh_coordinates)
        F1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return F1, recall, precision

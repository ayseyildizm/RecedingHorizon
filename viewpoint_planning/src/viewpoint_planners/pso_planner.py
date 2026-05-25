import torch
import torch.nn as nn
import numpy as np

from scene_representation.voxel_grid import VoxelGrid
from viewpoint_planners.viewpoint_sampler import ViewpointSampler

from utils.rviz_visualizer import RvizVisualizer
from utils.py_utils import numpy_to_pose, numpy_to_pose_array
from utils.torch_utils import look_at_rotation, transform_from_rotation_translation

import time
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def _append_obj_history(obj_history_i, score, position_tensor):
    """Replace oldest entry in obj_history row with new (score, position) pair."""
    new_entry = np.empty((1, 2), dtype=object)
    new_entry[0, 0] = float(score)
    new_entry[0, 1] = position_tensor.detach().cpu().numpy()
    return np.concatenate([obj_history_i[1:], new_entry], axis=0)


class PsoPlanner:
    """
    Class to plan viewpoints using particle swarm optimization
    """

    def __init__(
        self,
        start_pose: np.array,
        mesh_coordinates: np.array,
        mesh_tree,
        grid_size: np.array = np.array([0.3, 0.3, 0.3]),
        voxel_size: np.array = np.array([0.002]),
        grid_center: np.array = np.array([0.5, -0.4, 1.1]),
        image_size: np.array = np.array([600, 450]),
        intrinsics: np.array = np.array(
            [
                [685.5028076171875, 0.0, 485.35955810546875],
                [0.0, 685.6409912109375, 270.7330627441406],
                [0.0, 0.0, 1.0],
            ],
        ),
        num_pts_per_ray: int = 128,
        num_features: int = 4,
        num_samples: int = 1,
        target_params: np.array = np.array([0.5, -0.4, 1.1]),

        c1: float = 1.25,
        c2: float = 0.5,
        w: float = 1.5,
        bc: float = 0.75,
        n_particles: int = 4,
        

    ) -> None:
        """
        Initialize the planner
        :param grid_size: size of the voxel grid in meters
        :param voxel_size: size of the voxels in meters
        :param grid_center: center of the voxel grid in meters
        :param image_size: size of the image in pixels
        :param num_pts_per_ray: number of points sampled per ray
        :param num_features: number of features per voxel
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        grid_size = torch.tensor(grid_size, dtype=torch.float32, device=self.device)
        voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=self.device)
        grid_center = torch.tensor(grid_center, dtype=torch.float32, device=self.device)
        self.optimization_params(start_pose, target_params)
        self.voxel_grid = VoxelGrid(
            grid_size=grid_size,
            voxel_size=voxel_size,
            grid_center=grid_center,
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


        # Hyperparameters of the algorithm
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.bc = bc #the bouncing coefficient
        
        self.n_particles = n_particles
        np.random.seed(int(time.time()))

        # Generate random numbers for `self.n_particles` particles, each with 3 features (x,y,z coordinates --> rotation always computed with look_at_rotation function)
        random_values = np.random.rand(self.n_particles, 3)  # Random values between 0 and 1

        # Scale each column based on camera bounds
        random_values[:, 0] = random_values[:, 0] * 0.4 + start_pose[0] - 0.2  
        random_values[:, 1] = random_values[:, 1] * 0.2 + start_pose[1] - 0.1 
        random_values[:, 2] = random_values[:, 2] * 0.3 + start_pose[2] - 0.15  

        # Convert to PyTorch tensor
        self.X = torch.tensor(
            random_values,
            dtype=torch.float32,
            device=self.device,
        )

        velocities = np.random.rand(self.n_particles, 3)
        for i in range(self.n_particles):  #Initializing the particle velocities in the direction of the target, wrt to different camera reach in each direction

            if self.X[i][0] < self.target_params[0]:
                velocities[i][0] = velocities[i][0] * 0.12
            else:
                velocities[i][0] = velocities[i][0] * -0.12

            if self.X[i][1] < self.target_params[1]:
                velocities[i][1] = velocities[i][1] * 0.06
            else:
                velocities[i][1] = velocities[i][1] * -0.06
            
            if self.X[i][2] < self.target_params[2]:
                velocities[i][2] = velocities[i][2] * 0.09
            else:
                velocities[i][2] = velocities[i][2] * -0.09


        self.V = torch.tensor(
            velocities,
            dtype=torch.float32,
            device=self.device,
        )
        
        self.init = True

        self.pbest = self.X
        self.pbest_obj = np.zeros(self.n_particles)

        self.recall = 3  # for the idea of having pbest depend on only last n iterations
        self.obj_history = np.empty((n_particles, self.recall, 2), dtype=object)
        for i in range(n_particles):
            for j in range(self.recall):
                self.obj_history[i, j, 0] = np.inf
                self.obj_history[i, j, 1] = np.zeros(3)

        i = 0
        for x in self.X:
            self.pbest_obj[i] = self.voxel_grid.compute_gain(x, self.target_params)[0]
            i += 1

        for i in range(self.n_particles):
            self.obj_history[i] = _append_obj_history(self.obj_history[i], self.pbest_obj[i], self.X[i])

        self.pbest = torch.tensor(
            np.array([min(particle, key=lambda x: x[0])[1] for particle in self.obj_history]),
            dtype=torch.float32,
            device=self.device,
        )
        
        self.pbest_obj = np.array([min(particle, key=lambda x: x[0])[0] for particle in self.obj_history])

        self.gbest = self.pbest[self.pbest_obj.argmin()]
        self.gbest_obj = self.pbest_obj.min()

        self.particle_trajectories = [self.X.detach().cpu().numpy()] #only for visualizing the particle trajectories

        self.target_voxels = np.array(0)

    def optimization_params(
        self, start_pose: np.array, target_params: np.array
    ) -> None:
        """
        Initialize the optimization parameters
        """
        self.target_params = torch.tensor(
            target_params,
            dtype=torch.float32,
            device=self.device,
        )
        self.camera_bounds = torch.tensor(
            [
                [
                    start_pose[0] - 0.2,
                    start_pose[1] - 0.1,
                    start_pose[2] - 0.15,
                    target_params[0] - 0.1,
                    target_params[1] - 0.1,
                    target_params[2] - 0.1,
                ],
                [
                    start_pose[0] + 0.2,
                    start_pose[1] + 0.1,
                    start_pose[2] + 0.15,
                    target_params[0] + 0.1,
                    target_params[1] + 0.1,
                    target_params[2] + 0.1,
                ],
            ],
            dtype=torch.float32,
            device=self.device,
        )

    def update_voxel_grid(
        self, depth_image: np.array, semantics: torch.tensor, viewpoint: np.array
    ) -> None:
        """
        Process depth and semantic images and insert them into the voxel grid
        :param depth_image: depth image (H, W)
        :param semantics: confidence scores and class ids (H, W, 2)
        :param viewpoint: camera position (xyz) and orientation (wxyz) w.r.t the 'world_frame'
        """
        depth_image = torch.tensor(depth_image, dtype=torch.float32, device=self.device)
        position = torch.tensor(viewpoint[:3], dtype=torch.float32, device=self.device)
        orientation = torch.tensor(
            viewpoint[3:], dtype=torch.float32, device=self.device
        )
        transform = transform_from_rotation_translation(
            orientation[None, :], position[None, :]
        )
        coverage = self.voxel_grid.insert_depth_and_semantics(
            depth_image, semantics, transform
        )

        if coverage is not None:
            coverage = coverage.cpu().numpy()
        return coverage


    def update(self) -> np.array:
        "Function to do one iteration of particle swarm optimization"

        r1, r2 = np.random.rand(2, self.n_particles) + 0.25

        r1 = torch.tensor(r1.reshape(self.n_particles,1), dtype=torch.float32, device=self.device,)

        r2 = torch.tensor(r2.reshape(self.n_particles,1), dtype=torch.float32, device=self.device,)

        self.V = self.w * self.V + self.c1*r1*(self.pbest - self.X) + self.c2*r2*(self.gbest.repeat(self.n_particles,1)- self.X)
        self.X = self.X + self.V


        for i in range(self.n_particles):     
            for j in range(3):
                if self.X[i][j] >= self.camera_bounds[1][j]: #exceeded upper bound
                    bouncing_force = self.camera_bounds[1][j] - self.X[i][j]
                    self.X[i][j] = self.camera_bounds[1][j] #ensure that particles do not exceed the camera bounds
                    self.V[i][j] = bouncing_force * self.bc #adapt the velocity to have the particle bounce back when it hits the bound, to encourage exploration
                if self.X[i][j] <= self.camera_bounds[0][j]: #exceeded lower bound
                    bouncing_force = self.camera_bounds[0][j] - self.X[i][j]
                    self.X[i][j] = self.camera_bounds[0][j] #ensure that particles do not exceed the camera bounds
                    self.V[i][j] = bouncing_force * self.bc #adapt the velocity to have the particle bounce back when it hits the bound, to encourage exploration
      
       
        #for printing particle trajectories
        current_particles = self.X
        self.particle_trajectories.append(current_particles.detach().cpu().numpy())

        obj = np.zeros(self.n_particles)
        i = 0
        for x in self.X:
            obj[i] = self.voxel_grid.compute_gain(x, self.target_params)[0]
            i += 1

        # Update the record of previous positions and their utilities
        for i in range(self.n_particles):
            self.obj_history[i] = _append_obj_history(self.obj_history[i], obj[i], self.X[i])

        #update local and global best positions and utilities
        self.pbest = torch.tensor(
            np.array([min(particle, key=lambda x: x[0])[1] for particle in self.obj_history]),
            dtype=torch.float32,
            device=self.device,
        )
        self.pbest_obj = np.array([min(particle, key=lambda x: x[0])[0] for particle in self.obj_history])

        self.gbest = self.pbest[self.pbest_obj.argmin()]
        self.gbest_obj = self.pbest_obj.min()

        return obj

    
    def pso_view(self) -> np.array:
        "determine which viewpoint (out of the simulated particles) to move the robot to for updating the grid"
        
        if self.init == True:
            self.init = False
            pos = self.gbest #avoid doing 2 iterations of pso before doing the first voxel grid update
            vp_utility = self.gbest_obj
        else:    
            obj = self.update()
            pos = self.X[obj.argmin()] #the particle with the highest utility / smallest loss in the current step
            vp_utility = obj.min()

        quat = look_at_rotation(pos, self.target_params)
        viewpoint = np.zeros(7)
        viewpoint[:3] = pos.detach().cpu().numpy()
        viewpoint[3:] = quat.detach().cpu().numpy()

        return viewpoint, vp_utility
        

    def get_occupied_points(self):
        voxel_points, sem_conf_scores, sem_class_ids = (
            self.voxel_grid.get_occupied_points()
        )
        voxel_points = voxel_points.cpu().numpy()
        sem_conf_scores = sem_conf_scores.cpu().numpy()
        sem_class_ids = sem_class_ids.cpu().numpy()
        return voxel_points, sem_conf_scores, sem_class_ids

    def visualize(self):
        """
        Visualize the voxel grid, the target and the camera bounds in rviz
        """
        voxel_points, sem_conf_scores, sem_class_ids = self.get_occupied_points()
        self.rviz_visualizer.visualize_voxels(
            voxel_points, sem_conf_scores, sem_class_ids
        )
        # Visualize target
        target = self.target_params.detach().cpu().numpy()
        rois = np.array([[*target, 1.0, 0.0, 0.0, 0.0]])
        self.rviz_visualizer.visualize_rois(numpy_to_pose_array(rois))
        # Visualize camera bounds
        camera_bounds = self.camera_bounds.cpu().numpy()[:, :3]
        self.rviz_visualizer.visualize_camera_bounds(camera_bounds)

   
    def calculate_F1(self):

        # Retrieve the data
        voxel_points, sem_conf_scores, sem_class_ids = self.get_occupied_points()

        # Filter voxel points by class (target class: sem_class_ids == 0)
        target_voxels = np.array([voxel for i, voxel in enumerate(voxel_points) if sem_class_ids[i] == 0])

        self.target_voxels = target_voxels

        if len(target_voxels) == 0:
            return 0,0,0  # If no target voxels, return F1 score of 0
        
        # Build a k-d tree from mesh coordinates and target voxels
        mesh_tree = KDTree(self.mesh_coordinates)
        voxel_tree = KDTree(target_voxels)  

        cube_half_size = 0.002  # the size of a voxel
        search_radius = np.sqrt(3 * (cube_half_size ** 2)) 
        nr_correct_voxels = 0

        for voxel in target_voxels:
            # Query the tree for nearby mesh coordinates
            candidate_indices = mesh_tree.query_ball_point(voxel, r=search_radius)
            for idx in candidate_indices:
                coord = self.mesh_coordinates[idx]
                if (
                    abs(voxel[0] - coord[0]) <= cube_half_size and
                    abs(voxel[1] - coord[1]) <= cube_half_size and
                    abs(voxel[2] - coord[2]) <= cube_half_size
                ):
                    nr_correct_voxels += 1
                    break  # Stop after the first valid match
        
        nr_recalled_mesh_coords = 0

        for coord in self.mesh_coordinates:
            # Query the tree for nearby voxel points
            candidate_indices = voxel_tree.query_ball_point(coord, r=search_radius)
            for idx in candidate_indices:
                voxel = target_voxels[idx]
                if (
                    abs(voxel[0] - coord[0]) <= cube_half_size and
                    abs(voxel[1] - coord[1]) <= cube_half_size and
                    abs(voxel[2] - coord[2]) <= cube_half_size
                ):
                    nr_recalled_mesh_coords += 1
                    break  # Stop after the first valid match

        # Calculate precision, recall, and F1 score
        precision = nr_correct_voxels / len(target_voxels)
        recall = nr_recalled_mesh_coords / len(self.mesh_coordinates)
        F1_score = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0
        )

        return F1_score, recall, precision

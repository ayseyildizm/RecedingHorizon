#!/usr/bin/env python3
# ROS node to run the viewpoint planning algorithms

import rospy
from viewpoint_planners.viewpoint_planning import ViewpointPlanning
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def plot_trail_viewpoints():

        # Extracting data for trail plotting
        points_gradient = np.array(viewpoint_planner.trail_gradient).T
        points_gradient2 = np.array(viewpoint_planner.trail_gradient2).T
        points_pso = np.array(viewpoint_planner.trail_pso).T
        points_random = np.array(viewpoint_planner.trail_random).T
        points_pso2 = np.array(viewpoint_planner.trail_pso2).T
        points_pso3 = np.array(viewpoint_planner.trail_pso3).T
        points_pso5 = np.array(viewpoint_planner.trail_pso5).T

        occlusion_coords = [0.65, -0.3, 1.1]

        # Dimensions of the occlusion box
        occlusion_width = 0.3
        occlusion_depth = 0.03
        occlusion_height = 0.3

        # Setting up the plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot paths
        ax.plot(points_gradient[0], points_gradient[1], points_gradient[2], marker='x', linestyle='-', label="Gradient lr = 0.03")
        ax.scatter(*points_gradient.T[0], color='red')  # Starting point

        ax.plot(points_gradient2[0], points_gradient2[1], points_gradient2[2], marker='x', linestyle=':', label="Gradient lr = 0.5")
        ax.scatter(*points_gradient2.T[0], color='red')  # Starting point

        ax.plot(points_random[0], points_random[1], points_random[2], marker='x', linestyle='-.', label="Random")
        ax.scatter(*points_random.T[0], color='red')  # Starting point
        
        ax.plot(points_pso3[0], points_pso3[1], points_pso3[2], marker='x', linestyle='--', label="PSO 3 particles")
        ax.scatter(*points_pso3.T[0], color='red')  # Starting point

        # Plot mesh_coordinates
        ax.scatter(
            viewpoint_planner.pso_planner3.mesh_coordinates[:, 0],
            viewpoint_planner.pso_planner3.mesh_coordinates[:, 1],
            viewpoint_planner.pso_planner3.mesh_coordinates[:, 2],
            c='red', s=1
        )

        # Add occlusion as a box
        x, y, z = occlusion_coords
        ax.bar3d(
            x - occlusion_width / 2,  # Bottom-left corner x
            y - occlusion_depth / 2,  # Bottom-left corner y
            z - occlusion_height / 2,  # Bottom-left corner z
            occlusion_width,          # Width of the box
            occlusion_depth,          # Depth of the box
            occlusion_height,         # Height of the box
            color='orange',
            alpha=0.5,                # Transparency for better visibility
            
        )

        # Configure plot
        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

def plot_trail_particles():

        # Extracting data for trail plotting
        trajectories_particles = np.array(viewpoint_planner.pso_planner3.particle_trajectories).transpose(1,2,0)

        occlusion_coords = [0.65, -0.3, 1.1]

        # Dimensions of the occlusion box
        occlusion_width = 0.3
        occlusion_depth = 0.03
        occlusion_height = 0.3
        

        # Setting up the plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot paths   
        for particle in trajectories_particles:
            ax.plot(particle[0], particle[1], particle[2], marker='o')
            ax.scatter(*particle.T[0], color='red')  # Starting point


        # Plot mesh_coordinates
        ax.scatter(
            viewpoint_planner.pso_planner3.mesh_coordinates[:, 0],
            viewpoint_planner.pso_planner3.mesh_coordinates[:, 1],
            viewpoint_planner.pso_planner3.mesh_coordinates[:, 2],
            c='red', s=1
        )

        #ax.scatter(*target_coords, color='blue', s=400, marker='o')

        # Add occlusion as a box
        x, y, z = occlusion_coords
        ax.bar3d(
            x - occlusion_width / 2,  # Bottom-left corner x
            y - occlusion_depth / 2,  # Bottom-left corner y
            z - occlusion_height / 2,  # Bottom-left corner z
            occlusion_width,          # Width of the box
            occlusion_depth,          # Depth of the box
            occlusion_height,         # Height of the box
            color='orange',
            alpha=0.5,                # Transparency for better visibility
          
        )

        # Configure plot
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()


def plot_results(target_coverages, computation_times, trajectory_distances, recalls, precisions, i):

    # Plot 1: Target coverage
    plt.plot(target_coverages[0], label="PSO 2 particles", linestyle='-', marker='o')
    plt.plot(target_coverages[1], label="PSO 3 particles", linestyle='--', marker='s')
    plt.plot(target_coverages[2], label="PSO 4 particles", linestyle='-.', marker='^')
    plt.plot(target_coverages[3], label="PSO 5 particles", linestyle=':', marker='d')
    plt.plot(target_coverages[4], label="Random", linestyle='-', marker='x')
    plt.plot(target_coverages[5], label="Gradient lr = 0.03", linestyle='--', marker='*')
    plt.plot(target_coverages[6], label="Gradient lr = 0.5", linestyle='-.', marker='+')

    plt.xlabel("Iterations")
    plt.ylabel(f"Target coverage (%), average over {i} experiments")
    plt.legend()
    plt.show()

    # Plot 2: Computation time
    plt.plot(computation_times[0], label="PSO 2 particles", linestyle='-', marker='o')
    plt.plot(computation_times[1], label="PSO 3 particles", linestyle='--', marker='s')
    plt.plot(computation_times[2], label="PSO 4 particles", linestyle='-.', marker='^')
    plt.plot(computation_times[3], label="PSO 5 particles", linestyle=':', marker='d')
    plt.plot(computation_times[4], label="Random", linestyle='-', marker='x')
    plt.plot(computation_times[5], label="Gradient lr = 0.03", linestyle='--', marker='*')
    plt.plot(computation_times[6], label="Gradient lr = 0.5", linestyle='-.', marker='+')
    
    plt.xlabel("Iterations")
    plt.ylabel(f"Computation time, average over {i} experiments")
    plt.legend()
    plt.show()

    # Plot 3: Trajectory distance
    plt.plot(trajectory_distances[0], label="PSO 2 particles", linestyle='-', marker='o')
    plt.plot(trajectory_distances[1], label="PSO 3 particles", linestyle='--', marker='s')
    plt.plot(trajectory_distances[2], label="PSO 4 particles", linestyle='-.', marker='^')
    plt.plot(trajectory_distances[3], label="PSO 5 particles", linestyle=':', marker='d')
    plt.plot(trajectory_distances[4], label="Random", linestyle='-', marker='x')
    plt.plot(trajectory_distances[5], label="Gradient lr = 0.03", linestyle='--', marker='*')
    plt.plot(trajectory_distances[6], label="Gradient lr = 0.5", linestyle='-.', marker='+')

    plt.xlabel("Iterations")
    plt.ylabel(f"Trajectory distance, average over {i} experiments")
    plt.legend()
    plt.show()

    # Plot 4: Recall
    plt.plot(recalls[0], label="PSO 2 particles", linestyle='-', marker='o')
    plt.plot(recalls[1], label="PSO 3 particles", linestyle='--', marker='s')
    plt.plot(recalls[2], label="PSO 4 particles", linestyle='-.', marker='^')
    plt.plot(recalls[3], label="PSO 5 particles", linestyle=':', marker='d')
    plt.plot(recalls[4], label="Random", linestyle='-', marker='x')
    plt.plot(recalls[5], label="Gradient lr = 0.03", linestyle='--', marker='*')
    plt.plot(recalls[6], label="Gradient lr = 0.5", linestyle='-.', marker='+')

    plt.xlabel("Iterations")
    plt.ylabel(f"Recall of reconstructed target voxels (%), average over {i} experiments")
    plt.legend()
    plt.show()

    # Plot 5: Precision
    plt.plot(precisions[0], label="PSO 2 particles", linestyle='-', marker='o')
    plt.plot(precisions[1], label="PSO 3 particles", linestyle='--', marker='s')
    plt.plot(precisions[2], label="PSO 4 particles", linestyle='-.', marker='^')
    plt.plot(precisions[3], label="PSO 5 particles", linestyle=':', marker='d')
    plt.plot(precisions[4], label="Random", linestyle='-', marker='x')
    plt.plot(precisions[5], label="Gradient lr = 0.03", linestyle='--', marker='*')
    plt.plot(precisions[6], label="Gradient lr = 0.5", linestyle='-.', marker='+')

    plt.xlabel("Iterations")
    plt.ylabel(f"Precision of reconstructed target voxels (%), average over {i} experiments")
    plt.legend()
    plt.show()


def merge_results(viewpoint_planner, target_coverages, computation_times, trajectory_distances, recalls, precisions, i):
     
    target_coverages[0] = (1 - 1/i) * target_coverages[0] + (1/i) * viewpoint_planner.coverages_pso2
    target_coverages[1] = (1 - 1/i) * target_coverages[1] + (1/i) * viewpoint_planner.coverages_pso3
    target_coverages[2] = (1 - 1/i) * target_coverages[2] + (1/i) * viewpoint_planner.coverages_pso
    target_coverages[3] = (1 - 1/i) * target_coverages[3] + (1/i) * viewpoint_planner.coverages_pso5
    target_coverages[4] = (1 - 1/i) * target_coverages[4] + (1/i) * viewpoint_planner.coverages_random
    target_coverages[5] = (1 - 1/i) * target_coverages[5] + (1/i) * viewpoint_planner.coverages_gradient
    target_coverages[6] = (1 - 1/i) * target_coverages[6] + (1/i) * viewpoint_planner.coverages_gradient2

    computation_times[0] = (1 - 1/i) * computation_times[0] + (1/i) * viewpoint_planner.cumulative_time_pso2
    computation_times[1] = (1 - 1/i) * computation_times[1] + (1/i) * viewpoint_planner.cumulative_time_pso3
    computation_times[2] = (1 - 1/i) * computation_times[2] + (1/i) * viewpoint_planner.cumulative_time_pso
    computation_times[3] = (1 - 1/i) * computation_times[3] + (1/i) * viewpoint_planner.cumulative_time_pso5
    computation_times[4] = (1 - 1/i) * computation_times[4] + (1/i) * viewpoint_planner.cumulative_time_random
    computation_times[5] = (1 - 1/i) * computation_times[5] + (1/i) * viewpoint_planner.cumulative_time_gradient
    computation_times[6] = (1 - 1/i) * computation_times[6] + (1/i) * viewpoint_planner.cumulative_time_gradient2

    trajectory_distances[0] = (1 - 1/i) * trajectory_distances[0] + (1/i) * viewpoint_planner.trajectory_distance_pso2
    trajectory_distances[1] = (1 - 1/i) * trajectory_distances[1] + (1/i) * viewpoint_planner.trajectory_distance_pso3
    trajectory_distances[2] = (1 - 1/i) * trajectory_distances[2] + (1/i) * viewpoint_planner.trajectory_distance_pso
    trajectory_distances[3] = (1 - 1/i) * trajectory_distances[3] + (1/i) * viewpoint_planner.trajectory_distance_pso5
    trajectory_distances[4] = (1 - 1/i) * trajectory_distances[4] + (1/i) * viewpoint_planner.trajectory_distance_random
    trajectory_distances[5] = (1 - 1/i) * trajectory_distances[5] + (1/i) * viewpoint_planner.trajectory_distance_gradient         
    trajectory_distances[6] = (1 - 1/i) * trajectory_distances[6] + (1/i) * viewpoint_planner.trajectory_distance_gradient2 

    recalls[0] = (1 - 1/i) * recalls[0] + (1/i) * viewpoint_planner.recall_pso2
    recalls[1] = (1 - 1/i) * recalls[1] + (1/i) * viewpoint_planner.recall_pso3
    recalls[2] = (1 - 1/i) * recalls[2] + (1/i) * viewpoint_planner.recall_pso
    recalls[3] = (1 - 1/i) * recalls[3] + (1/i) * viewpoint_planner.recall_pso5
    recalls[4] = (1 - 1/i) * recalls[4] + (1/i) * viewpoint_planner.recall_random
    recalls[5] = (1 - 1/i) * recalls[5] + (1/i) * viewpoint_planner.recall_gradient
    recalls[6] = (1 - 1/i) * recalls[6] + (1/i) * viewpoint_planner.recall_gradient2 

    precisions[0] = (1 - 1/i) * precisions[0] + (1/i) * viewpoint_planner.precision_pso2
    precisions[1] = (1 - 1/i) * precisions[1] + (1/i) * viewpoint_planner.precision_pso3
    precisions[2] = (1 - 1/i) * precisions[2] + (1/i) * viewpoint_planner.precision_pso
    precisions[3] = (1 - 1/i) * precisions[3] + (1/i) * viewpoint_planner.precision_pso5
    precisions[4] = (1 - 1/i) * precisions[4] + (1/i) * viewpoint_planner.precision_random
    precisions[5] = (1 - 1/i) * precisions[5] + (1/i) * viewpoint_planner.precision_gradient
    precisions[6] = (1 - 1/i) * precisions[6] + (1/i) * viewpoint_planner.precision_gradient2 

    return target_coverages, computation_times, trajectory_distances, recalls, precisions



if __name__ == "__main__":
    rospy.init_node("viewpoint_planning")

    target_coverages, computation_times, trajectory_distances, recalls, precisions = np.zeros((5, 7, 13)) #7 methods and 12 iterations + init
    
    for i in range(1):
        viewpoint_planner = ViewpointPlanning(0)

        for _ in range(12):
            viewpoint_planner.run_pso2()   
        for _ in range(12):
            viewpoint_planner.run_pso3()  
        for _ in range(12):
            viewpoint_planner.run_pso()  
        for _ in range(12):
            viewpoint_planner.run_pso5()  
        for _ in range(12):
            viewpoint_planner.run_random()
        for _ in range(12):
            viewpoint_planner.run_gradient_nbv()
        for _ in range(12):
            viewpoint_planner.run_gradient_nbv2()
        for _ in range(12):
            viewpoint_planner.run_rh()

        plot_trail_viewpoints()
        plot_trail_particles()
        target_coverages, computation_times, trajectory_distances, recalls, precisions = merge_results(viewpoint_planner, target_coverages, computation_times, trajectory_distances, recalls, precisions, i+1)
        plot_results(target_coverages, computation_times, trajectory_distances, recalls, precisions, i+1)

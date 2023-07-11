# generate a plane point cloud with 1024 points

import numpy as np
import open3d as o3d

def generate_plane_points(center, size):

    x = np.linspace(center[0] - size/2, center[0] + size/2, num=32)
    y = np.linspace(center[1] - size/2, center[1] + size/2, num=32)
    z = np.linspace(center[2] - size/2, center[2] + size/2, num=32)
    xx, yy, zz = np.meshgrid(x, y, z)
    points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
    return points

# Generate ground plane
ground_size =40
num_ground_points = 1024
ground_points = np.random.uniform(-ground_size, ground_size, size=(num_ground_points, 3))
ground_points[:, 2] = np.random.uniform(-0.1, 0.1, size=(num_ground_points,))
ground_labels = np.full((num_ground_points,), 0) # Ground label is 0

# visualize the ground plane and all points should be green
# Create Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(ground_points)

# Assign colors based on labels
label_to_color = {1: [1, 0, 0], 0: [0, 1, 0]}
colors = [label_to_color[label] for label in ground_labels]
pcd.colors = o3d.utility.Vector3dVector(colors)

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd])



# generate a cube point cloud with 1024 points

def generate_cube_points(center, size):
    x = np.linspace(center[0] - size/2, center[0] + size/2, num=32)
    y = np.linspace(center[1] - size/2, center[1] + size/2, num=32)
    z = np.linspace(center[2] - size/2, center[2] + size/2, num=32)
    xx, yy, zz = np.meshgrid(x, y, z)
    points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
    return points



# Generate cube point cloud data
num_cubes = 5

cube_centers = np.random.uniform(-40, 40, size=(num_cubes, 3))
# z coordinate of cube centers should be greater than 10
cube_centers[:, 2] = np.random.uniform(10, 40, size=(num_cubes,))
cube_size = np.random.uniform(1, 2, size=(num_cubes,))
cube_points = np.vstack([generate_cube_points(center, np.random.uniform(10, 25, size=(num_cubes,))) for center in cube_centers])
cube_labels = np.full((cube_points.shape[0],), 1) # Cube label is 1


# Select random subset of cube points
selected_indices = np.random.choice(cube_points.shape[0], size=500, replace=False)
cube_points = cube_points[selected_indices, :]
cube_labels = cube_labels[selected_indices]

# visualize the cube and all points should be red
# Create Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(cube_points)

# Assign colors based on labels
label_to_color = {1: [1, 0, 0], 0: [0, 1, 0]}
colors = [label_to_color[label] for label in cube_labels]
pcd.colors = o3d.utility.Vector3dVector(colors)

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd])


# Concatenate cube and ground points
all_points = np.vstack([cube_points, ground_points])
all_labels = np.concatenate([cube_labels, ground_labels])

# remove random points to make total points 1024
selected_indices = np.random.choice(all_points.shape[0], size=1024, replace=False)
all_points = all_points[selected_indices, :]
all_labels = all_labels[selected_indices]

# visualize the cube and ground plane and all points should be red and green
# Create Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(all_points)

# Assign colors based on labels
label_to_color = {1: [1, 0, 0], 0: [0, 1, 0]}
colors = [label_to_color[label] for label in all_labels]
pcd.colors = o3d.utility.Vector3dVector(colors)

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd])

print(all_points.shape)

# Save point cloud data and labels
np.savez('point_cloud_data_test.npz', points=all_points, labels=all_labels)






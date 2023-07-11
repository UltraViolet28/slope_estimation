#!/usr/bin/env python3

import numpy as np
import open3d as o3d
import sys
import os

# code to compute normals from point cloud in ply format

def get_normals(point_cloud):
    # Compute surface normals from point cloud
    # Input:
    #   point_cloud: Nx3 array of 3D points
    # Output:
    #   normals: Nx3 array of surface normals
    #   (Note: these are not unit normals)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.estimate_normals()
    normals = np.asarray(pcd.normals)
    return normals


# read point cloud from ply file from command line
if len(sys.argv) < 2:
    print("Usage: python3 get_normals.py point_cloud.ply")
    sys.exit()
point_cloud_file = sys.argv[1]
point_cloud = np.asarray(o3d.io.read_point_cloud(point_cloud_file).points)

# compute normals
normals = get_normals(point_cloud)
print(normals.shape)


# save normals to file
normals_file = os.path.splitext(point_cloud_file)[0] + '_normals.txt'
np.savetxt(normals_file, normals)

# visualize normals and point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud)
pcd.normals = o3d.utility.Vector3dVector(normals)
o3d.visualization.draw_geometries([pcd])


print(normals[10:15,:])
print(point_cloud[10:15,:])


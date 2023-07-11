# function to get the normals of of points in the point cloud with a range of normals belonging to the same class
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Semantic segmentation of point clouds

# point clouds in .npy format
def get_normals(point_cloud):
    pass
    # load the point cloud
    # point_cloud = np.load('point_cloud.npy')
    # # convert the point cloud to a open3d point cloud
    # point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(point_cloud))
    # # estimate the normals of the point cloud
    # point_cloud.estimate_normals()
    # # get the normals of the point cloud
    # normals = np.asarray(point_cloud.normals)
    # # return the normals
    # return normals



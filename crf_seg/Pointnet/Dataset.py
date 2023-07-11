#!/usr/bin/env python3
#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import open3d as o3d
import matplotlib.pyplot as plt

# create a dataloader for point cloud data
# read data from npz file

CATEGORIES = {'Cube': 0, 'Ground': 1}
COLORS = {'Cube': [1, 0, 0], 'Ground': [0, 1, 0]}

class PointCloudDataset(Dataset):
    def __init__(self, data_dir, num_points=2500):
        super(PointCloudDataset, self).__init__()
        self.data_dir = data_dir
        self.num_points = num_points
        self.data = np.load(self.data_dir)
        self.points = self.data['points']
        # print(self.points.shape)

        # convert labels to label encoding
        self.labels = self.data['labels']
        self.labels = np.array([CATEGORIES[label] for label in self.labels])

    def __len__(self):
        return self.points.shape[0]
    
    def __getitem__(self, idx):
        points = self.points[idx]
        labels = self.labels[idx]
 
        return points, labels


file_name = 'point_cloud_data.npz'

# create dataset
dataset = PointCloudDataset(file_name, num_points=2500)

# get a sample
sample = dataset[0]
print(sample[0].shape)

# # create dataloader
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# # visualize a batch of data
# batch = next(iter(dataloader))
# print(batch[0].shape)

# # print labels
# print(batch[1])



# ## Visualize the point cloud

# # Get the points and labels from the dataset
# points = dataset.points
# labels = dataset.labels

# # Create an Open3D point cloud
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)

# # Assign colors based on labels
# label_to_color = {'Cube': [1, 0, 0], 'Ground': [0, 1, 0]}
# colors = [label_to_color[label] for label in labels]
# pcd.colors = o3d.utility.Vector3dVector(colors)

# # Visualize the point cloud
# o3d.visualization.draw_geometries([pcd])
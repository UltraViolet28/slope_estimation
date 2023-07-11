import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from split_point_cloud import split_point_cloud

class PointCloudDataset(Dataset):
    def __init__(self, point_clouds, labels):
        self.point_clouds = point_clouds
        self.labels = labels
    
    def __len__(self):
        return len(self.point_clouds)
    
    def __getitem__(self, index):
        point_cloud = self.point_clouds[index]
        label = self.labels[index]
        return point_cloud, label

def preprocessing(point_cloud, labels, num_points,batch_size,  train_ratio):
    # Split the point cloud into smaller point clouds of size 1024
    point_clouds, labels = split_point_cloud(point_cloud, num_points, labels)

    # Convert data to tensors
    point_clouds = [torch.tensor(pc, dtype=torch.float32) for pc in point_clouds]
    labels = [torch.tensor(lbl, dtype=torch.long) for lbl in labels]

    # Pad or truncate point clouds to a fixed number of points (num_points)
    point_clouds = [pc[:num_points] if pc.shape[0] >= num_points else np.pad(pc, ((0, num_points - pc.shape[0]), (0, 0)), mode='constant') for pc in point_clouds]

    # Create a tensor of shape (batch_size, num_points, 3) to hold the input point clouds
    point_clouds = torch.stack(point_clouds)

    # Create a tensor of shape (batch_size, num_points) to hold the ground truth labels
    labels = torch.stack(labels)

    # Create train and test splits
    train_size = int(train_ratio * len(point_clouds))

    train_point_clouds = point_clouds[:train_size]
    train_labels = labels[:train_size]

    test_point_clouds = point_clouds[train_size:]
    test_labels = labels[train_size:]

    # Create the train and test datasets
    train_dataset = PointCloudDataset(train_point_clouds, train_labels)
    test_dataset = PointCloudDataset(test_point_clouds, test_labels)

    # Create train and test data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader



# SAMPLE USAGE 
# # Read point cloud data and labels from file
# # There in a large pointcloud in the file 
# data = np.load('point_cloud_data.npz')
# point_cloud = data['points']
# labels = data['labels']

# train_dataloader, test_dataloader = preprocessing(point_cloud, labels, 1024, 32, 0.8)

# print(train_dataloader.dataset[0][0].shape)
# print(train_dataloader.dataset[0][1].shape) # 1024 points

# batch = next(iter(train_dataloader))
# print(batch[0].shape)

# # Split the point cloud into smaller point clouds of size 1024
# window_size = 1024
# point_clouds, labels = split_point_cloud(point_cloud, window_size, labels)

# # #--------------------------------------------------------------------------------

# # Convert data to tensors
# point_clouds = [torch.tensor(pc, dtype=torch.float32) for pc in point_clouds]
# labels = [torch.tensor(lbl, dtype=torch.long) for lbl in labels]


# # Pad or truncate point clouds to a fixed number of points (num_points)
# num_points = 1024
# point_clouds = [pc[:num_points] if pc.shape[0] >= num_points else np.pad(pc, ((0, num_points - pc.shape[0]), (0, 0)), mode='constant') for pc in point_clouds]


# # Create a tensor of shape (batch_size, num_points, 3) to hold the input point clouds
# point_clouds = torch.stack(point_clouds)

# # Create a tensor of shape (batch_size, num_points) to hold the ground truth labels
# labels = torch.stack(labels)

# # print(point_clouds.shape)
# # print(labels.shape)


# #--------------------------------------------------------------------------------


# # Create train and test splits
# train_ratio = 0.8
# train_size = int(train_ratio * len(point_clouds))

# train_point_clouds = point_clouds[:train_size]
# train_labels = labels[:train_size]

# test_point_clouds = point_clouds[train_size:]
# test_labels = labels[train_size:]

# # Create the train and test datasets
# train_dataset = PointCloudDataset(train_point_clouds, train_labels)
# test_dataset = PointCloudDataset(test_point_clouds, test_labels)

# # Create train and test data loaders
# batch_size = 32
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



# print(train_dataloader.dataset[0][0].shape)
# print(train_dataloader.dataset[0][1].shape)

# # batch of point clouds
# batch = next(iter(train_dataloader))
# print(batch[0].shape)
# print(batch[1].shape)


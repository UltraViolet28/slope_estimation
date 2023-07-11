import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader


# create a dataloader for point cloud data
# read data from npz file
# input shape: (batch_size, num_points, 3)

class PointCloudDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=4):
        super(PointCloudDataLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=self.collate_fn)

    # collate_fn: combines a list of samples to form a mini-batch of shape (batch_size, num_points, 3)
    def collate_fn(self, batch):
        points = []
        labels = []
        for b in batch:
            points.append(b[0])
            labels.append(b[1])
        points = np.array(points)
        labels = np.array(labels)
        # print(points.shape)
        return points, labels






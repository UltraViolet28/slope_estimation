#!/usr/bin/env python
# coding: utf-8
import os
import sys
import time
import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter


# import from other files
from Dataset import PointCloudDataset
from Pointnet import PointNet
from Dataloader import PointCloudDataLoader



# create summary writer
writer = SummaryWriter('runs/pointnet_experiment_1')

# Hyperparameters
num_epochs = 10
batch_size = 32
learning_rate = 0.001
num_classes = 2


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)



# create a dataloader for point cloud data
# read data from npz file
file_name = 'point_cloud_data.npz'
dataset = PointCloudDataset(file_name, num_points=2500)

# create dataloader
# input shape: (batch_size, num_points, 3)
# label shape: (batch_size,)
dataloader = PointCloudDataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)


# visualize a batch of data
batch = next(iter(dataloader))
print(batch[0].shape)

print(batch[1])
print(batch[1].shape)

sys.exit()

model = PointNet(classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)

# Single training step
def train_step(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for points, labels in dataloader:
        points = points.to(device)
        print(labels)
        labels = labels.to(device)
        # Forward pass
        points = points.transpose(1, 2)
        outputs, _ = model(points)
        loss = criterion(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

# Single validation step
def val_step(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    for points, labels in dataloader:
        points = points.to(device)
        labels = labels.to(device)
        # Forward pass
        outputs, _ = model(points)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
    return running_loss / len(dataloader)

a = train_step(model, dataloader, criterion, optimizer, device)
print(a)




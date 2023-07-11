import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import open3d as o3d
import sys
import argparse
import datetime
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from simple_model import PointNetSegmentation
from simple_dataset import preprocessing

# Create a tensorboard summary in the folder 'runs'
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
writer = SummaryWriter(f'runs/data_visualization_{current_time}')

# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Hyperparameters Defaults
num_points = 1024 # do not change this value
train_ratio = 0.8
num_classes = 2
max_epochs = 20
batch_size = 8

# Get the hyperparameters from the command line
parser = argparse.ArgumentParser()
parser.add_argument('--train_ratio', type=float, default=train_ratio)
parser.add_argument('--num_classes', type=int, default=num_classes)
parser.add_argument('--max_epochs', type=int, default=max_epochs)
parser.add_argument('--batch_size', type=int, default=batch_size)
parser.add_argument('--model_name', type=str , default="model.pth")
args = parser.parse_args()

# Set the hyperparameters
train_ratio = args.train_ratio
num_classes = args.num_classes
max_epochs = args.max_epochs
batch_size = args.batch_size

# Format for running the script
# python simple_train.py --train_ratio 0.8 --num_classes 2 --max_epochs 20 --batch_size 8 --model_name model.pth

# Save the hyperparameters to tensorboard
writer.add_hparams(
    {'num_points': num_points, 'train_ratio': train_ratio, 'num_classes': num_classes, 'max_epochs': max_epochs,
     'batch_size': batch_size}, {})


# Read point cloud data and labels from file
data = np.load('point_cloud_data.npz')
point_cloud = data['points']
labels = data['labels']

# Preprocess the data
train_dataloader, test_dataloader = preprocessing(point_cloud, labels, num_points, batch_size, train_ratio)
batch = next(iter(train_dataloader))
print(batch[0].shape)  # (batch_size, num_points, 3)

print(len(train_dataloader))


# Create the model
model = PointNetSegmentation(num_classes=num_classes)
model = model.to(device)
print(model)


# Model on Tensorboard
writer.add_graph(model, batch[0].transpose(2, 1).to(device))


# Optimizer and Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()


# Training
# Training
global_step = 0  # Initialize global step
for epoch in range(max_epochs):
    model.train()
    running_loss = 0.0
    i = 0

    for point_clouds, labels in train_dataloader:
        # Zero the gradients
        optimizer.zero_grad()

        point_clouds = point_clouds.transpose(2, 1)  # (32, 3, 1024)

        reshaped_point_clouds = point_clouds.reshape(-1, 3)
        reshaped_labels = labels.reshape(-1)

        # Use the reshaped tensors in the add_embedding() function
        writer.add_embedding(reshaped_point_clouds, metadata=reshaped_labels, tag='training data', global_step=global_step)

        # Move the data to the device that is used
        point_clouds = point_clouds.to(device)
        labels = labels.to(device)  # (32, 1024)

        # Forward pass
        outputs = model(point_clouds)  # (32, 1024, 2)

        # Calculate the loss
        loss = loss_fn(outputs.view(-1, 2), labels.view(-1))

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Print iteration loss every 10 iterations
        if (i + 1) % 3 == 0:
            print(f"Iteration {i + 1}/{len(train_dataloader)}, Loss: {loss.item():.4f}")
        i += 1

        # Increment global_step
        global_step += 1

        # Add the loss to the tensorboard
        writer.add_scalar('training loss', loss.item(), epoch * len(train_dataloader) + i)

    # Print the average loss for the epoch
    avg_loss = running_loss / len(train_dataloader)
    print(f"Epoch [{epoch + 1}/{max_epochs}], Loss: {avg_loss:.4f}")

# Testing
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for point_clouds, labels in train_dataloader:
        point_clouds = point_clouds.transpose(2, 1).to(device)  # (batch_size, 3, num_points)
        labels = labels.to(device)  # (batch_size, num_points)

        # Forward pass
        outputs = model(point_clouds)  # (batch_size, num_points, num_classes)
        predicted_labels = torch.argmax(outputs, dim=2)  # (batch_size, num_points)

        # Calculate accuracy
        total += labels.size(0) * labels.size(1)
        correct += (predicted_labels == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

# Tensorboard
writer.add_scalar('test accuracy', accuracy, epoch * len(train_dataloader) + i)

# # Iterate over the test_dataloader
# for batch in test_dataloader:
#     point_cloud = batch[0].numpy()  # Assuming the point cloud data is in the first element of the batch
#     labels = batch[1].numpy()  # Assuming the labels are in the second element of the batch

#     # Create Open3D point cloud object
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(point_cloud.reshape(-1, 3))  # Reshape the point cloud to Nx3 shape

#     # Set colors for visualization (assuming num_classes = 2)
#     color_palette = np.array([[0, 255, 0],  # Class 0 color (red)
#                               [0, 0, 255]])  # Class 1 color (blue)
#     point_colors = color_palette[labels.reshape(-1)]

#     # Assign colors to point cloud
#     pcd.colors = o3d.utility.Vector3dVector(point_colors / 255.0)

#     # Visualize the point cloud
#     o3d.visualization.draw_geometries([pcd])

# Save the model
torch.save(model.state_dict(), args.model_name)

# Close Tensorboard writer
writer.close()

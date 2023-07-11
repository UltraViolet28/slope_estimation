import torch
import torch.nn as nn
import torch.nn.functional as F


# Input Transform Network (T-Net)
class InputTransformNet(nn.Module):
    def __init__(self, input_dim):
        super(InputTransformNet, self).__init__()
        self.input_dim = input_dim
        
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, input_dim * input_dim)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        batch_size = x.size(0)
        
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        x = torch.max(x, 2, keepdim=True)[0]
        
        x = x.view(batch_size, -1)
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        eye = torch.eye(self.input_dim, requires_grad=True).unsqueeze(0)
        if x.is_cuda:
            eye = eye.cuda()
        
        x = x.view(-1, self.input_dim, self.input_dim) + eye
        
        return x

# Feature Transform Network (T-Net)
class FeatureTransformNet(nn.Module):
    def __init__(self, input_dim):
        super(FeatureTransformNet, self).__init__()
        self.input_dim = input_dim
        
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, input_dim * input_dim)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        batch_size = x.size(0)
        
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        x = torch.max(x, 2, keepdim=True)[0]
        
        x = x.view(batch_size, -1)
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        eye = torch.eye(self.input_dim, requires_grad=True).unsqueeze(0)
        if x.is_cuda:
            eye = eye.cuda()
        
        x = x.view(-1, self.input_dim, self.input_dim) + eye
        
        return x


class PointNetSegmentation(nn.Module):
    def __init__(self, num_classes):
        super(PointNetSegmentation, self).__init__()

        self.num_classes = num_classes

        # Input Transform Network (T-Net)
        self.input_tnet = InputTransformNet(3)

        # Feature Transform Network (T-Net)
        self.feature_tnet = FeatureTransformNet(256)

        # PointNet Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # Decoder (output shape: (batch_size, num_classes * num_points)
        num_points = 1024
        self.decoder = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_classes * num_points)
        )

    def forward(self, point_cloud):
        # Apply Input Transform Network (T-Net)
        transformed_points = self.input_tnet(point_cloud)

        # PointNet Encoder
        global_features = self.encoder(transformed_points)

        # Apply Feature Transform Network (T-Net)
        transformed_features = self.feature_tnet(global_features)

        # Max pooling to obtain global feature representation
        global_features = torch.max(transformed_features, dim=2)[0]

        # MLP
        x = self.mlp(global_features)

        # Decoder
        logits = self.decoder(x)  # (batch_size, num_points*num_classes)

        # Reshape to (batch_size, num_points, num_classes)
        logits = logits.view(-1, 1024, self.num_classes)

        # Apply softmax to get class probabilities
        # (batch_size, num_points, num_classes)
        class_probs = F.softmax(logits, dim=2)

        return class_probs

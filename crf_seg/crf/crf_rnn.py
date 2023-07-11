
import torch
import torch.nn as nn


class CrfRnn(nn.Module):
    """
    PyTorch implementation of the CRF-RNN module for point cloud data.
    """

    def __init__(self, num_labels, num_iterations=5, num_layers=1, hidden_dim=128):
        """
        Create a new instance of the CRF-RNN layer.

        Args:
            num_labels:         Number of semantic labels in the dataset
            num_iterations:     Number of mean-field iterations to perform
            num_layers:         Number of layers in the compatibility matrix
            hidden_dim:         Dimension of hidden layers in the compatibility matrix
        """
        super(CrfRnn, self).__init__()

        self._softmax = torch.nn.Softmax(dim=0)
        self.num_iterations = num_iterations
        self.num_labels = num_labels

        # Compatibility transform layers
        self.compatibility_layers = nn.ModuleList()
        input_dim = num_labels
        output_dim = hidden_dim
        for _ in range(num_layers):
            self.compatibility_layers.append(nn.Linear(input_dim, output_dim))
            input_dim = output_dim

        self.compatibility_layers.append(nn.Linear(input_dim, num_labels))

    def forward(self, points, logits, normals):
        """
        Perform CRF inference on point cloud data.

        Args:
            points:     Tensor of shape (3, n) containing the 3D coordinates of the points
            logits:     Tensor of shape (num_classes, n) containing the unary logits
            normals:    Tensor of shape (3, n) containing the normal vectors of the points
        Returns:
            Log-Q distributions (logits) after CRF inference
        """
        n = points.shape[1]  # Number of points

        cur_logits = logits

        for _ in range(self.num_iterations):
            # Normalization
            q_values = self._softmax(cur_logits)

            # Compute pairwise potentials using dot product between normals
            pairwise_potentials = torch.matmul(normals.t(), normals)  # Shape: (n, n)

            # Compatibility transform
            msg_passing_out = torch.matmul(pairwise_potentials, q_values)  # Shape: (n, num_labels)

            for layer in self.compatibility_layers:
                msg_passing_out = layer(msg_passing_out)

            # Adding unary potentials
            cur_logits = msg_passing_out + logits

        return cur_logits


# Sample usage

# Example point cloud data
points = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]) # (3, 3)
logits = torch.tensor([[0.1, 0.2], [0.4, 0.5], [0.7, 0.8]])  # (3, 2)
normals = torch.tensor([[0.5, 0.5, 0.0], [-0.5, 0.5, 0.0], [0.0, -0.5, 0.5]]) # (3, 3)

# Initialize the CRF-RNN model
num_labels = 2
print(num_labels)
num_iterations = 5
num_layers = 2
hidden_dim = 64
crf_rnn = CrfRnn(num_labels, num_iterations, num_layers, hidden_dim)

# Perform CRF inference
output_logits = crf_rnn(points, logits, normals)

# Print the output logits
print(output_logits)



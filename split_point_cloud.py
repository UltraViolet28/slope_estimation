import numpy as np

def split_point_cloud(point_cloud, window_size, labels):
    num_points = point_cloud.shape[0]
    sub_point_clouds = []
    sub_labels = []

    for i in range(0, num_points - window_size + 1, window_size):
        sub_point_cloud = point_cloud[i:i+window_size]
        sub_label = labels[i:i+window_size]
        sub_point_clouds.append(sub_point_cloud)
        sub_labels.append(sub_label)

    return sub_point_clouds, sub_labels

# Example usage
# point_cloud = np.random.rand(100, 3)  # Example point cloud of shape (num_points, 3)
# labels = np.random.randint(0, 10, size=(100,))  # Example labels for each point
# window_size = 10  # Size of the sliding window

# sub_point_clouds, sub_labels = split_point_cloud(point_cloud, window_size, labels)

# # Print the sub-point clouds and corresponding labels
# for sub_point_cloud, sub_label in zip(sub_point_clouds, sub_labels):
#     print(sub_point_cloud)
#     print(sub_label)
#     print("---")

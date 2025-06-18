import torch
from sklearn.model_selection import train_test_split


# ✅ Reduce to 10% of the original hdpftl_data:


def random_downsample(X, y, fraction=0.1, seed=None):
    """
    Randomly downsample dataset to a fraction of original size.

    Args:
        X (np.array or similar): Input features
        y (np.array or similar): Labels
        fraction (float): Fraction of samples to keep (0 < fraction <= 1)
        seed (int, optional): Random seed for reproducibility

    Returns:
        (X_downsampled, y_downsampled): Downsampled data and labels
    """
    if seed is not None:
        np.random.seed(seed)

    total_samples = len(X)
    sample_size = int(fraction * total_samples)
    if sample_size == 0:
        raise ValueError("Fraction too small, results in zero samples.")

    indices = np.random.choice(total_samples, sample_size, replace=False)

    return X[indices], y[indices]


# Useful when you want the subset to maintain the same class proportions.
def stratified_downsample(X, y, fraction=0.1):
    X_small, _, y_small, _ = train_test_split(
        X, y, train_size=fraction, stratify=y, random_state=42
    )
    return X_small, y_small


# If one class is too large, reduce it to balance the hdpftl_dataset.
def class_specific_downsample(X, y, max_per_class=1000):
    indices = []
    for cls in torch.unique(y):
        cls_idx = (y == cls).nonzero(as_tuple=True)[0]
        selected = cls_idx[torch.randperm(len(cls_idx))[:max_per_class]]
        indices.append(selected)
    all_selected = torch.cat(indices)
    return X[all_selected], y[all_selected]


import numpy as np


def create_non_iid_partitions(X, y, num_clients=5, fraction=0.5, seed=None):
    """
    Create non-IID partitions of data by downsampling different classes for each client.

    Args:
        X (np.array): Features
        y (np.array): Labels (integer classes)
        num_clients (int): Number of clients
        fraction (float): Fraction of samples per client (relative to total dataset size)
        seed (int): Random seed

    Returns:
        client_data_dict: dict mapping client_id -> (X_client, y_client)
    """
    if seed is not None:
        np.random.seed(seed)

    unique_classes = np.unique(y)
    client_data_dict = {}

    # Split classes unevenly: assign each client a subset of classes
    classes_per_client = max(1, len(unique_classes) // num_clients)

    for client_id in range(num_clients):
        # Pick classes for this client — some overlap possible or disjoint sets
        start = client_id * classes_per_client
        end = start + classes_per_client
        client_classes = unique_classes[start:end]

        # Get indices of samples belonging to these classes
        indices = np.where(np.isin(y, client_classes))[0]

        # Downsample these indices randomly
        sample_size = int(fraction * len(X))
        if sample_size > len(indices):
            sample_size = len(indices)  # can't sample more than available

        sampled_indices = np.random.choice(indices, sample_size, replace=False)

        X_client = X[sampled_indices]
        y_client = y[sampled_indices]

        client_data_dict[client_id] = (X_client, y_client)

    return client_data_dict

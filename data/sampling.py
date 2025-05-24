import numpy as np
import torch
from sklearn.model_selection import train_test_split


# ✅ Reduce to 10% of the original data:
def random_downsample(X, y, fraction=0.1):
    total_samples = len(X)
    sample_size = int(fraction * total_samples)

    indices = np.random.choice(total_samples, sample_size, replace=False)

    return X[indices], y[indices]


# Useful when you want the subset to maintain the same class proportions.
def stratified_downsample(X, y, fraction=0.1):
    X_small, _, y_small, _ = train_test_split(
        X, y, train_size=fraction, stratify=y, random_state=42
    )
    return X_small, y_small


# If one class is too large, reduce it to balance the dataset.
def class_specific_downsample(X, y, max_per_class=1000):
    indices = []
    for cls in torch.unique(y):
        cls_idx = (y == cls).nonzero(as_tuple=True)[0]
        selected = cls_idx[torch.randperm(len(cls_idx))[:max_per_class]]
        indices.append(selected)
    all_selected = torch.cat(indices)
    return X[all_selected], y[all_selected]

import logging

import numpy as np
import torch

from hdpftl_aggregation.hdpftl_bayesian import aggregate_bayesian
from hdpftl_aggregation.hdpftl_fedavg import aggregate_fed_avg
from hdpftl_evaluation.evaluate_global_model import evaluate_global_model
from hdpftl_result.final_model import save
from hdpftl_training.train_device_model import train_device_model
from hdpftl_utility.config import NUM_CLIENTS
from hdpftl_utility.utils import setup_device


def dirichlet_partition(X, y, n_classes, alpha, seed=42):
    """
    Partition hdpftl_data indices using Dirichlet distribution.

    Args:
        X: Dataset features (unused, only y matters here for indexing).
        y: Dataset labels (1D numpy array or torch tensor).
        alpha: Dirichlet concentration parameter.
        n_clients: Number of partitions/clients.
        n_classes: Number of unique classes (optional).
        seed: Random seed for reproducibility.

    Returns:
        List of index lists, one per client.
    """
    if isinstance(y, torch.Tensor):
        y = y.numpy()

    np.random.seed(seed)
    if n_classes is None:
        n_classes = len(np.unique(y))

    client_indices = [[] for _ in range(NUM_CLIENTS)]

    for c in range(n_classes):
        class_indices = np.where(y == c)[0]
        np.random.shuffle(class_indices)

        proportions = np.random.dirichlet([alpha] * NUM_CLIENTS)
        # Scale to number of class samples
        proportions = (proportions * len(class_indices)).astype(int)

        # Correct any overflows/deficits
        while proportions.sum() > len(class_indices):
            proportions[np.argmax(proportions)] -= 1
        while proportions.sum() < len(class_indices):
            proportions[np.argmin(proportions)] += 1

        splits = np.split(class_indices, np.cumsum(proportions)[:-1])
        for i, split in enumerate(splits):
            client_indices[i].extend(split.tolist())

    return client_indices


def safe_split(tensor, proportions, dim=0):
    """
    Safely splits a tensor along any dimension using proportions that sum to 1.0.

    Args:
        tensor (torch.Tensor): The tensor to split.
        proportions (list or torch.Tensor): Proportions (floats), ideally summing to 1.0.
        dim (int): The dimension along which to split.

    Returns:
        list of torch.Tensor: Split tensor chunks.
    """
    total = tensor.size(dim)

    # Ensure proportions is a torch tensor
    if not isinstance(proportions, torch.Tensor):
        proportions = torch.tensor(proportions, dtype=torch.float)

    # Normalize proportions to sum to 1.0
    proportions = proportions / proportions.sum()

    # Compute split sizes and adjust for rounding
    split_sizes = (proportions * total).long()
    diff = total - split_sizes.sum()
    split_sizes[0] += diff  # Apply correction to the first split

    return torch.split(tensor, split_sizes.tolist(), dim=dim)


# --- Main HDPFTL pipeline ---

def hdpftl_pipeline(X_train, y_train, X_test, y_test, base_model_fn, alpha=0.5):
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    #               TRAINING
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    print("\n[1] Partitioning hdpftl_data using Dirichlet...")
    device = setup_device()
    num_classes = len(torch.unique(y_train))
    client_partitions = dirichlet_partition(X_train, y_train, num_classes, alpha)
    client_partitions_test = dirichlet_partition(X_test, y_test, num_classes, alpha)

    print("\n[2] Fleet-level local hdpftl_training...")
    local_models = []
    for cid in range(NUM_CLIENTS):
        model = base_model_fn.to(device)
        idx = client_partitions[cid]
        # Check class balance per client (important to avoid ValueError: target has only one class):
        labels = y_train[idx]
        if len(np.unique(labels)) < 2:
            print(f"Client {cid} has only one class. Skipping hdpftl_training or apply resampling.")
            continue

        X_val_tensor = X_train[idx].detach().clone().float()
        y_val_tensor = y_train[idx].detach().clone().long()

        trained_model = train_device_model(
            model,
            X_train[idx],
            y_train[idx],
            val_data=X_val_tensor,
            val_labels=y_val_tensor,
            epochs=20,
            lr=0.001,
            batch_size=32,
            early_stopping_patience=5,
            verbose=True
        )
        local_models.append(trained_model)

    #global_model, personalized_models = aggregate_fed_avg(local_models, base_model_fn, X_train, y_train,client_partitions)

    global_model,personalized_models = aggregate_bayesian(local_models, base_model_fn, X_train, y_train,client_partitions)

    logging.info("HDPFTL hdpftl_training and personalization completed.")
    save(global_model, personalized_models)
    logging.info("Models saved.")

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    #               EVALUATION
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    # evaluate_per_client(global_model, X_test, y_test, client_partitions_test)
    """During evaluation: Use  global model for generalization tests 
    Use personalized models to report per - client performance"""

    print("\n[6] Evaluating personalized hdpftl_models...")
    for cid, model in personalized_models.items():
        acc = evaluate_global_model(model, X_test[client_partitions_test[cid]], y_test[client_partitions_test[cid]])
        logging.info(f"Global Accuracy After Personalization for Client {cid}: {acc:.4f}")

    return global_model, personalized_models


"""
print("\n[3] Aggregating fleet models...")
    # global_model = aggregate_models(local_models, base_model_fn)
    global_model = bayesian_aggregate_models(local_models, base_model_fn)
    print("\n[4] Evaluating global model...")
    acc = evaluate_global_model(global_model, X_test, y_test)
    print(f"Global Accuracy Before Personalization: {acc:.4f}")
    logging.info(f"Global Accuracy Before Personalization: {acc:.4f}")

    evaluate_per_client(global_model, X_test, y_test, client_partitions_test)

    print("\n[5] Personalizing each client...")
    personalized_models = personalize_clients(global_model, X_train, y_train, client_partitions)

    print("\n[6] Evaluating personalized hdpftl_models...")
    for cid, model in personalized_models.items():
        acc = evaluate_global_model(model, X_test[client_partitions_test[cid]], y_test[client_partitions_test[cid]],device)
        print(f"Global Accuracy After Personalization for Client {cid}: {acc:.4f}")
        logging.info(f"Global Accuracy After Personalization for Client {cid}: {acc:.4f}")

        # print(f"Personalized Accuracy for Client {cid}: {acc:.4f}")

    return global_model, personalized_models

"""

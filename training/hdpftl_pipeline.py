from copy import deepcopy

import numpy as np
import torch
from pyro.infer import Trace_ELBO, SVI
from pyro.infer.autoguide import AutoDiagonalNormal
from torch import optim, nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from models.BayesianTabularNet import BayesianTabularNet
from utility.config import NUM_CLIENTS
from utility.utils import setup_device


def dirichlet_partition(X, y, alpha, num_classes):
    data_per_client = {i: [] for i in range(NUM_CLIENTS)}
    class_indices = [torch.where(y == i)[0] for i in range(num_classes)]

    for c in range(num_classes):
        indices = class_indices[c][torch.randperm(len(class_indices[c]))]
        proportions = np.random.dirichlet(np.repeat(alpha, NUM_CLIENTS))
        proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        split = safe_split(indices, proportions.tolist())
        for i, chunk in enumerate(split):
            data_per_client[i].extend(chunk.tolist())
    return data_per_client


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

def train_device_model(model, data, labels, epochs=3, lr=0.001, batch_size=32, verbose=False):
    model = model.to(setup_device())
    data = data.to(setup_device())
    labels = labels.to(setup_device())
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    print("Train samples:", len(data))
    print("Test samples:", len(labels))
    if len(data) > 0 and len(labels) > 0:
        loader = DataLoader(TensorDataset(data, labels), batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            running_loss = 0.0
            for x, y in loader:
                x, y = x.to(setup_device()), y.to(setup_device())
                optimizer.zero_grad()
                output = model(x)
                loss = loss_fn(output, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            if verbose:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(loader):.4f}")

    return model

def aggregate_models(models, base_model_fn):
    new_model = base_model_fn().to(setup_device())
    new_state_dict = {}
    with torch.no_grad():
        for key in models[0].state_dict().keys():
            # Average the parameters across all models
            new_state_dict[key] = torch.stack([m.state_dict()[key].float() for m in models], dim=0).mean(dim=0)

    # Load the averaged weights into the new model
    new_model.load_state_dict(new_state_dict)
    return new_model


def train_bayesian_local(X_train, y_train, input_dim, num_classes, prior_params, device='cpu'):
    model = BayesianTabularNet(input_dim, num_classes, prior_params=prior_params).to(device)
    guide = AutoDiagonalNormal(model)
    svi = SVI(model, guide, Adam({"lr": 1e-3}), loss=Trace_ELBO())

    for step in range(200):
        svi.step(X_train, y_train)

    return guide

def bayesian_aggregate_models(models, base_model_fn, epsilon=1e-8):
    """
    Bayesian model aggregation via inverse variance weighting.
    """
    n_models = len(models)
    model_keys = models[0].state_dict().keys()

    # Convert state_dicts to float tensors and collect all weights
    all_states = [{k: v.float() for k, v in m.state_dict().items()} for m in models]

    # Initialize new model and new state dict
    new_model = base_model_fn.to(setup_device())
    new_state_dict = {}

    with torch.no_grad():
        for key in model_keys:
            # Stack weights from all models: shape [n_models, *param_shape]
            stacked = torch.stack([state[key] for state in all_states], dim=0)

            # Mean and variance across models
            mean = torch.mean(stacked, dim=0)
            var = torch.var(stacked, dim=0, unbiased=False) + epsilon  # add epsilon to avoid div by 0

            # Inverse variance weighting
            weights = 1.0 / var
            weighted_sum = torch.sum(weights * stacked, dim=0)
            norm_factor = torch.sum(weights, dim=0)
            bayes_avg = weighted_sum / norm_factor

            new_state_dict[key] = bayes_avg

    new_model.load_state_dict(new_state_dict)
    return new_model


def evaluate_global_model(model, X_test, y_test, batch_size=32, device=setup_device()):
    model.eval()

    # Ensure the inputs are PyTorch tensors on the correct device
    if not torch.is_tensor(X_test):
        X_test = torch.tensor(X_test, dtype=torch.float32)
    if not torch.is_tensor(y_test):
        y_test = torch.tensor(y_test, dtype=torch.long)

    X_test = X_test.to(device)
    y_test = y_test.to(device)

    dataloader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            output = model(x)  # ✅ model(x), not model.forward()
            _, pred = torch.max(output, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = correct / total if total > 0 else 0
    print(f"Global Accuracy: {acc:.4f}")
    return acc

def evaluate_per_client(model, X, y, client_partitions, batch_size=32):
    accs = {}
    for cid, idx in client_partitions.items():
        loader = DataLoader(TensorDataset(X[idx].to(setup_device()), y[idx].to(setup_device())), batch_size=batch_size)
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, yb in loader:
                output = model(x)
                _, pred = torch.max(output, 1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        accs[cid] = correct / total if total > 0 else 0.0
        print(f"Client {cid} Accuracy: {accs[cid]:.4f}")
    return accs


def personalize_clients(global_model, X, y, client_partitions, epochs=2, batch_size=32):
    models = {}
    for cid, idx in client_partitions.items():
        local_model = deepcopy(global_model).to(setup_device())
        models[cid] = train_device_model(local_model, X[idx], y[idx] ,epochs=epochs, batch_size=batch_size)
        print(f"Personalized model trained for Client {cid}")
    return models


# --- Main HDPFTL pipeline ---

def hdpftl_pipeline(X_train, y_train, X_test, y_test, base_model_fn,alpha=0.5):
    print("\n[1] Partitioning data using Dirichlet...")

    num_classes = len(torch.unique(y_train))
    client_partitions = dirichlet_partition(X_train, y_train, alpha, num_classes)

    print("\n[2] Fleet-level local training...")
    local_models = []
    for cid in range(NUM_CLIENTS):
        model = base_model_fn.to(setup_device())
        idx = client_partitions[cid]
        trained_model = train_device_model(model, X_train[idx], y_train[idx])
        # priors = extract_priors(trained_model)
        # guide = train_bayesian_local(X_train, y_train, input_dim, num_classes, priors, device)
        # guides.append(guide)
        local_models.append(trained_model)

    print("\n[3] Aggregating fleet models...")
    #global_model = aggregate_models(local_models, base_model_fn)
    global_model = bayesian_aggregate_models(local_models, base_model_fn)
    print("\n[4] Evaluating global model...")
    evaluate_global_model(global_model, X_test, y_test)
    evaluate_per_client(global_model, X_train, y_train, client_partitions)

    print("\n[5] Personalizing each client...")
    personalized_models = personalize_clients(global_model, X_train, y_train, client_partitions)

    print("\n[6] Evaluating personalized models...")
    for cid, model in personalized_models.items():
        acc = evaluate_global_model(model, X_train[client_partitions[cid]], y_train[client_partitions[cid]],
                                    device=setup_device())
        print(f"Personalized Accuracy for Client {cid}: {acc:.4f}")

    return global_model, personalized_models

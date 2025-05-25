"""
To evaluate a trained model's accuracy on each client's local hdpftl_data.
This helps in understanding client-specific performance,
especially when hdpftl_data is non-IID (not identically distributed across clients).
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from hdpftl_utility.log import safe_log
from hdpftl_utility.utils import setup_device


# evaluate_personalized_models function! This version works for evaluate personalized models per client
def evaluate_personalized_models_per_client(personalized_models, X, y, client_partitions_test, batch_size=32):
    accs = {}
    device = setup_device()

    for cid, idx in enumerate(client_partitions_test):
        if not idx:
            accs[cid] = 0.0
            continue

        model = personalized_models[cid].to(device)
        model.eval()

        x_client = X[idx].to(device)
        y_client = y[idx].to(device)

        loader = DataLoader(TensorDataset(x_client, y_client), batch_size=batch_size)

        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in loader:
                output = model(xb)
                _, pred = torch.max(output, 1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)

        accs[cid] = correct / total if total > 0 else 0.0

    return accs


# evaluate_per_client function! This version works for evaluating a shared (global) model across all clients
# accs = evaluate_per_client(global_model, X_train, y_train, client_partitions)
"""
    Evaluate the global model on each client's test data.
"""
def evaluate_per_client(global_model, X, y, client_partitions_test, batch_size=32):
    accs = {}
    device = setup_device()
    model = global_model.to(device)
    model.eval()

    for cid, idx in enumerate(client_partitions_test):
        if not idx:  # Skip clients with no data
            accs[cid] = 0.0
            continue

        x_client = X[idx].to(device)
        y_client = y[idx].to(device)

        loader = DataLoader(TensorDataset(x_client, y_client), batch_size=batch_size)

        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in loader:
                output = model(xb)
                _, pred = torch.max(output, 1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)

        accs[cid] = correct / total if total > 0 else 0.0
        safe_log(f"Client {cid} Accuracy for Global Model: {accs[cid]:.4f}")

    return accs

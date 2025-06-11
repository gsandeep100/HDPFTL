"""
To evaluate a trained model's accuracy on each client's local hdpftl_data.
This helps in understanding client-specific performance,
especially when hdpftl_data is non-IID (not identically distributed across clients).
"""

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from hdpftl_training.hdpftl_models.TabularNet import create_model_fn
from hdpftl_utility.config import BATCH_SIZE, PERSONALISED_MODEL_PATH_TEMPLATE, NUM_CLIENTS
from hdpftl_utility.log import safe_log
from hdpftl_utility.utils import setup_device


def load_personalized_models_fromfile():
    """
    Loads personalized models for each client from disk.

    Returns:
        Dict[int, nn.Module]: Mapping from client_id to their loaded model (in eval mode).
    """
    personalized_models = {}
    device = setup_device()

    for cid in range(NUM_CLIENTS):
        model = create_model_fn().to(device)  # New model for each client
        model_path = PERSONALISED_MODEL_PATH_TEMPLATE.substitute(n=cid)

        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=device)

                if isinstance(state_dict, dict):
                    model.load_state_dict(state_dict)
                    model.eval()
                    personalized_models[cid] = model
                    safe_log(f"✅ Loaded model for client {cid} from {model_path}")
                else:
                    safe_log(f"⚠️ Unexpected format in model {cid}: state_dict not found.", level="error")
            except Exception as e:
                safe_log(f"❌ Error loading model for client {cid}: {e}", level="error")
        else:
            safe_log(f"❌ Model file not found for client {cid}: {model_path}", level="error")

    return personalized_models


# evaluate_personalized_models function! This version works for evaluate personalized models per client
def evaluate_personalized_models_per_client(personalized_models, client_data_dict_test):
    accs = {}
    device = setup_device()

    for client_id, client_data in client_data_dict_test.items():
        if not client_data or len(client_data) != 2:
            safe_log(f"Client '{client_id}' has invalid test data. Assigning accuracy 0.0.", level="warning")
            accs[client_id] = 0.0
            continue

        x_client_np, y_client_np = client_data

        if len(x_client_np) == 0:
            safe_log(f"Client '{client_id}' has no actual data. Assigning accuracy 0.0.", level="warning")
            accs[client_id] = 0.0
            continue

        if client_id not in personalized_models:
            safe_log(f"No model found for client '{client_id}'. Skipping.", level="error")
            accs[client_id] = 0.0
            continue

        safe_log(f"Evaluating client: '{client_id}'")

        # Handle model or state_dict
        if isinstance(personalized_models[client_id], dict):
            model = create_model_fn(x_client_np.shape[1], len(np.unique(y_client_np))).to(device)
            model.load_state_dict(personalized_models[client_id])
        else:
            model = personalized_models[client_id].to(device)

        model.eval()

        x_tensor = torch.tensor(x_client_np).float().to(device)
        y_tensor = torch.tensor(y_client_np).long().to(device)

        loader = DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=BATCH_SIZE)

        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in loader:
                output = model(xb)
                _, pred = torch.max(output, 1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)

        accs[client_id] = correct / total if total > 0 else 0.0
        safe_log(f"  Accuracy for client '{client_id}': {accs[client_id]:.4f}")

    return accs


# evaluate_per_client function! This version works for evaluating a shared (global) model across all clients
# accs = evaluate_per_client(global_model, X_train, y_train, client_partitions)
"""
    Evaluate the global model on each client's test data.
"""


# Client accuracy for Global model
def evaluate_per_client(global_model, client_partitions_test):
    """
    Evaluates a global model's performance on per-client test datasets.

    Args:
        global_model (torch.nn.Module): The model to evaluate.
        client_partitions_test (dict): client_id -> (X_client, y_client) as numpy arrays or tensors.

    Returns:
        dict: client_id -> accuracy
    """
    accs = {}
    device = setup_device()
    model = global_model.to(device)
    model.eval()

    for cid, client_data in client_partitions_test.items():
        if not isinstance(client_data, (tuple, list)) or len(client_data) != 2:
            safe_log(f"Client '{cid}' has invalid test data. Assigning accuracy 0.0.", level="warning")
            accs[cid] = 0.0
            continue

        X_client, y_client = client_data

        if len(X_client) == 0:
            safe_log(f"Client '{cid}' has no data. Skipping.", level="warning")
            accs[cid] = 0.0
            continue

        # Convert to torch tensors if not already
        X_tensor = torch.tensor(X_client, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y_client, dtype=torch.long).to(device)

        loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=BATCH_SIZE)

        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in loader:
                output = model(xb)
                pred = output.argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)

        accs[cid] = correct / total if total > 0 else 0.0
        safe_log(f"✅ Client {cid} accuracy: {accs[cid]:.4f}")

    return accs

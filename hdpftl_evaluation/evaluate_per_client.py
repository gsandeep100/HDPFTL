"""
To evaluate a trained model's accuracy on each client's local hdpftl_data.
This helps in understanding client-specific performance,
especially when hdpftl_data is non-IID (not identically distributed across clients).
"""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from hdpftl_training.hdpftl_models.TabularNet import create_model_fn_personalized
from hdpftl_utility.config import BATCH_SIZE, PERSONALISED_MODEL_PATH_TEMPLATE, NUM_CLIENTS
from hdpftl_utility.log import safe_log
from hdpftl_utility.utils import setup_device


def load_personalized_models_fromfile():
    """
    Loads personalized models for each client from disk.

    Returns:
        List[nn.Module]: List of models, one per client, set to eval mode.
    """
    personalized_models = []
    device = setup_device()
    try:
        model = create_model_fn_personalized().to(device)

        for cid in range(NUM_CLIENTS):
            model_path = PERSONALISED_MODEL_PATH_TEMPLATE.substitute(n=cid)

            if os.path.exists(model_path):
                try:
                    state_dict = torch.load(model_path, map_location=device)

                    if isinstance(state_dict, dict):
                        model.load_state_dict(state_dict)
                        print(f"✅ Loaded model for client {cid} from {model_path}")
                    else:
                        print(f"⚠️ Unexpected format: {model_path} does not contain a valid state_dict")
                except Exception as e:
                    print(f"❌ Failed to load model for client {cid} due to error: {e}")
            else:
                print(f"❌ Model file not found for client {cid}: {model_path}")

        model.eval()
        personalized_models.append(model)
    except Exception as e:
        print("❌ Failed to load personal model.")
        print(f"Error: {e}")
        return None
    return personalized_models


# evaluate_personalized_models function! This version works for evaluate personalized models per client
def evaluate_personalized_models_per_client(personalized_models, X, y, client_data_dict_test):
    """
    Evaluates personalized models for each client.

    Args:
        personalized_models (dict): Dict of client_id -> model.
        X (numpy.ndarray): Global features dataset.
        y (numpy.ndarray): Global labels dataset.
        client_data_dict_test (dict): client_id -> list of indices into X and y.

    Returns:
        dict: client_id -> accuracy
    """
    accs = {}
    device = setup_device()

    for client_id, client_indices in client_data_dict_test.items():
        # 1. Handle invalid or empty indices
        if not isinstance(client_indices, (list, np.ndarray)) or len(client_indices) == 0:
            print(f"Warning: Client '{client_id}' has no test data. Assigning accuracy 0.0.")
            accs[client_id] = 0.0
            continue

        # 2. Check if model exists for client
        if client_id not in personalized_models:
            print(f"Error: No model found for client '{client_id}'. Skipping.")
            accs[client_id] = 0.0
            continue

        print(f"Evaluating client: '{client_id}'")

        model = personalized_models[client_id].to(device)
        model.eval()

        x_client_np = X[client_indices]
        y_client_np = y[client_indices]

        if len(x_client_np) == 0:
            print(f"Warning: Client '{client_id}' has no actual data after indexing. Assigning accuracy 0.0.")
            accs[client_id] = 0.0
            continue

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
        print(f"  Accuracy for client '{client_id}': {accs[client_id]:.4f}")

    return accs


# evaluate_per_client function! This version works for evaluating a shared (global) model across all clients
# accs = evaluate_per_client(global_model, X_train, y_train, client_partitions)
"""
    Evaluate the global model on each client's test data.
"""


# Client accuracy for Global model
def evaluate_per_client(global_model, X, y, client_partitions_test):
    """
    Evaluates a global model's performance on client-specific test data.

    Args:
        global_model (torch.nn.Module): The PyTorch model to evaluate.
        X (np.ndarray or pd.DataFrame): The full dataset features.
            Expected to be a NumPy array or convertible to one.
        y (np.ndarray or pd.DataFrame or pd.Series): The full dataset labels.
            Expected to be a NumPy array or convertible to one (1D for labels).
        client_partitions_test (dict or list): A dictionary where keys are client IDs
            and values are lists/arrays of indices corresponding to that client's
            data in X and y. Or, if `enumerate` is used on a list, it should be
            a list of index arrays.

    Returns:
        dict: A dictionary of accuracies for each client.
    """
    accs = {}
    device = setup_device()

    # Ensure the global_model is on the correct device and in evaluation mode
    model = global_model.to(device)
    model.eval()  # Important: Sets the model to evaluation mode (disables dropout, batch norm updates, etc.)

    # --- Data Type Conversion to NumPy Arrays (One-time, before the loop) ---
    # Convert X to a NumPy array if it's a Pandas DataFrame
    if isinstance(X, pd.DataFrame):
        X_np = X.values
        safe_log("Converted X from pandas.DataFrame to numpy.ndarray.")
    elif isinstance(X, np.ndarray):
        X_np = X
    else:
        # Handle other types if necessary, or raise an error
        raise TypeError(f"Unsupported type for X: {type(X)}. Expected numpy.ndarray or pandas.DataFrame.")

    # Convert y to a 1D NumPy array if it's a Pandas DataFrame/Series
    if isinstance(y, pd.DataFrame):
        # If it's a single column DataFrame, flatten it to 1D
        if y.shape[1] == 1:
            y_np = y.values.flatten()
            safe_log("Converted y from single-column pandas.DataFrame to 1D numpy.ndarray.")
        else:
            raise ValueError("y (labels) should be a single-column DataFrame or Series.")
    elif isinstance(y, pd.Series):
        y_np = y.values
        safe_log("Converted y from pandas.Series to numpy.ndarray.")
    elif isinstance(y, np.ndarray):
        # Ensure y is 1D for classification labels if it's a NumPy array
        if y.ndim > 1 and y.shape[1] == 1:
            y_np = y.flatten()
            safe_log("Flattened 2D y numpy.ndarray to 1D.")
        else:
            y_np = y
    else:
        raise TypeError(
            f"Unsupported type for y: {type(y)}. Expected numpy.ndarray, pandas.DataFrame, or pandas.Series.")
    # --- End Data Type Conversion ---

    for cid, idx in enumerate(client_partitions_test):
        # Skip clients with no data (empty index list/array)
        if not (isinstance(idx, (list, np.ndarray)) and len(idx) > 0):
            accs[cid] = 0.0
            safe_log(f"Client {cid} has no valid data indices, accuracy set to 0.0.")
            continue

        # Extract client-specific data using the prepared NumPy arrays
        # Then convert these NumPy subarrays to PyTorch tensors and move to device
        x_client_tensor = torch.tensor(X_np[idx]).float().to(device)
        y_client_tensor = torch.tensor(y_np[idx]).long().to(device)  # Labels are typically long/int64

        # Create DataLoader for the current client's test set
        loader = DataLoader(TensorDataset(x_client_tensor, y_client_tensor), BATCH_SIZE)

        correct, total = 0, 0
        with torch.no_grad():  # Disable gradient calculations for evaluation
            for xb, yb in loader:
                # xb and yb are already on the correct device because x_client_tensor/y_client_tensor were
                output = model(xb)
                # Get the predicted class (index of the max log-probability)
                _, pred = torch.max(output, 1)
                # Accumulate correct predictions and total samples
                correct += (pred == yb).sum().item()
                total += yb.size(0)

        # Calculate accuracy for the current client
        accs[cid] = correct / total if total > 0 else 0.0
        safe_log(f"Client {cid} Accuracy for Global Model: {accs[cid]:.4f}")

    return accs

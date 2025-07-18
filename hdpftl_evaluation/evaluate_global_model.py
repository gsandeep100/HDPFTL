import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

import hdpftl_utility.config as config
import hdpftl_utility.log as log_util
import hdpftl_utility.utils as util

from hdpftl_training.hdpftl_models.TabularNet import create_model_fn, TabularNet


def evaluate_global_model_fromfile(base_model_fn):
    """
    Loads the global model from file and returns it. Does not evaluate on any test data.

    Returns:
        nn.Module or None: Loaded model in eval mode, or None if loading fails.
    """
    device = util.setup_device()

    try:
        # Instantiate the model
        global_model = create_model_fn().to(device)

        # Allowlist custom model class for safe unpickling
        torch.serialization.add_safe_globals([TabularNet])

        # Load weights
        global_model.load_state_dict(
            torch.load(config.GLOBAL_MODEL_PATH_TEMPLATE.substitute(n=util.get_today_date()), map_location=device))
        global_model.eval()

        log_util.safe_log("✅ Global model loaded and ready for evaluation.")
        return global_model

    except Exception as e:
        log_util.safe_log("❌ Failed to load global model.")
        log_util.safe_log(f"Error: {e}", level="error")
        return None


def evaluate_global_model(model, X_test, y_test):
    """
    Evaluates the performance of a global PyTorch model on a test dataset.

    Args:
        model (torch.nn.Module): The PyTorch model to evaluate.
        X_test (np.ndarray, pd.DataFrame, or torch.Tensor): The test features.
        y_test (np.ndarray, pd.Series, or torch.Tensor): The test labels.

    Returns:
        float: The accuracy of the global model on the test set.
    """
    log_util.safe_log("[4] Evaluating global model...")
    device = util.setup_device()

    # Ensure model is on correct device and in eval mode
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode (disables dropout, batch norm updates, etc.)

    # --- Robust Input Conversion to PyTorch Tensors ---
    # Convert X_test to a PyTorch tensor if it's not already
    if isinstance(X_test, np.ndarray):
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    elif isinstance(X_test, pd.DataFrame):
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)  # Convert DataFrame to NumPy first
    elif torch.is_tensor(X_test):
        X_test_tensor = X_test.to(torch.float32)  # Ensure correct dtype
    else:
        raise TypeError(
            f"Unsupported type for X_test: {type(X_test)}. Expected numpy.ndarray, pandas.DataFrame, or torch.Tensor.")

    # Convert y_test to a PyTorch tensor if it's not already
    if isinstance(y_test, np.ndarray):
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    elif isinstance(y_test, pd.Series):
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)  # Convert Series to NumPy first
    elif isinstance(y_test, pd.DataFrame):  # Handle single-column DataFrames for labels
        if y_test.shape[1] == 1:
            y_test_tensor = torch.tensor(y_test.values.flatten(), dtype=torch.long)
        else:
            raise ValueError("y_test (labels) DataFrame should have a single column.")
    elif torch.is_tensor(y_test):
        y_test_tensor = y_test.to(torch.long)  # Ensure correct dtype
    else:
        raise TypeError(
            f"Unsupported type for y_test: {type(y_test)}. Expected numpy.ndarray, pandas.Series, pandas.DataFrame, or torch.Tensor.")

    # Move tensors to the specified device (CPU/GPU)
    X_test_on_device = X_test_tensor.to(device)
    y_test_on_device = y_test_tensor.to(device)

    # --- End Robust Input Conversion ---

    assert len(X_test_on_device) == len(y_test_on_device), "Mismatched test data and labels lengths after conversion."

    # Create DataLoader. Using a default batch size as it's not provided,
    # or you can make BATCH_SIZE a parameter or global variable.
    # For global evaluation, often a single batch or large batch size is used.
    test_batch_size = 256  # Or define globally like BATCH_SIZE
    dataloader = DataLoader(TensorDataset(X_test_on_device, y_test_on_device), batch_size=test_batch_size,
                            pin_memory=False)

    correct, total = 0, 0
    with torch.no_grad():  # Disable gradient calculations for inference
        for x_batch, y_batch in dataloader:
            # x_batch and y_batch are already on the correct device due to DataLoader's input
            output = model(x_batch)
            _, pred = torch.max(output, 1)  # Get the predicted class index
            correct += (pred == y_batch).sum().item()  # Count correct predictions
            total += y_batch.size(0)  # Accumulate total samples

    acc = correct / total if total > 0 else 0.0
    log_util.safe_log(f"Global Accuracy: {acc:.4f}")  # Corrected log message to include "Accuracy"

    # The original return statement was 'return ACC' which was not defined.
    # Assuming you want to return the calculated accuracy 'acc'.
    return acc

import logging

import torch
from torch.utils.data import DataLoader, TensorDataset

from hdpftl_utility.utils import setup_device


# acc = evaluate_global_model(global_model, X_test, y_test)
def evaluate_global_model(model, X_test, y_test, batch_size=32):
    logging.info("\n[4] Evaluating global model...")
    device = setup_device()

    # Ensure model is on correct device and in eval mode
    model = model.to(device)
    model.eval()

    # Ensure inputs are tensors
    if not torch.is_tensor(X_test):
        X_test = torch.tensor(X_test, dtype=torch.float32)
    if not torch.is_tensor(y_test):
        y_test = torch.tensor(y_test, dtype=torch.long)

    assert len(X_test) == len(y_test), "Mismatched test data and labels"

    with torch.no_grad():
        X_test = X_test.to(device)
        y_test = y_test.to(device)
        assert len(X_test) == len(y_test), "Mismatch in number of samples and labels"
        dataloader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)
        correct, total = 0, 0

        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            output = model(x)
            _, pred = torch.max(output, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = correct / total if total > 0 else 0.0
    return acc

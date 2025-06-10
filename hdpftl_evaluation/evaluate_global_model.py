import torch
from torch.utils.data import DataLoader, TensorDataset

from hdpftl_training.hdpftl_models.TabularNet import create_model_fn_global
from hdpftl_utility.config import GLOBAL_MODEL_PATH
from hdpftl_utility.log import safe_log
from hdpftl_utility.utils import setup_device


def evaluate_global_model_fromfile():
    device = setup_device()

    try:
        # Instantiate architecture
        global_model = create_model_fn_global().to(device)

        # Load saved weights
        global_model.load_state_dict(torch.load(GLOBAL_MODEL_PATH, map_location=device))
        global_model.eval()
        print("✅ Global model loaded and ready for evaluation.")
        return global_model

    except Exception as e:
        print("❌ Failed to load global model.")
        print(f"Error: {e}")
        return None


def evaluate_global_model(model, X_test, y_test):
    safe_log("[4] Evaluating global model...")
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
        dataloader = DataLoader(TensorDataset(X_test, y_test))
        correct, total = 0, 0

        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            output = model(x)
            _, pred = torch.max(output, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = correct / total if total > 0 else 0.0
    safe_log(f"Global Accuracy{acc:.4f}")

    return acc

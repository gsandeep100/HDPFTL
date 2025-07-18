import torch
import torch.nn.functional as F

import hdpftl_utility.utils as util

def predict(new_data, global_model, label_map=None, return_proba=False, device=None):
    """
    Predicts classes (and optionally probabilities) using a trained PyTorch model.

    Args:
        new_data (array-like or tensor): Input hdpftl_data of shape (n_samples, n_features).
        global_model (torch.nn.Module): Trained model.
        label_map (dict, optional): Mapping from class indices to labels.
        return_proba (bool): Whether to return class probabilities.
        device (torch.device or None): Device to run on. Auto-detects if None.

    Returns:
        predictions (List[str or int]): Predicted labels or indices.
        probabilities (List[List[float]], optional): If return_proba=True, class probabilities.
    """
    # Auto-select device
    if device is None:
        device = setup_device()

    # Ensure input is a float32 tensor
    if not isinstance(new_data, torch.Tensor):
        new_data = torch.tensor(new_data, dtype=torch.float32)

    # Reshape if single sample (1D)
    if new_data.dim() == 1:
        new_data = new_data.unsqueeze(0)

    new_data = new_data.to(device)
    global_model = global_model.to(device)
    global_model.eval()

    with torch.no_grad():
        outputs = global_model(new_data)
        probs = F.softmax(outputs, dim=1)
        pred_indices = torch.argmax(probs, dim=1)

    # Convert to label names if label_map provided
    if label_map:
        predictions = [label_map[int(idx)] for idx in pred_indices]
    else:
        predictions = pred_indices.cpu().tolist()

    if return_proba:
        return predictions, probs.cpu().tolist()
    else:
        return predictions

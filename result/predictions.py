import torch

import predictions
from utility.utils import setup_device


def predict(new_data, global_model):
    # Preprocess new_data if not already
    if not torch.is_tensor(new_data):
        new_data = torch.tensor(new_data, dtype=torch.float32)

    # Move to device
    new_data = new_data.to(setup_device())
    global_model.eval()

    with torch.no_grad():
        outputs = global_model(new_data)
        predictions = torch.argmax(outputs, dim=1)


print("Predictions:", predictions.cpu().numpy())

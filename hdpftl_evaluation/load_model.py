import torch

from hdpftl_utility.utils import setup_device


def load_global_model(base_model_fn, path):
    """
    Loads a global model from a .pth file using a factory function.

    Args:
        base_model_fn (function): A function that returns an instance of the model.
        path (str): Path to the .pth file.

    Returns:
        nn.Module: The loaded model in eval mode on the correct device.
    """
    device = setup_device()

    # Instantiate the model
    global_model = base_model_fn().to(device)

    # Load state dict with device map
    global_model = torch.load(path, map_location=device, weights_only=False)

    # Set to evaluation mode
    global_model.eval()

    return global_model


def load_personalized_model(base_model_fn, path):
    """
    Loads a personalized model for a client.

    Args:
        base_model_fn (function): Returns an instance of the model.
        path (str): Path to the personalized model's .pth file.

    Returns:
        nn.Module: The loaded model in eval mode.
    """
    device = setup_device()
    model = base_model_fn().to(device)
    model.load_state_dict(torch.load(path, map_location=device), weights_only=False)
    model.eval()
    return model

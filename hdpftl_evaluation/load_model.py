import torch


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

    # Load the state dict
    state_dict = torch.load(path, map_location=device)

    # If you only saved the weights
    if isinstance(state_dict, dict):
        global_model.load_state_dict(state_dict)
    else:
        raise ValueError("Loaded file is not a state_dict. Use torch.save(model.state_dict(), path) when saving.")

    # Set to eval mode
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

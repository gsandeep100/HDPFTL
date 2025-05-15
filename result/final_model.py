# Save global model
import torch

from utility.utils import setup_device


def save(global_model, personalized_models):
    torch.save(global_model.state_dict(), "global_model.pth")

    # Save personalized models (if you have them)
    for i, model in enumerate(personalized_models):
        torch.save(model.state_dict(), f"personalized_model_client_{i}.pth")


def load(base_model_fn):
    # Load global model
    loaded_model = base_model_fn().to(setup_device())
    loaded_model.load_state_dict(torch.load("global_model.pth"))
    loaded_model.eval()

# Save global model
import torch

from utility.utils import setup_device


def save(global_model, personalized_models):
    torch.save(global_model.state_dict(), "./trained-models/global_model.pth")

    # Save personalized models (if you have them)
    for cid, model in personalized_models.items():
        # print(f"Client {cid} -> type: {type(entry)}, content: {entry}")
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"Expected a model at index {i}, but got {type(model)}")
        torch.save(model.state_dict(), f"./trained-models/personalized_model_client_{cid}.pth")


def load(base_model_fn):
    # Load global model
    loaded_model = base_model_fn().to(setup_device())
    loaded_model.load_state_dict(torch.load("./trained-models/global_model.pth"))
    loaded_model.eval()

# Save global model
import torch

from hdpftl_utility.utils import setup_device


def save(global_model, personalized_models):
    torch.save(global_model, "./trained-hdpftl_models/global_model.pth")

    # Save personalized hdpftl_models (if you have them)
    for i, (cid, model) in enumerate(personalized_models.items()):
        # print(f"Client {cid} -> type: {type(entry)}, content: {entry}")
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"Expected a model at index {i}, but got {type(model)}")
        torch.save(model.state_dict(), f"./trained-hdpftl_models/personalized_model_client_{cid}.pth")

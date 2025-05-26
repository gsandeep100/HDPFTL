# Save global model
import torch


def save(global_model, personalized_models):
    torch.save(global_model, "./hdpftl_trained_models/global_model.pth")

    # Save personalized hdpftl_models (if you have them)
    for i, (cid, model) in enumerate(personalized_models.items()):
        # print(f"Client {cid} -> type: {type(entry)}, content: {entry}")
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"Expected a model at index {i}, but got {type(model)}")
        torch.save(model.state_dict(), f"./hdpftl_trained_models/personalized_model_client_{cid}.pth")

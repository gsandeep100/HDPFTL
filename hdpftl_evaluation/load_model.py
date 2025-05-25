import torch

from hdpftl_utility.utils import setup_device

def load(base_model_fn):
    # Load global model
    loaded_model = base_model_fn().to(setup_device())
    loaded_model.load_state_dict(torch.load("./trained-hdpftl_models/global_model.pth"))
    loaded_model.eval()
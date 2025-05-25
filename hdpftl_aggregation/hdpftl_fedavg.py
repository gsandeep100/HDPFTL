import torch

from hdpftl_personalised_client.personalize_clients import personalize_clients
from hdpftl_utility.log import safe_log
from hdpftl_utility.utils import setup_device


def aggregate_models(models, base_model_fn):
    device = setup_device()
    new_model = base_model_fn.to(device)
    new_state_dict = {}
    with torch.no_grad():
        for key in models[0].state_dict().keys():
            # Average the parameters across all hdpftl_models
            new_state_dict[key] = torch.stack([m.state_dict()[key].float() for m in models], dim=0).mean(dim=0)

    # Load the averaged weights into the new model
    new_model.load_state_dict(new_state_dict)
    return new_model


def aggregate_fed_avg(local_models, base_model_fn, X_train, y_train, client_partitions):
    global_model = aggregate_models(local_models, base_model_fn)
    safe_log("[6] Aggregated FedAvg fleet hdpftl_models...")

    personalized_models = personalize_clients(global_model, X_train, y_train, client_partitions)
    safe_log("[7] Personalizing each client...")

    return global_model, personalized_models

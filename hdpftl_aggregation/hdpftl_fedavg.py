import logging

import torch

from hdpftl_personalised_client.personalize_clients import personalize_clients
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
    logging.info("\n[3] Aggregating FedAvg leet hdpftl_models...")
    global_model = aggregate_models(local_models, base_model_fn)

    logging.info("\n[5] Personalizing each client...")
    personalized_models = personalize_clients(global_model, X_train, y_train, client_partitions)
    return global_model, personalized_models

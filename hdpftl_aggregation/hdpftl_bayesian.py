import logging

import torch

from hdpftl_personalised_client.personalize_clients import personalize_clients
from hdpftl_utility.utils import setup_device


def aggregate_bayesian(local_models, base_model_fn, X_train, y_train, client_partitions):
    logging.info("\n[3] Aggregating Bayesian fleet hdpftl_models...")
    global_model = hdpftl_bayesian(local_models, base_model_fn)

    logging.info("\n[5] Personalizing each client...")
    personalized_models = personalize_clients(global_model, X_train, y_train, client_partitions)
    return personalized_models, global_model


def hdpftl_bayesian(models, base_model_fn, epsilon=1e-8):
    """
    Perform Bayesian aggregation of multiple PyTorch hdpftl_models using inverse variance weighting.

    Args:
        models (List[torch.nn.Module]): List of trained hdpftl_models with identical architectures.
        base_model_fn (torch.nn.Module): An untrained model instance (used for structure).
        epsilon (float): Small constant to prevent division by zero in variance.

    Returns:
        torch.nn.Module: Aggregated model with Bayesian-weighted parameters.
    """
    device = setup_device()
    n_models = len(models)
    if n_models == 0:
        raise ValueError("Model list is empty.")

    model_keys = models[0].state_dict().keys()

    # Convert state_dicts to float tensors
    all_states = [{k: v.float().to(device) for k, v in m.state_dict().items()} for m in models]

    # Initialize new model
    new_model = base_model_fn.to(device)
    new_state_dict = {}

    with torch.no_grad():
        for key in model_keys:
            stacked = torch.stack([state[key] for state in all_states], dim=0)

            mean = torch.mean(stacked, dim=0)
            var = torch.var(stacked, dim=0, unbiased=False) + epsilon

            weights = 1.0 / var
            weighted_sum = torch.sum(weights * stacked, dim=0)
            norm_factor = torch.sum(weights, dim=0)
            bayes_avg = weighted_sum / norm_factor

            new_state_dict[key] = bayes_avg

    new_model.load_state_dict(new_state_dict)
    return new_model

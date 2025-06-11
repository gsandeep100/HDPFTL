import copy

from hdpftl_training.hdpftl_personalised_client.personalize_clients import personalize_clients
from hdpftl_utility.log import safe_log


def aggregate_models(model_state_dicts, base_model_fn):
    """
    Aggregate multiple model state_dicts by averaging parameters (FedAvg).

    Args:
        model_state_dicts (list of dict): List of model.state_dict() to average.
        base_model_fn (callable): Function returning a fresh model instance.

    Returns:
        dict: Aggregated state_dict.
    """
    if not model_state_dicts:
        return base_model_fn().state_dict()

    # Deep copy and clone tensors to avoid modifying input state_dicts
    aggregated_state = {}
    for key in model_state_dicts[0].keys():
        aggregated_state[key] = model_state_dicts[0][key].clone()

    # Sum remaining models' parameters
    for state_dict in model_state_dicts[1:]:
        for key in aggregated_state:
            aggregated_state[key] += state_dict[key]

    # Average
    for key in aggregated_state:
        aggregated_state[key] /= len(model_state_dicts)

    # Load averaged weights into fresh model instance (optional)
    aggregated_model = base_model_fn()
    aggregated_model.load_state_dict(aggregated_state)

    return aggregated_model.state_dict()


def aggregate_fed_avg(local_models, base_model_fn, X_train, y_train, client_partitions):
    global_model = aggregate_models(local_models, base_model_fn)
    safe_log("[6] Aggregated FedAvg fleet hdpftl_models...")

    personalized_models = personalize_clients(global_model, X_train, y_train, client_partitions)
    safe_log("[7] Personalizing each client...")

    return global_model, personalized_models

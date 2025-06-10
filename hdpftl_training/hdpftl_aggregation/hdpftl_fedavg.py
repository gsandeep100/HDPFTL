import copy

from hdpftl_training.hdpftl_personalised_client.personalize_clients import personalize_clients
from hdpftl_utility.log import safe_log


def aggregate_models(model_state_dicts, base_model_fn):
    # This is a simplified placeholder for FedAvg.
    # In a real scenario, you'd calculate weighted averages of parameters.
    if not model_state_dicts:
        return base_model_fn().state_dict()  # Return a fresh model's state_dict if nothing to aggregate

    # Initialize aggregated state dict with zeros or the first model's state
    aggregated_state = copy.deepcopy(model_state_dicts[0])
    for key in aggregated_state:
        # Sum up weights from all models for each parameter
        for i in range(1, len(model_state_dicts)):
            aggregated_state[key] += model_state_dicts[i][key]
        # Average the weights
        aggregated_state[key] /= len(model_state_dicts)

    # Instantiate a new model and load the aggregated state
    aggregated_model = base_model_fn()
    aggregated_model.load_state_dict(aggregated_state)
    return aggregated_model.state_dict()  # Crucial: return the state_dict


def aggregate_fed_avg(local_models, base_model_fn, X_train, y_train, client_partitions):
    global_model = aggregate_models(local_models, base_model_fn)
    safe_log("[6] Aggregated FedAvg fleet hdpftl_models...")

    personalized_models = personalize_clients(global_model, X_train, y_train, client_partitions)
    safe_log("[7] Personalizing each client...")

    return global_model, personalized_models

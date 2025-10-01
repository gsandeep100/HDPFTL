# Save global model
import os

import torch

import hdpftl_utility.config as config
import hdpftl_utility.log as log_util
import hdpftl_utility.utils as util


def save(global_model, personalized_models):
    """
    Saves the global model and all personalized models.

    Args:
        global_model (torch.nn.Module or dict): The global model (can be a full model or its state_dict).
        personalized_models (dict): A dictionary mapping client IDs to their personalized
                                    model state_dicts (OrderedDicts).
    """
    # Ensure the directory for personalized models exists
    os.makedirs(config.TRAINED_MODEL_DIR, exist_ok=True)  # Creates the directory if it doesn't exist

    # --- Save the global model ---
    # It's best practice to save only the state_dict, not the entire model object.
    if isinstance(global_model, torch.nn.Module):
        torch.save(global_model.state_dict(), config.GLOBAL_MODEL_PATH_TEMPLATE.substitute(n=util.get_today_date()))
        log_util.safe_log(
            f"✅ Global model state_dict saved to {config.GLOBAL_MODEL_PATH_TEMPLATE.substitute(n=util.get_today_date())}")
    elif isinstance(global_model, dict):  # If global_model is already a state_dict from aggregation
        torch.save(global_model, config.GLOBAL_MODEL_PATH_TEMPLATE.substitute(n=get_today_date()))
        log_util.safe_log(
            f"✅ Global model (state_dict) saved to {config.GLOBAL_MODEL_PATH_TEMPLATE.substitute(n=util.get_today_date())}")
    else:
        log_util.safe_log(f"❌Unexpected type for global_model ({type(global_model)}). Not saving global model.",
                          level="warning")

    # --- Save personalized models ---
    log_util.safe_log(f"Saving {len(personalized_models)} personalized models...")
    for i, (cid, model_data) in enumerate(personalized_models.items()):
        # model_data is expected to be an OrderedDict (the state_dict)
        if isinstance(model_data, dict):  # Check if it's a dictionary (which OrderedDict is)
            save_path = os.path.join(config.TRAINED_MODEL_DIR + util.get_today_date() + "/",
                                     f"personalized_model_client_{cid}.pth")
            torch.save(model_data, save_path)
            log_util.safe_log(f"  Saved personalized model state_dict for client {cid} to {save_path}")
        else:
            # This block will catch if something unexpected gets into personalized_models
            log_util.safe_log(
                f"  ❌Expected a state_dict (dict) for client {cid}, but got {type(model_data)}. Skipping save for this client.",
                level="warning")

    log_util.safe_log("✅ All HDPFTL models saved successfully.")

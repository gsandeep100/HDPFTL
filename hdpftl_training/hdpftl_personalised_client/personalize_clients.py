from copy import deepcopy

import hdpftl_training.train_device_model as train_device_model
import hdpftl_utility.config as config
import hdpftl_utility.utils as util


def personalize_clients(global_model, X, y, client_partitions, epochs=2):
    models = {}
    device = util.setup_device()
    for cid, idx in enumerate(client_partitions):
        local_model = deepcopy(global_model).to(device)
        models[cid] = train_device_model.train_device_model(local_model, X[idx], y[idx], config.BATCH_SIZE,
                                                            epochs=epochs)
    return models

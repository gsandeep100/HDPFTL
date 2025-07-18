from copy import deepcopy

from hdpftl_training.train_device_model import train_device_model


def personalize_clients(global_model, X, y, client_partitions, epochs=2):
    models = {}
    device = setup_device()
    for cid, idx in enumerate(client_partitions):
        local_model = deepcopy(global_model).to(device)
        models[cid] = train_device_model(local_model, X[idx], y[idx], BATCH_SIZE, epochs=epochs)
    return models

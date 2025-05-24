from copy import deepcopy

from hdpftl_training.train_device_model import train_device_model
from hdpftl_utility.utils import setup_device


def personalize_clients(global_model, X, y, client_partitions, epochs=2, batch_size=32):
    models = {}
    device = setup_device()
    for cid, idx in enumerate(client_partitions):
        local_model = deepcopy(global_model).to(device)
        models[cid] = train_device_model(local_model, X[idx], y[idx], epochs=epochs, batch_size=batch_size)
        # print(f"Personalized model trained for Client {cid}")
    return models

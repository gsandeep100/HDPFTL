import copy
import torch
import torch.nn as nn
import torch.optim as optim

# --------------------------------------
# 1. Device Layer Methods
# --------------------------------------

def train_fedavg_device(model, device_data, epochs=1, lr=0.01):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for x, y in device_data:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
    return model.state_dict()


def train_fedprox_device(model, global_params, device_data, mu=0.01, epochs=1, lr=0.01):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for x, y in device_data:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            # Proximal term
            prox_loss = 0.0
            for name, param in model.named_parameters():
                prox_loss += ((param - global_params[name]) ** 2).sum()
            loss += (mu / 2) * prox_loss
            loss.backward()
            optimizer.step()
    return model.state_dict()


def train_fedper_device(model, device_data, shared_layers, private_layers, epochs=1, lr=0.01):
    optimizer = optim.SGD(list(shared_layers.parameters()) + list(private_layers.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for x, y in device_data:
            optimizer.zero_grad()
            out = shared_layers(x)
            out = private_layers(out)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
    # Return both shared + private params
    return {**shared_layers.state_dict(), **private_layers.state_dict()}


# --------------------------------------
# 2. Edge Layer Methods
# --------------------------------------

def aggregate_fedavg_edge(device_models):
    """Simple averaging of device models"""
    avg_model = copy.deepcopy(device_models[0])
    for key in avg_model.keys():
        for dm in device_models[1:]:
            avg_model[key] += dm[key]
        avg_model[key] = avg_model[key] / len(device_models)
    return avg_model


def aggregate_fedprox_edge(device_models, global_params, mu=0.01):
    """FedProx aggregation at edge"""
    prox_model = copy.deepcopy(device_models[0])
    for key in prox_model.keys():
        prox_model[key] = sum(dm[key] for dm in device_models) / len(device_models)
        # Apply proximal adjustment
        prox_model[key] = prox_model[key] + mu * (global_params[key] - prox_model[key])
    return prox_model


def aggregate_fedper_edge(device_models, shared_keys, private_keys):
    """Edge aggregation keeps shared layers averaged, private layers untouched"""
    edge_model = copy.deepcopy(device_models[0])
    # Average shared layers
    for key in shared_keys:
        edge_model[key] = sum(dm[key] for dm in device_models) / len(device_models)
    # Private layers remain per device (not aggregated)
    for key in private_keys:
        edge_model[key] = device_models[0][key]  # placeholder: could keep first device's private head
    return edge_model


# --------------------------------------
# 3. Global Layer Methods
# --------------------------------------

def aggregate_fedavg_global(edge_models):
    return aggregate_fedavg_edge(edge_models)


def aggregate_fedprox_global(edge_models, global_model_params, mu=0.01):
    return aggregate_fedprox_edge(edge_models, global_model_params, mu=mu)


def aggregate_fedper_global(edge_models, shared_keys, private_keys):
    return aggregate_fedper_edge(edge_models, shared_keys, private_keys)


# --------------------------------------
# 4. HPFL Round Example
# --------------------------------------

def hpfl_round(devices, edges, global_model,
               device_method, edge_method, global_method, shared_keys=None, private_keys=None):
    device_models = []
    # Device layer
    for device in devices:
        if device_method == "FedAvg":
            dm = train_fedavg_device(global_model, device.data)
        elif device_method == "FedProx":
            dm = train_fedprox_device(global_model, global_model.state_dict(), device.data)
        elif device_method == "FedPer":
            dm = train_fedper_device(global_model, device.data, global_model.shared_layers, global_model.private_layers)
        device_models.append(dm)

    # Edge layer
    edge_models = []
    for edge in edges:
        edge_device_models = [device_models[i] for i in edge.devices]
        if edge_method == "FedAvg":
            em = aggregate_fedavg_edge(edge_device_models)
        elif edge_method == "FedProx":
            em = aggregate_fedprox_edge(edge_device_models, global_model.state_dict())
        elif edge_method == "FedPer":
            em = aggregate_fedper_edge(edge_device_models, shared_keys, private_keys)
        edge_models.append(em)

    # Global layer
    if global_method == "FedAvg":
        gm = aggregate_fedavg_global(edge_models)
    elif global_method == "FedProx":
        gm = aggregate_fedprox_global(edge_models, global_model.state_dict())
    elif global_method == "FedPer":
        gm = aggregate_fedper_global(edge_models, shared_keys, private_keys)

    global_model.load_state_dict(gm)
    return global_model

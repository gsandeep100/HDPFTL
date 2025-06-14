import copy

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from hdpftl_evaluation.evaluate_global_model import evaluate_global_model
from hdpftl_training.hdpftl_aggregation.hdpftl_fedavg import aggregate_models
from hdpftl_training.save_model import save
from hdpftl_utility.config import NUM_DEVICES_PER_CLIENT, NUM_FEDERATED_ROUND
from hdpftl_utility.log import safe_log
from hdpftl_utility.utils import named_timer, setup_device


def split_among_devices(X_client, y_client, seed=42):
    """
    Split a single client's data among its devices roughly evenly.
    You can also do stratified splitting here if needed.
    """
    np.random.seed(seed)
    indices = np.arange(len(y_client))
    np.random.shuffle(indices)

    device_data = []
    size = len(y_client) // NUM_DEVICES_PER_CLIENT

    for i in range(NUM_DEVICES_PER_CLIENT):
        start = i * size
        # Last device takes remainder
        end = (i + 1) * size if i < NUM_DEVICES_PER_CLIENT - 1 else len(y_client)
        device_indices = indices[start:end]

        X_device = X_client[device_indices]
        y_device = y_client[device_indices]
        device_data.append((X_device, y_device))

    return device_data


def dirichlet_partition(X, y, alpha, num_clients, seed=42):
    """
    Partition dataset indices using Dirichlet distribution to simulate non-IID data.

    Args:
        X: Features (numpy array or DataFrame).
        y: Labels (numpy array, torch tensor, or pandas Series).
        alpha: Dirichlet concentration parameter.
        num_clients: Number of clients to partition into.
        seed: Random seed for reproducibility.

    Returns:
        client_data_dict: dict mapping client_id to (X_subset, y_subset)
    """

    # Convert y to numpy array if needed
    if hasattr(y, 'values'):
        y = y.values
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()

    np.random.seed(seed)
    unique_classes = np.unique(y)

    client_indices = [[] for _ in range(num_clients)]

    for c in unique_classes:
        class_indices = np.where(y == c)[0]
        np.random.shuffle(class_indices)

        proportions = np.random.dirichlet(alpha * np.ones(num_clients))
        proportions = (proportions * len(class_indices)).astype(int)

        diff = len(class_indices) - proportions.sum()
        if diff > 0:
            proportions[np.argmax(proportions)] += diff
        elif diff < 0:
            proportions[np.argmax(proportions)] += diff

        splits = np.split(class_indices, np.cumsum(proportions)[:-1])
        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split.tolist())

    X_np = X.values if hasattr(X, 'values') else X
    y_np = y

    client_data_dict = {}
    for client_id, indices in enumerate(client_indices):
        indices = np.array(indices)
        client_data_dict[client_id] = (X_np[indices], y_np[indices])

    return client_data_dict


def dirichlet_partition_with_devices(X, y, alpha=0.3, num_clients=5, num_devices_per_client=2):
    """
    Hierarchical partitioning:
      1. Partition dataset among clients via Dirichlet.
      2. Split each client dataset evenly among devices.

    Returns:
        client_data_dict: dict of client_id -> (X_subset, y_subset)
        hierarchical_data: dict of client_id -> list of device datasets [(X_device, y_device), ...]
    """
    client_data_dict = dirichlet_partition(X, y, alpha=alpha, num_clients=num_clients)

    hierarchical_data = {}
    for client_id, (X_c, y_c) in client_data_dict.items():
        num_samples = len(X_c)
        device_indices = np.array_split(np.arange(num_samples), num_devices_per_client)

        device_data = []
        for d_idx in device_indices:
            # Convert to torch tensors for training later
            device_data.append((torch.tensor(X_c[d_idx]).float(), torch.tensor(y_c[d_idx]).long()))

        hierarchical_data[client_id] = device_data

    return client_data_dict, hierarchical_data


def safe_split(tensor, proportions, dim=0):
    """
    Safely splits a tensor along any dimension using proportions that sum to 1.0.

    Args:
        tensor (torch.Tensor): The tensor to split.
        proportions (list or torch.Tensor): Proportions (floats), ideally summing to 1.0.
        dim (int): The dimension along which to split.

    Returns:
        list of torch.Tensor: Split tensor chunks.
    """
    total = tensor.size(dim)

    # Ensure proportions is a torch tensor
    if not isinstance(proportions, torch.Tensor):
        proportions = torch.tensor(proportions, dtype=torch.float)

    # Normalize proportions to sum to 1.0
    proportions = proportions / proportions.sum()

    # Compute split sizes and adjust for rounding
    split_sizes = (proportions * total).long()
    diff = total - split_sizes.sum()
    split_sizes[0] += diff  # Apply correction to the first split

    return torch.split(tensor, split_sizes.tolist(), dim=dim)


def average_models(model_state_dicts):
    """Averages a list of model state_dicts"""
    avg_state = {}
    for key in model_state_dicts[0]:
        avg_state[key] = sum([sd[key] for sd in model_state_dicts]) / len(model_state_dicts)
    return avg_state


def train_on_device(model, device_data, epochs=1, batch_size=32, lr=0.01):
    """
    Train a model locally on device data for a few epochs.

    Args:
        model (torch.nn.Module): The model to train.
        device_data (tuple): Tuple (X_data, y_data) as either NumPy arrays or tensors.
        epochs (int): Number of training epochs.
        batch_size (int): Training batch size.
        lr (float): Learning rate.

    Returns:
        dict: Trained model's state_dict.
    """
    device = setup_device()
    model = model.to(device)
    model.train()

    X_data, y_data = device_data

    # Convert to tensors if needed
    if isinstance(X_data, np.ndarray):
        X_tensor = torch.tensor(X_data).float().to(device)
    else:
        X_tensor = X_data.float().to(device)

    if isinstance(y_data, np.ndarray):
        y_tensor = torch.tensor(y_data).long().to(device)
    else:
        y_tensor = y_data.long().to(device)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()

    return model.state_dict()


def federated_round(base_model_fn, global_model, hierarchical_data, epochs=1):
    """
    One round of federated training with hierarchical aggregation:
      - Train on each device
      - Aggregate device models per client
      - Aggregate client models into global model

    Args:
        base_model_fn: function returning a fresh model instance
        global_model: current global model
        hierarchical_data: dict client_id -> list of (X_device, y_device)
        epochs: number of epochs to train per device

    Returns:
        Updated global_model with aggregated weights
        :param epochs:
        :param hierarchical_data:
        :param global_model:
        :param base_model_fn:
        :param update_callback:
        :param total_steps:
        :param current_step:
    """
    device = setup_device()
    client_state_dicts = []

    for client_id, devices_data in hierarchical_data.items():
        device_state_dicts = []
        safe_log(f"Training client {client_id} on {len(devices_data)} devices...")

        for device_data in devices_data:
            if device_data[0].size(0) == 0:  # no samples on this device
                safe_log(f"Skipping empty device for client {client_id}")
                continue

            local_model = copy.deepcopy(global_model).to(device)
            local_state_dict = train_on_device(local_model, device_data, epochs=epochs)
            device_state_dicts.append(local_state_dict)

        if not device_state_dicts:
            safe_log(f"No devices trained for client {client_id}, skipping client aggregation.", level="warning")
            continue

        # Aggregate devices per client
        client_agg_state_dict = aggregate_models(device_state_dicts, base_model_fn)
        client_state_dicts.append(client_agg_state_dict)

    if not client_state_dicts:
        safe_log("No clients trained in this round, returning previous global model.", level="warning")
        return global_model

    # Aggregate all clients into new global model
    new_global_state_dict = aggregate_models(client_state_dicts, base_model_fn)
    global_model.load_state_dict(new_global_state_dict)
    return global_model


"""
✅ hierarchical_data: {client_id: [(X_dev1, y_dev1), (X_dev2, y_dev2), ...]}

✅ train_on_device() – for device training

✅ federated_round() – wraps device & client-level training

✅ aggregate_models() – to average states

✅ save() – to persist models

✅ evaluate_global_model() (optional) – to track metrics
"""


# --- Main HDPFTL pipeline ---
def hdpftl_pipeline(base_model_fn, hierarchical_data, X_test, y_test, writer=None):
    device = setup_device()
    safe_log("\n🚀 Starting HDPFTL pipeline...\n")

    # Instantiate global model properly
    # global_model = base_model_fn().to(device)
    global_model = copy.deepcopy(base_model_fn()).to(device)

    # Federated training rounds
    for round_num in range(NUM_FEDERATED_ROUND):
        # safe_log(f"\n🔁 Federated Round {round_num + 1}/{NUM_FEDERATED_ROUND}")
        with named_timer(f"federated_round_{round_num + 1}", writer, tag="federated_round"):
            global_model = federated_round(base_model_fn, global_model, hierarchical_data)
            acc = evaluate_global_model(global_model, X_test, y_test)
            safe_log(f"🌍 Global Accuracy after round {round_num + 1}: {acc:.4f}")

    # Personalized fine-tuning per client
    personalized_models = {}
    for client_id, devices_data in hierarchical_data.items():
        safe_log(f"🔧 Personalizing model for client {client_id}")

        # Combine device data for the client
        X_client = np.concatenate([d[0] for d in devices_data])
        y_client = np.concatenate([d[1] for d in devices_data])

        # Fresh model for personalization
        trained_model = base_model_fn().to(device)

        # Start from global weights
        trained_model.load_state_dict(global_model.state_dict())

        # Train on client-specific data
        trained_model = train_on_device(trained_model, (X_client, y_client), batch_size=32)

        personalized_models[client_id] = trained_model

    save(global_model, personalized_models)
    safe_log("\n✅ HDPFTL complete. Returning global and personalized models.\n")
    return global_model, personalized_models

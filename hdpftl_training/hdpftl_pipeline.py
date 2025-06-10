import copy

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from hdpftl_training.hdpftl_aggregation.hdpftl_fedavg import aggregate_models
from hdpftl_training.save_model import save
from hdpftl_utility.config import NUM_CLIENTS, NUM_DEVICES_PER_CLIENT
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


def dirichlet_partition_with_devices(X, y, alpha=0.3, num_clients=5, devices_per_client=3):
    # Step 1: Partition into clients
    client_data_dict = dirichlet_partition(X, y, alpha=alpha)

    # Step 2: Subdivide each client’s data among devices
    hierarchical_data = {}
    for client_id, (X_c, y_c) in client_data_dict.items():
        num_samples = len(X_c)
        device_indices = np.array_split(np.arange(num_samples), devices_per_client)

        device_data = []
        for d_idx in device_indices:
            device_data.append((torch.tensor(X_c[d_idx]), torch.tensor(y_c[d_idx])))

        hierarchical_data[client_id] = device_data

    return client_data_dict, hierarchical_data


def dirichlet_partition(X, y, alpha, seed=42):
    """
    Partition hdpftl_data indices using Dirichlet distribution.

    Args:
        X: Dataset features (unused, only y matters here for indexing).
        y: Dataset labels (1D numpy array or torch tensor).
        alpha: Dirichlet concentration parameter.
        n_clients: Number of partitions/clients.
        n_classes: Number of unique classes (optional).
        seed: Random seed for reproducibility.

    Returns:
        List of index lists, one per client.
    """
    n_classes = len(torch.unique(torch.tensor(y.values)))


    client_data_dict = {}
    if isinstance(y, torch.Tensor):
        y = y.numpy()

    np.random.seed(seed)
    if n_classes is None:
        n_classes = len(np.unique(y))

    client_indices = [[] for _ in range(NUM_CLIENTS)]

    for c in range(n_classes):
        class_indices = np.where(y == c)[0]
        np.random.shuffle(class_indices)

        proportions = np.random.dirichlet([alpha] * NUM_CLIENTS)
        # Scale to number of class samples
        proportions = (proportions * len(class_indices)).astype(int)

        # Correct any overflows/deficits
        while proportions.sum() > len(class_indices):
            proportions[np.argmax(proportions)] -= 1
        while proportions.sum() < len(class_indices):
            proportions[np.argmin(proportions)] += 1

        splits = np.split(class_indices, np.cumsum(proportions)[:-1])
        for i, split in enumerate(splits):
            client_indices[i].extend(split.tolist())

        # Build client datasets
        for client_id, indices in enumerate(client_indices):
            indices = np.array(indices)
            X_np = X.values if hasattr(X, 'values') else X
            y_np = y.values if hasattr(y, 'values') else y

            client_data_dict[client_id] = (X_np[indices], y_np[indices])

    return client_data_dict


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


def train_on_device(model, device_data, epochs=1, batch_size=32, device='cpu'):
    model.to(device)
    model.train()

    X_device, y_device = device_data
    dataset = TensorDataset(torch.tensor(X_device).float(), torch.tensor(y_device).long())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    return model.state_dict()


def federated_round(global_model, hierarchical_data, device='cpu'):
    client_states = []

    for client_id, devices_data in hierarchical_data.items():
        device_states = []

        for device_data in devices_data:
            # Load global model weights for device training start
            local_model = copy.deepcopy(global_model)
            state = train_on_device(local_model, device_data, device=device)
            device_states.append(state)

        # Aggregate device models within this client
        client_state = aggregate_models(device_states)
        client_states.append(client_state)

    # Aggregate all client models to update global model
    new_global_state = aggregate_models(client_states)
    global_model.load_state_dict(new_global_state)

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
def hdpftl_pipeline(base_model_fn, hierarchical_data, writer=None, num_rounds=5):
    """
    HDPFTL pipeline for hierarchical federated learning.

    Args:
        base_model_fn: Initialized model object (used for cloning).
        hierarchical_data: Dict[client_id] -> List of (X_device, y_device).
        writer: Optional TensorBoard writer.
        num_rounds: Number of federated rounds.

    Returns:
        global_model: Trained global model.
        personalized_models: Dict[client_id] -> fine-tuned personalized model.
    """
    device = setup_device()
    safe_log("\n🚀 Starting HDPFTL pipeline...\n")

    global_model = copy.deepcopy(base_model_fn).to(device)

    for round_num in range(num_rounds):
        safe_log(f"\n🔁 Federated Round {round_num + 1}/{num_rounds}")
        with named_timer(f"federated_round_{round_num + 1}", writer, tag="federated_round"):
            global_model = federated_round(global_model, hierarchical_data, device=device)

    personalized_models = {}
    for client_id, devices_data in hierarchical_data.items():
        model = copy.deepcopy(global_model).to(device)

        # Combine all device-level data for client-level personalization
        X_client = np.concatenate([d[0] for d in devices_data])
        y_client = np.concatenate([d[1] for d in devices_data])

        trained_model = train_on_device(
            model,
            (X_client, y_client),
            epochs=5,
            batch_size=32,
            device=device
        )
        personalized_models[client_id] = trained_model

    save(global_model, personalized_models)
    safe_log("\n✅ HDPFTL complete. Returning global and personalized models.\n")
    return global_model, personalized_models

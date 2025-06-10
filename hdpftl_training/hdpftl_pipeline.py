import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from hdpftl_training.hdpftl_aggregation.hdpftl_fedavg import aggregate_models
from hdpftl_training.save_model import save
from hdpftl_utility.config import NUM_CLIENTS, NUM_DEVICES_PER_CLIENT, NUM_EPOCHS_FINE_TUNE, NUM_TRAIN_ON_DEVICE
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


def dirichlet_partition_with_devices(X, y, alpha=0.3, num_clients=5):
    # Step 1: Partition into clients
    client_data_dict = dirichlet_partition(X, y, alpha=alpha)

    # Step 2: Subdivide each client’s data among devices
    hierarchical_data = {}
    for client_id, (X_c, y_c) in client_data_dict.items():
        num_samples = len(X_c)
        device_indices = np.array_split(np.arange(num_samples), NUM_DEVICES_PER_CLIENT)

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


def train_on_device(model, device_data, batch_size=32):
    """
    Trains a PyTorch model on data from a single device.

    Args:
        model (torch.nn.Module): The model to be trained (usually a local copy of the global model).
        device_data (tuple): A tuple (X_device, y_device) where X_device is features
                             and y_device is labels (can be NumPy arrays or PyTorch tensors).
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for the DataLoader.

    Returns:
        dict: The state_dict of the trained model.
    """
    device = setup_device()
    model.to(device)  # Ensure the model is on the correct device
    model.train()  # Set model to training mode

    X_device_raw, y_device_raw = device_data
    if not isinstance(X_device_raw, (np.ndarray, list)) or len(X_device_raw) == 0:
        print(f"  Info: train_on_device received empty data for training. Skipping training for this device.")
        # Return the current model's state_dict as no training occurred
        return model.state_dict()

    # Robust conversion from NumPy array or existing PyTorch tensor to PyTorch tensor
    # .detach().clone() is used if X_device_raw is already a tensor, to ensure a fresh copy.
    # If X_device_raw is a NumPy array, torch.tensor() handles the conversion and copying.
    X_device_tensor = torch.tensor(X_device_raw).float()
    y_device_tensor = torch.tensor(y_device_raw).long()  # Assuming classification labels

    # Create DataLoader for batching and shuffling
    dataset = TensorDataset(X_device_tensor, y_device_tensor)
    if len(dataset) == 0:
        print(f"  Info: TensorDataset created with no samples. Skipping training for this device.")
        return model.state_dict()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()  # Or nn.MSELoss() for regression, etc.

    # Training loop
    for epoch in range(NUM_TRAIN_ON_DEVICE):
        total_loss = 0
        for features, labels in loader:
            # Move batch data to the same device as the model
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()  # Zero gradients for the current batch
            outputs = model(features)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate loss
            loss.backward()  # Backward pass (compute gradients)
            optimizer.step()  # Update model parameters

            total_loss += loss.item()  # Accumulate loss (optional, for logging/monitoring)

        # Optional: Print epoch loss for monitoring
        # print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(loader):.4f}")

    # Return the state_dict of the trained model, not the full model object.
    # This is standard practice for aggregation in federated learning.
    return model.state_dict()


def federated_round(base_model_fn, global_model, hierarchical_data):
    """
    Performs one round of federated learning with hierarchical aggregation.

    Args:
        base_model_fn (callable): A function that returns a new instance of the model architecture.
        global_model (torch.nn.Module): The current global model instance.
        hierarchical_data (dict): A dictionary where keys are client IDs and values are
                                  lists of device data (e.g., [(X_dev1, y_dev1), (X_dev2, y_dev2)]).

    Returns:
        torch.nn.Module: The updated global model instance after aggregation.
    """
    device = setup_device()
    client_state_dicts = []  # Will store state_dicts from each client

    for client_id, devices_data in hierarchical_data.items():
        print(f"  Processing client: {client_id}")
        device_state_dicts = []  # Will store state_dicts from each device of this client

        for device_data in devices_data:
            X_device_np, y_device_np = device_data  # Assuming device_data is (X_np, y_np)
            if len(X_device_np) == 0:
                print(
                    f"Warning: Device for client '{client_id}' has no training data. Skipping local training for this device.")
                continue
            local_model = copy.deepcopy(global_model).to(device)

            # Train the local model on the device's data.
            # train_on_device should handle data conversion to tensors and moving to device.
            # It should return the state_dict of the trained local model.
            local_state_dict = train_on_device(local_model, device_data)
            device_state_dicts.append(local_state_dict)

        # Handle clients with no actual trained devices
        if not device_state_dicts:
            print(
                f"  Warning: Client '{client_id}' had no devices with data or all devices skipped training. Skipping aggregation for this client.")
            continue  # Skip this client if no state_dicts were generated

        # Aggregate device models' state_dicts within this client to get a client-level state_dict
        # aggregate_models should return a state_dict.
        client_aggregated_state_dict = aggregate_models(device_state_dicts, base_model_fn)
        client_state_dicts.append(client_aggregated_state_dict)

    # Handle case where no clients actually trained in the entire round
    if not client_state_dicts:
        print("Warning: No clients contributed updates in this federated round. Global model remains unchanged.")
        return global_model  # Return the original global model if no updates occurred

    # Aggregate all client models' state_dicts to update the global model's state
    # aggregate_models should return a state_dict.
    new_global_state_dict = aggregate_models(client_state_dicts, base_model_fn)

    # Load the new global state dict into the existing global model instance
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
def hdpftl_pipeline(base_model_fn, hierarchical_data, writer=None):
    device = setup_device()
    safe_log("\n🚀 Starting HDPFTL pipeline...\n")

    # Instantiate global model properly
    global_model = base_model_fn().to(device)

    # Federated training rounds
    for round_num in range(NUM_EPOCHS_FINE_TUNE):
        safe_log(f"\n🔁 Federated Round {round_num + 1}/{NUM_EPOCHS_FINE_TUNE}")
        with named_timer(f"federated_round_{round_num + 1}", writer, tag="federated_round"):
            global_model = federated_round(base_model_fn, global_model, hierarchical_data)

    # Personalized fine-tuning per client
    personalized_models = {}
    for client_id, devices_data in hierarchical_data.items():
        model = copy.deepcopy(global_model).to(device)

        # Combine device-level data into client-level dataset
        X_client = np.concatenate([d[0] for d in devices_data])
        y_client = np.concatenate([d[1] for d in devices_data])

        # Instantiate fresh model from base_model_fn, move to device
        trained_model = base_model_fn().to(device)

        # Load global weights as starting point
        trained_model.load_state_dict(model.state_dict())

        trained_model = train_on_device(trained_model, (X_client, y_client), batch_size=32)

        personalized_models[client_id] = trained_model

    save(global_model, personalized_models)
    safe_log("\n✅ HDPFTL complete. Returning global and personalized models.\n")
    return global_model, personalized_models

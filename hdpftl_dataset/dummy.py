import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


# ========== 6. Simulated IIoT Data ==========
def create_fleet_data(num_fleets=20, devices_per_fleet=30, samples_per_device=200):
    fleets_dataloaders = []
    for _ in range(num_fleets):
        fleet = []
        for _ in range(devices_per_fleet):
            X = torch.tensor(np.random.normal(0, 1, (samples_per_device, 3)), dtype=torch.float32)
            y = torch.tensor(np.random.choice([0, 1], size=(samples_per_device,), p=[0.9, 0.1]), dtype=torch.long)
            dataset = TensorDataset(X, y)
            loader = DataLoader(dataset, batch_size=32, shuffle=True)
            fleet.append(loader)
        fleets_dataloaders.append(fleet)
    return fleets_dataloaders

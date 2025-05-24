import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# ========== 1. Base Model ==========
class TabularNet(nn.Module):
    def __init__(self, input_dim=3, output_dim=2):
        super(TabularNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# ========== 2. Device-Level Training ==========
def train_device_model(base_model_fn, dataloader, device):
    model = base_model_fn().to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(2):  # Tune as needed
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
    return model.state_dict()


# ========== 3. Fleet-Level Aggregation ==========
def aggregate_fleet_models(models):
    avg_model = copy.deepcopy(models[0])
    for key in avg_model:
        for i in range(1, len(models)):
            avg_model[key] += models[i][key]
        avg_model[key] = avg_model[key] / len(models)
    return avg_model


# ========== 4. Transfer + Personalize ==========
def transfer_and_personalize(global_model_weights, target_loader, base_model_fn, device):
    model = base_model_fn().to(device)
    model.load_state_dict(global_model_weights)

    # Freeze all layers except last
    for param in list(model.parameters())[:-2]:  # Freeze early layers
        param.requires_grad = False

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(1):  # Fine-tuning
        for X, y in target_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
    return model


# ========== 5. Federated Pipeline ==========
def hdpftl_pipeline(fleet_dataloaders, base_model_fn, device):
    fleet_models = []

    for fleet_id, device_loaders in enumerate(fleet_dataloaders):
        print(f"\n🚛 Fleet {fleet_id + 1} - Aggregating Device Models")
        device_models = []
        for loader in device_loaders:
            weights = train_device_model(base_model_fn, loader, device)
            device_models.append(weights)

        fleet_model = aggregate_fleet_models(device_models)
        fleet_models.append(fleet_model)

    # Global Aggregation (Hierarchical)
    global_model = aggregate_fleet_models(fleet_models)

    # Transfer and Personalize to a target device (e.g., first one)
    print("\n🔁 Transfer Learning + Personalization")
    target_loader = fleet_dataloaders[0][0]
    personalized_model = transfer_and_personalize(global_model, target_loader, base_model_fn, device)
    return personalized_model


# ========== 6. Simulated IIoT Data ==========
def create_fleet_data(num_fleets=200, devices_per_fleet=300, samples_per_device=200):
    fleet_dataloaders = []
    for _ in range(num_fleets):
        fleet = []
        for _ in range(devices_per_fleet):
            X = torch.tensor(np.random.normal(0, 1, (samples_per_device, 3)), dtype=torch.float32)
            y = torch.tensor(np.random.choice([0, 1], size=(samples_per_device,), p=[0.9, 0.1]), dtype=torch.long)
            dataset = TensorDataset(X, y)
            loader = DataLoader(dataset, batch_size=32, shuffle=True)
            fleet.append(loader)
        fleet_dataloaders.append(fleet)
    return fleet_dataloaders


# ========== 7. Evaluation ==========
def evaluate(model, device):
    model.eval()
    with torch.no_grad():
        X = torch.tensor(np.random.normal(0, 1, (100, 3)), dtype=torch.float32).to(device)
        y = torch.tensor(np.random.choice([0, 1], size=(100,)), dtype=torch.long).to(device)
        outputs = model(X)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == y).float().mean()
        print(f"\n✅ Personalized Model Test Accuracy: {acc.item():.2f}")


# ========== 8. Main ==========
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("📡 Running HDPFTL Pipeline for IIoT...\n")
    fleet_dataloaders = create_fleet_data()
    personalized_model = hdpftl_pipeline(fleet_dataloaders, TabularNet, device)
    evaluate(personalized_model, device)


if __name__ == "__main__":
    main()

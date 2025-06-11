import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from hdpftl_utility.config import BATCH_SIZE
from hdpftl_utility.log import safe_log
from hdpftl_utility.utils import setup_device


def train_device_model(
        model,
        train_data,
        train_labels,
        val_data=None,
        val_labels=None,
        epochs=20,
        lr=0.001,
        early_stopping_patience=5,
        verbose=True
):
    global best_model_state
    device = setup_device()
    model = model.to(device)

    train_data = train_data.to(device)
    train_labels = train_labels.to(device)

    train_dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)

    if val_data is not None and val_labels is not None:
        val_data = val_data.to(device)
        val_labels = val_labels.to(device)
        val_dataset = TensorDataset(val_data, val_labels)
        val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False)
    else:
        val_loader = None

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        if val_loader:
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    outputs = model(x_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += y_batch.size(0)
                    correct += (predicted == y_batch).sum().item()

            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * correct / total

            # if verbose:
            #   safe_log(f"Epoch [{epoch + 1}/{epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

            scheduler.step(avg_val_loss)

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                best_model_state = model.state_dict()
            else:
                epochs_no_improve += 1
                if verbose:
                    safe_log(f"Early stopping at epoch {epoch + 1}", level="warning")
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break
        else:
            # No validation
            # if verbose:
            #     safe_log(f"Epoch [{epoch + 1}/{epochs}] Train Loss: {avg_train_loss:.4f}")
            scheduler.step(avg_train_loss)

    return model

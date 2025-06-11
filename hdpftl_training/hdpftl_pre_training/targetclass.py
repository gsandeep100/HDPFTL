# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:        targetclass.py
   Description:      HDPFTL - Preventing Zero-day Attacks on IoT Devices using
                     Hierarchical Decentralized Personalized Federated Transfer Learning (HDPFTL)
                     with ResNet-18 Model for Cross-Silo Collaboration on Heterogeneous Non-IID Data
   Author:           Sandeep Ghosh
   Created Date:     2025-04-27
   Python3 Version:   3.12.8
-------------------------------------------------
"""
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from hdpftl_training.hdpftl_models.TabularNet import TabularNet
from hdpftl_utility.config import BATCH_SIZE, EPOCH_DIR, EPOCH_FILE_FINE, PRE_MODEL_PATH, FINETUNE_MODEL_PATH, \
    NUM_EPOCHS_PRE_TRAIN
from hdpftl_utility.log import safe_log
from hdpftl_utility.utils import setup_device

"""
2. Fine-tuning phase
Use X_finetune, y_finetune — more specific, target data.

Fine-tune pretrained model for your specific task.
"""


def finetune_model(X_finetune, y_finetune, input_dim, target_classes):
    safe_log("\n=== Fine-tuning Phase ===")
    device = setup_device()

    # Convert to tensors
    def to_tensor(data, dtype):
        if hasattr(data, 'values'):  # pandas DataFrame or Series
            data_np = data.values
        else:
            data_np = data
        return torch.tensor(data_np, dtype=dtype)

    target_features = to_tensor(X_finetune, dtype=torch.float32)
    finetune_labels = to_tensor(y_finetune, dtype=torch.long)

    safe_log(f"target_features shape: {target_features.shape}")
    safe_log(f"finetune_labels shape: {finetune_labels.shape}")

    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        target_features, finetune_labels, test_size=0.2, random_state=42
    )

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    # Load model and pretrained weights
    transfer_model = TabularNet(input_dim, target_classes).to(device)

    try:
        state_dict = torch.load(PRE_MODEL_PATH)
        missing, unexpected = transfer_model.load_state_dict(state_dict, strict=False)
        safe_log("✅ Loaded pretrained model (strict=False)")
        if missing:
            safe_log(f"⚠️ Missing keys: {missing}", level="warning")
        if unexpected:
            safe_log(f"⚠️ Unexpected keys: {unexpected}", level="error")
    except Exception as e:
        safe_log("❌ Could not load pretrained model")
        safe_log(f"Error: {e}", level="error")

    # Replace classifier head
    transfer_model.classifier = nn.Linear(64, target_classes).to(device)  # Assumes last shared layer has 64 units

    # Unfreeze selected shared layers
    for name, param in transfer_model.shared.named_parameters():
        param.requires_grad = '1' in name or '2' in name

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, transfer_model.parameters()), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    epoch_losses = []
    os.makedirs(EPOCH_DIR, exist_ok=True)

    for epoch in range(NUM_EPOCHS_PRE_TRAIN):
        transfer_model.train()
        running_loss, correct, total = 0.0, 0, 0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = transfer_model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)

        # Validation
        transfer_model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = transfer_model(features)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        epoch_losses.append(avg_loss)

        safe_log(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS_PRE_TRAIN}] - "
            f"Loss: {avg_loss:.4f} - Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(transfer_model.state_dict(), FINETUNE_MODEL_PATH)
            safe_log(f"💾 Best model saved at epoch {epoch + 1} with Val Acc: {val_acc:.2f}%")

        np.save(EPOCH_FILE_FINE, np.array(epoch_losses))

    return transfer_model

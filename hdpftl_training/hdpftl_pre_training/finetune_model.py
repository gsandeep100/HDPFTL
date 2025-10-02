# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:        finetune_model.py
   Description:      HPFL - Preventing Zero-day Attacks on IoT Devices using
                     Hierarchical Decentralized Personalized Federated Transfer Learning (HPFL)
                     with ResNet-18 Model for Cross-Silo Collaboration on Heterogeneous Non-IID Data
   Author:           Sandeep Ghosh
   Created Date:     2025-04-27
   Python3 Version:   3.12.8
-------------------------------------------------
"""
import gc
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import hdpftl_utility.config as config
import hdpftl_utility.log as log_util
import hdpftl_utility.utils as util
from hdpftl_training.hdpftl_models.TabularNet import TabularNet

"""
2. Fine-tuning phase
Use X_finetune, y_finetune — more specific, target data.

Fine-tune pretrained model for your specific task.
"""


def finetune_model(X_finetune, y_finetune, input_dim, target_classes):
    log_util.safe_log("\n=== Fine-tuning Phase ===")
    device = util.setup_device()

    # Convert to tensors
    def to_tensor(data, dtype):
        if hasattr(data, 'values'):  # pandas DataFrame or Series
            data_np = data.values
        else:
            data_np = data
        return torch.tensor(data_np, dtype=dtype)

    target_features = to_tensor(X_finetune, dtype=torch.float32)
    finetune_labels = to_tensor(y_finetune, dtype=torch.long)

    # safe_log(f"target_features shape: {target_features.shape}")
    # safe_log(f"finetune_labels shape: {finetune_labels.shape}")

    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        target_features, finetune_labels, test_size=0.2, random_state=42
    )

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=config.BATCH_SIZE_TRAINING, shuffle=True,
                              pin_memory=False)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=config.BATCH_SIZE_TRAINING, shuffle=False,
                            pin_memory=False)

    # Load model and pretrained weights
    transfer_model = TabularNet(input_dim, target_classes).to(device)

    try:
        state_dict = torch.load(config.PRE_MODEL_PATH_TEMPLATE.substitute(n=util.get_today_date()))
        missing, unexpected = transfer_model.load_state_dict(state_dict, strict=False)
        log_util.safe_log("FineTuning model (strict=False)")
        if missing:
            log_util.safe_log(f"⚠️ Missing keys: {missing}", level="warning")
        if unexpected:
            log_util.safe_log(f"⚠️ Unexpected keys: {unexpected}", level="error")
    except Exception as e:
        log_util.safe_log(f"❌Could not load pretrained model Error: {e}", level="error")
        del transfer_model
        torch.cuda.empty_cache()
        gc.collect()
        return None

    # Replace classifier head
    transfer_model.classifier = nn.Linear(64, target_classes).to(device)  # Assumes last shared layer has 64 units

    # Unfreeze selected shared layers
    for name, param in transfer_model.shared.named_parameters():
        param.requires_grad = '1' in name or '2' in name

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, transfer_model.parameters()), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    epoch_losses = []
    os.makedirs(config.EPOCH_DIR, exist_ok=True)

    for epoch in range(config.NUM_EPOCHS_PRE_TRAIN):
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

        log_util.safe_log(
            f"Epoch [{epoch + 1}/{config.NUM_EPOCHS_PRE_TRAIN}] - "
            f"Loss: {avg_loss:.4f} - Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(transfer_model.state_dict(),
                       config.FINETUNE_MODEL_PATH_TEMPLATE.substitute(n=util.get_today_date()))
            log_util.safe_log(f"💾 Best model saved at epoch {epoch + 1} with Val Acc: {val_acc:.2f}%")

        np.save(config.EPOCH_FILE_FINE, np.array(epoch_losses))

    # === CLEANUP ===
    del optimizer, criterion, train_loader, val_loader
    del X_train, X_val, y_train, y_val, target_features, finetune_labels
    torch.cuda.empty_cache()
    gc.collect()

    return transfer_model


def init_model(input_dim, target_classes):
    """Return a fresh untrained model for PFL or FL."""
    model = TabularNet(input_dim, target_classes)
    return model

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
from hdpftl_utility.config import BATCH_SIZE, input_dim, target_classes, FINETUNE_MODEL_PATH, EPOCH_DIR_FINE, \
    EPOCH_FILE_FINE
from hdpftl_utility.utils import setup_device


def target_class():
    print("\n=== Fine-tuning Phase ===")
    device = setup_device()

    # 1. Generate target data (replace this with real data in production)
    target_features = torch.randn(1000, input_dim)
    target_labels = torch.randint(0, target_classes, (1000,))

    # 2. Train/Validation split
    X_train, X_val, y_train, y_val = train_test_split(
        target_features, target_labels, test_size=0.2, random_state=42
    )
    train_loader = DataLoader(TensorDataset(X_train, y_train), BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), BATCH_SIZE, shuffle=False)

    # 3. Load pretrained model
    transfer_model = TabularNet(input_dim, target_classes).to(device)
    try:
        state_dict = torch.load(FINETUNE_MODEL_PATH)
        missing, unexpected = transfer_model.load_state_dict(state_dict, strict=False)
        print("✅ Loaded pretrained model (strict=False)")
        if missing:
            print(f"⚠️ Missing keys: {missing}")
        if unexpected:
            print(f"⚠️ Unexpected keys: {unexpected}")
    except Exception as e:
        print("❌ Could not load pretrained model")
        print(f"Error: {e}")

    # 4. Replace classifier
    transfer_model.classifier = nn.Linear(64, target_classes).to(device)

    # 5. Unfreeze specific shared layers
    for name, param in transfer_model.shared.named_parameters():
        param.requires_grad = '2' in name or '1' in name

    # 6. Optimizer and loss
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, transfer_model.parameters()), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 7. Fine-tuning loop
    best_val_acc = 0.0
    epoch_losses = []
    os.makedirs(EPOCH_DIR_FINE, exist_ok=True)

    for epoch in range(10):
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

        print(f"Epoch [{epoch + 1}/10] - Loss: {avg_loss:.4f} - Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(transfer_model.state_dict(), FINETUNE_MODEL_PATH)

        # Save every epoch
        np.save(EPOCH_FILE_FINE, np.array(epoch_losses))

    return transfer_model

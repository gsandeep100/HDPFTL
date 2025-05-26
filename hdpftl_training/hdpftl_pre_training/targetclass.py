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
from torch.utils.data import DataLoader, TensorDataset

from hdpftl_training.hdpftl_models.TabularNet import TabularNet
from hdpftl_utility.config import input_dim, target_classes, FINETUNE_MODEL_PATH, EPOCH_DIR_FINE, EPOCH_FILE_FINE
from hdpftl_utility.utils import setup_device


from sklearn.model_selection import train_test_split

def target_class():
    print("\n=== Fine-tuning Phase ===")
    device = setup_device()

    # 1. Create synthetic target data
    target_features = torch.randn(1000, input_dim)
    target_labels = torch.randint(0, target_classes, (1000,))

    # 2. Train/Validation split
    X_train, X_val, y_train, y_val = train_test_split(
        target_features, target_labels, test_size=0.2, random_state=42
    )

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32, shuffle=False)

    # 3. Load pretrained model
    transfer_model = TabularNet(input_dim, target_classes).to(device)
    try:
        transfer_model.load_state_dict(torch.load("./hdpftl_trained_models/pretrained_tabular_model.pth"))
        print("✅ Loaded pretrained model")
    except:
        print("❌ Could not load pretrained model")

    # 4. Replace final classifier layer
    transfer_model.classifier = nn.Linear(64, target_classes).to(device)

    # 5. Partially unfreeze shared layers (e.g., unfreeze layers 2 and 3)
    for name, param in transfer_model.shared.named_parameters():
        param.requires_grad = '2' in name or '1' in name

    # 6. Optimizer and loss
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, transfer_model.parameters()), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 7. Fine-tuning loop
    best_val_acc = 0.0
    epoch_losses = []
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
        #print(f"Epoch [{epoch+1}/10] - Loss: {avg_loss:.4f} - Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}%")
        # Save epoch
        os.makedirs(EPOCH_DIR_FINE, exist_ok=True)  # creates folder and any missing parents, no error if exists
        np.save(EPOCH_FILE_FINE, np.array(epoch_losses))

        # Save the fine-tuned model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(transfer_model.state_dict(), FINETUNE_MODEL_PATH)
    return transfer_model

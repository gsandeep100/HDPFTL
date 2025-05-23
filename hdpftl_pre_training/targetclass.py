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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from hdpftl_models.TabularNet import TabularNet
from hdpftl_utility.config import input_dim, target_classes, pretrain_classes
from hdpftl_utility.utils import setup_device


def target_class():
    print("\n=== Fine-tuning Phase ===")
    device = setup_device()
    # Create random target hdpftl_data
    target_features = torch.randn(1000, input_dim)
    target_labels = torch.randint(0, target_classes, (1000,))

    target_dataset = TensorDataset(target_features, target_labels)
    target_loader = DataLoader(target_dataset, batch_size=32, shuffle=True)

    # Create new model instance for fine-tuning
    transfer_model = TabularNet(input_dim, pretrain_classes).to(device)
    try:
        transfer_model.load_state_dict(torch.load("./trained-hdpftl_models/pretrained_tabular_model.pth"))
    except:
        print("❌ Something went wrong")
    # Replace final classifier layer
    transfer_model.classifier = nn.Linear(64, target_classes).to(device)
    # Optional: Freeze feature extractor if you want
    for param in transfer_model.shared.parameters():
        param.requires_grad = False
    optimizer = torch.optim.Adam(transfer_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    # Fine-tune the model
    for epoch in range(10):
        transfer_model.train()
        running_loss = 0
        correct = 0
        total = 0

        for features, labels in target_loader:
            features, labels = features.to(device), labels.to(device)

            outputs = transfer_model(features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(
            f"Fine-tune Epoch [{epoch + 1}/10], Loss: {running_loss / len(target_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")
    if os.path.exists("./trained-hdpftl_models/fine_tuned_tabular_model.pth"):
        print("✅ The file 'fine_tuned_tabular_model.pth' exists!")
        os.remove("./trained-hdpftl_models/fine_tuned_tabular_model.pth")
    else:
        print("❌ The file 'fine_tuned_tabular_model.pth' does not exist.")

    # Save fine-tuned model
    torch.save(transfer_model.state_dict(), "./trained-hdpftl_models/fine_tuned_tabular_model.pth")
    return transfer_model

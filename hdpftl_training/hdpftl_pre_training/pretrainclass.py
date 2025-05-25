# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:        pretrainclass.py
   Description:      HDPFTL - Preventing Zero-day Attacks on IoT Devices using
                     Hierarchical Decentralized Personalized Federated Transfer Learning (HDPFTL)
                     with ResNet-18 Model for Cross-Silo Collaboration on Heterogeneous Non-IID Data
   Author:           Sandeep Ghosh
   Created Date:     2025-04-26
   Python3 Version:   3.12.8
-------------------------------------------------
"""

import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from hdpftl_training.hdpftl_models.TabularNet import TabularNet
from hdpftl_utility.config import input_dim, pretrain_classes
from hdpftl_utility.utils import setup_device, make_dir


def extract_priors(model):
    return {
        'fc1.weight': model.fc1.weight.detach(),
        'fc1.bias': model.fc1.bias.detach(),
        'fc2.weight': model.fc2.weight.detach(),
        'fc2.bias': model.fc2.bias.detach(),
    }


def pretrain_class():
    print("\n=== Pretraining Phase ===")
    device = setup_device()
    # Create random pretraining hdpftl_data
    pretrain_features = torch.randn(2000, input_dim)
    pretrain_labels = torch.randint(0, pretrain_classes, (2000,))

    pretrain_dataset = TensorDataset(pretrain_features, pretrain_labels)
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=32, shuffle=True)

    # Create model for pretraining
    pretrain_model = TabularNet(input_dim, pretrain_classes).to(device)
    optimizer = torch.optim.Adam(pretrain_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    for epoch in range(5):
        pretrain_model.train()
        running_loss = 0
        for features, labels in pretrain_loader:
            features, labels = features.to(device), labels.to(device)

            outputs = pretrain_model(features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # print(f"Pretrain Epoch [{epoch + 1}/5], Loss: {running_loss / len(pretrain_loader):.4f}")
        make_dir("./trained-hdpftl_models/")
    if os.path.exists("./trained-hdpftl_models/pretrained_tabular_model.pth"):
        print("✅ The file 'pretrained_tabular_model.pth' exists!")
        os.remove("./trained-hdpftl_models/pretrained_tabular_model.pth")
    else:
        print("❌ The file 'pretrained_tabular_model.pth' does not exist.")

    # Save pretrained model
    torch.save(pretrain_model.state_dict(), "./trained-hdpftl_models/pretrained_tabular_model.pth")

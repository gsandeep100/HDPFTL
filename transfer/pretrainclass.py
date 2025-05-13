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

from config import input_dim, pretrain_classes
from models.TabularNet import TabularNet
from utils import setup_device, make_dir

def pretrain_class():
    print("\n=== Pretraining Phase ===")
    # Create random pretraining data
    pretrain_features = torch.randn(2000, input_dim)
    pretrain_labels = torch.randint(0, pretrain_classes, (2000,))

    pretrain_dataset = TensorDataset(pretrain_features, pretrain_labels)
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=32, shuffle=True)

    # Create model for pretraining
    pretrain_model = TabularNet(input_dim, pretrain_classes).to(setup_device())

    optimizer = torch.optim.Adam(pretrain_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    for epoch in range(5):
        pretrain_model.train()
        running_loss = 0
        for features, labels in pretrain_loader:
            features, labels = features.to(setup_device()), labels.to(setup_device())

            outputs = pretrain_model(features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Pretrain Epoch [{epoch + 1}/5], Loss: {running_loss / len(pretrain_loader):.4f}")

    # Save pretrained model
    make_dir("./trained-models/")
    torch.save(pretrain_model.state_dict(), "./trained-models/pretrained_tabular_model.pth")
    if os.path.exists("./trained-models/pretrained_tabular_model.pth"):
        print("✅ The file 'pretrained_tabular_model.pth' exists!")
    else:
        print("❌ The file 'pretrained_tabular_model.pth' does not exist.")

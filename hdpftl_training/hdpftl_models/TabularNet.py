# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:        TabularNet.py
   Description:      HPFL - Preventing Zero-day Attacks on IoT Devices using
                     Hierarchical Decentralized Personalized Federated Transfer Learning (HPFL)
                     with ResNet-18 Model for Cross-Silo Collaboration on Heterogeneous Non-IID Data
   Author:           Sandeep Ghosh
   Created Date:     2025-04-21
   Python3 Version:   3.12.8
-------------------------------------------------
"""
import torch.nn as nn

import hdpftl_utility.config as config


def create_model_fn(input_dim=config.INPUT_DIM, num_classes=config.NUM_CLASSES):
    return TabularNet(input_dim, num_classes)


class TabularNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TabularNet, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64)
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.shared(x)  # Feature extractor
        x = self.classifier(x)  # Final classification head
        return x

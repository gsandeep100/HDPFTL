# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:        aggregator.py
   Description:      HDPFTL - Preventing Zero-day Attacks on IoT Devices using
                     Hierarchical Decentralized Personalized Federated Transfer Learning (HDPFTL)
                     with ResNet-18 Model for Cross-Silo Collaboration on Heterogeneous Non-IID Data
   Author:           Sandeep Ghosh
   Created Date:     2025-04-28
   Python3 Version:   3.12.8
-------------------------------------------------
"""
import copy

import torch


def train_fleet(self):
    models = [client.train_local(self.model_fn()) for client in self.clients]
    return self.aggregate(models)

#TODO use bayesian aggregation technique
def aggregate(self, models):
    avg_model = copy.deepcopy(models[0])
    for key in avg_model.state_dict():
        avg_model.state_dict()[key] = torch.stack([m.state_dict()[key] for m in models], 0).mean(0)
    return avg_model
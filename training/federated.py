# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:        federated.py
   Description:      HDPFTL - Preventing Zero-day Attacks on IoT Devices using
                     Hierarchical Decentralized Personalized Federated Transfer Learning (HDPFTL)
                     with ResNet-18 Model for Cross-Silo Collaboration on Heterogeneous Non-IID Data
   Author:           Sandeep Ghosh
   Created Date:     2025-04-25
   Python3 Version:   3.12.8
-------------------------------------------------
"""
import torch


def aggregate_models(models, base_model_fn, device):
    new_model = base_model_fn().to(device)
    new_state_dict = {}
    with torch.no_grad():
        for key in models[0].state_dict().keys():
            new_state_dict[key] = sum(m.state_dict()[key] for m in models) / len(models)
    new_model.load_state_dict(new_state_dict)
    return new_model

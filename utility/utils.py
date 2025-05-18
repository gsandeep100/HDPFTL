# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:        utils.py
   Description:      HDPFTL - Preventing Zero-day Attacks on IoT Devices using
                     Hierarchical Decentralized Personalized Federated Transfer Learning (HDPFTL)
                     with ResNet-18 Model for Cross-Silo Collaboration on Heterogeneous Non-IID Data
   Author:           Sandeep Ghosh
   Created Date:     2025-04-22
   Python3 Version:   3.12.8
-------------------------------------------------
"""
import os

import torch

from models.TabularNet import TabularNet


def setup_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
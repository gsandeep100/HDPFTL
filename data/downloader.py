# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:        downloader.py
   Description:      HDPFTL - Preventing Zero-day Attacks on IoT Devices using
                     Hierarchical Decentralized Personalized Federated Transfer Learning (HDPFTL)
                     with ResNet-18 Model for Cross-Silo Collaboration on Heterogeneous Non-IID Data
   Author:           Sandeep Ghosh
   Created Date:     2025-04-23
   Python3 Version:   3.12.8
-------------------------------------------------
"""

import os
import shutil

import kagglehub
from kaggle import KaggleApi

from config import DATASET_PATH


def download_dataset():
    if not os.path.exists(DATASET_PATH):
        api = KaggleApi()
        api.authenticate()
        path = kagglehub.dataset_download("bertvankeulen/cicids-2017")
        #path1 = kagglehub.dataset_download("ishasingh03/friday-workinghours-afternoon-ddos")
        shutil.copytree(path, DATASET_PATH)
        #shutil.copytree(path1, DATASET_PATH)
        print("Dataset downloaded at:", DATASET_PATH)
    else:
        print("Dataset already exists at:", DATASET_PATH)

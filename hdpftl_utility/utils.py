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
import time
import torch
from imblearn.over_sampling import SMOTE, SVMSMOTE, KMeansSMOTE

from hdpftl_utility.log import safe_log
from contextlib import contextmanager
from datetime import datetime


def setup_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# ===== Timer Context Manager =====
@contextmanager
def named_timer(name, writer=None, global_step=None, tag=None):
    safe_log(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ⏱️ Starting {name}...")
    start = time.time()
    yield
    end = time.time()
    elapsed = end - start
    safe_log(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ {name} took {elapsed:.2f} seconds.")
    if writer and tag:
        writer.add_scalar(f"Timing/{tag}", elapsed, global_step if global_step is not None else 0)


# Example:
# X_sm, y_sm = time_resampling('smote', X, y)
def time_resampling(smote_type, X, y, k=5):
    smote_classes = {
        'smote': SMOTE,
        'svm': SVMSMOTE,
        'kmeans': KMeansSMOTE
    }
    sampler = smote_classes[smote_type](k_neighbors=k, random_state=42)
    start = time.time()
    X_res, y_res = sampler.fit_resample(X, y)
    safe_log(f"⏱ {smote_type.upper()} took {time.time() - start:.2f} seconds")
    return X_res, y_res


def createnewoutputfolder():
    # Generate a date-stamped folder name
    timestamp = datetime.now().strftime('%Y-%m-%d')
    folder_name = f"{timestamp}"

    # Create the folder
    os.makedirs(folder_name, exist_ok=True)

    safe_log(f"📁 Created folder: {folder_name}")

def get_today_date():
    return datetime.now().strftime('%Y-%m-%d')

def is_folder_exist(path):
    # Get today's date as a string (e.g., "2025-06-12")
    today_str = datetime.now().strftime('%Y-%m-%d')
    folder_name = f"{today_str}"

    # Check if the folder exists
    if os.path.exists(folder_name):
        return True
    else:
        os.makedirs(folder_name)
        return None


def get_output_folders(folder):
    return [f.name for f in os.scandir(folder) if f.is_dir()]


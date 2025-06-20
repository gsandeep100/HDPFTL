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
import gc
import os
import time
from contextlib import contextmanager
from datetime import datetime
from glob import glob

import numpy as np
import torch
from imblearn.over_sampling import SMOTE, SVMSMOTE, KMeansSMOTE

from hdpftl_utility.log import safe_log


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


def number_of_data_folders(folderpath):
    all_files = glob(os.path.join(folderpath, "*.csv")) + glob(os.path.join(folderpath, "*.CSV"))
    return len(all_files)


def is_folder_exist(path):
    if os.path.exists(path):
        return True
    else:
        os.makedirs(path)
        return False


def get_output_folders(folder):
    return [f.name for f in os.scandir(folder) if f.is_dir()]

def to_float32(X):
    if isinstance(X, np.ndarray):
        return X.astype(np.float32)
    elif hasattr(X, 'values'):
        return X.values.astype(np.float32)
    else:
        return np.array(X, dtype=np.float32)

def clear_memory():
    # Collect garbage (Python-level cleanup)
    gc.collect()

    # Clear PyTorch CUDA cache if using GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    print("🧹 Memory cleared (GC + CUDA)")

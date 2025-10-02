# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:        utils.py
   Description:      HPFL - Preventing Zero-day Attacks on IoT Devices using
                     Hierarchical Decentralized Personalized Federated Transfer Learning (HPFL)
                     with ResNet-18 Model for Cross-Silo Collaboration on Heterogeneous Non-IID Data
   Author:           Sandeep Ghosh
   Created Date:     2025-04-22
   Python3 Version:   3.12.8
-------------------------------------------------
"""
import gc
import os
import platform
import time
from contextlib import contextmanager
from datetime import datetime
from glob import glob

import numpy as np
import torch
from imblearn.over_sampling import KMeansSMOTE, SMOTE, SVMSMOTE

import config
import hdpftl_utility.log as log_util


def get_os():
    os_name = platform.system()
    if os_name == "Darwin":
        return "macOS"
    elif os_name == "Linux":
        # Further check if it's Pop!_OS
        try:
            with open("/etc/os-release") as f:
                release_info = f.read()
                if "Pop!_OS" in release_info:
                    return "Pop!_OS"
                elif "Ubuntu" in release_info:
                    return "Ubuntu"
                else:
                    return "Other Linux"
        except FileNotFoundError:
            return "Linux (unknown distro)"
    else:
        return os_name


def sync_config_params(config_params):
    """
    Update config_params dict values from config module attributes if they exist.
    Keeps existing value if attribute not found.
    """
    for key in config_params.keys():
        if hasattr(config, key):
            config_params[key] = getattr(config, key)
    return config_params


def setup_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def reload_config():
    """Reload config.py and return the fresh module."""
    import importlib
    import hdpftl_utility.config as cfg
    return importlib.reload(cfg)


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# ===== Timer Context Manager =====
@contextmanager
def named_timer(name, writer=None, global_step=None, tag=None):
    log_util.safe_log(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ⏱️ Starting {name}...")
    start = time.time()
    yield
    end = time.time()
    elapsed = end - start
    log_util.safe_log(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ {name} took {elapsed:.2f} seconds.")
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
    log_util.safe_log(f"⏱ {smote_type.upper()} took {time.time() - start:.2f} seconds")
    return X_res, y_res


def createnewoutputfolder():
    # Generate a date-stamped folder name
    timestamp = datetime.now().strftime('%Y-%m-%d')
    folder_name = f"{timestamp}"

    # Create the folder
    os.makedirs(folder_name, exist_ok=True)

    log_util.safe_log(f"📁 Created folder: {folder_name}")


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

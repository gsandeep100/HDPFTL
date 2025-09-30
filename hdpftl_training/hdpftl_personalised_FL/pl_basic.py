#!/usr/bin/env python3
"""
Hierarchical PFL (HPFL) pipeline with Dirichlet partitioning
- Device Layer: Intra-device sequential boosting
- Edge Layer: Inter-device / edge sequential boosting
- Gossip / Global Layer: Bayesian aggregation with fallback
- Residual Feedback: Forward + backward passes
- Calibration: Isotonic regression per class
- Variance pruning: Optional weak learner pruning
- Logging & dynamic folders
- Model saving: Device, edge, gossip
- Plots: accuracy, residual norms, device/edge/global trends
"""

import logging
import os
import warnings
from datetime import datetime
from typing import List, Tuple, Union

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import early_stopping, log_evaluation
from scipy.special import softmax as sp_softmax
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

from hdpftl_training.hdpftl_data.preprocess import safe_preprocess_data

logging.basicConfig(level=logging.INFO)

os.environ["OMP_NUM_THREADS"] = "4"

# -------------------------------------------------------------
# Config
# -------------------------------------------------------------
config = {
    "random_seed": 42,
    "n_edges": 20,
    "n_device": 100,
    "device_per_edge": 5,
    "epoch": 20,
    "device_boosting_rounds": 5,
    "edge_boosting_rounds": 5,
    "variance_prune": True,
    "variance_threshold": 1e-4,
    "bayes_n_samples": 500,
    "bayes_n_tune": 500,
    "save_results": True,
    "results_path": "results",
    "isotonic_min_positives": 5,
    "max_cores": 2,
    "n_estimators": 50,
    "num_leaves": 10,
    "alpha": 1.0,
    "learning_rate": 0.05,
    "max_depth": 5,
    "min_data_in_leaf": 10,
    "feature_fraction": 0.8,
    "early_stopping_rounds_device": 20,
    "early_stopping_rounds_edge": 20

}

rng = np.random.default_rng(config["random_seed"])
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series]
DeviceData = Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]



# ============================================================
# LightGBM Training
# ============================================================

def train_lightgbm(X_train, y_train, X_valid=None, y_valid=None, early_stopping_rounds=None, num_classes=None):
    """
    Train LightGBM classifier with safe NumPy arrays to avoid feature name warnings.
    Handles multiclass or binary classification.
    """
    X_np = safe_array(X_train)
    y_np = safe_array(y_train)

    if num_classes is None:
        num_classes = len(np.unique(y_np))

    objective = "multiclass" if num_classes > 2 else "binary"
    num_class = num_classes if num_classes > 2 else None

    model = lgb.LGBMClassifier(
        objective=objective,
        num_class=num_class,
        n_estimators=config["n_estimators"],
        random_state=config["random_seed"],
        num_leaves=config["num_leaves"],
        learning_rate=config["learning_rate"],
        max_depth=config["max_depth"],
        min_data_in_leaf=config["min_data_in_leaf"],
        min_child_samples=config["min_data_in_leaf"],

        feature_fraction=config["feature_fraction"],
        device="gpu",
        gpu_platform_id=0,
        gpu_device_id=0
    )
    fit_kwargs = {}
    mask = np.isin(y_valid, np.unique(y_np))
    X_valid_filtered = X_valid[mask]
    y_valid_filtered = y_valid[mask]

    if X_valid is not None and y_valid is not None and early_stopping_rounds:
        fit_kwargs.update({
            "eval_set": [(safe_array(X_valid_filtered), safe_array(y_valid_filtered))],
            "eval_metric": "multi_logloss" if num_classes > 2 else "logloss",
            "callbacks": [
                early_stopping(early_stopping_rounds),  # ✅ new API
                log_evaluation(10)  # optional logging
            ],
        })

    model.fit(X_np, y_np, **fit_kwargs)
    return model


# ============================================================
# Predict probabilities safely, filling missing classes
# ============================================================

def predict_proba_fixed(model, X, num_classes, le=None):
    """
    Predict probabilities safely, even if some classes were missing during training.
    Ensures shape (n_samples, num_classes) and uses NumPy arrays to avoid warnings.
    """
    X_np = safe_array(X)
    pred = model.predict_proba(X_np)
    pred = np.atleast_2d(pred)

    # Already matches num_classes
    if pred.shape[1] == num_classes:
        return pred

    # Fill missing classes
    full = np.zeros((pred.shape[0], num_classes), dtype=float)
    model_classes = getattr(model, "classes_", np.arange(pred.shape[1]))

    if le is not None:
        class_pos = {cls: i for i, cls in enumerate(le.classes_)}
        for i, cls in enumerate(model_classes):
            pos = class_pos.get(cls)
            if pos is not None:
                full[:, pos] = pred[:, i]
    else:
        for i, cls in enumerate(model_classes):
            full[:, int(cls)] = pred[:, i]

    # Normalize to avoid zeros
    row_sums = full.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    full /= row_sums

    return full


def get_leaf_indices(model, X):
    """
    Extract the leaf indices for all samples in all trees of a trained LightGBM model.

    Args:
        model: Trained LightGBM model (Booster or LGBMClassifier)
        X: np.ndarray or pandas DataFrame, input samples

    Returns:
        np.ndarray: shape (n_samples, n_trees)
            Each entry indicates the leaf index for that sample in that tree
    """
    # LightGBM: pred_leaf=True returns leaf indices
    leaf_indices = model.predict(X, pred_leaf=True)

    # Ensure it's a NumPy array
    return np.array(leaf_indices)


# ============================================================
# Device Layer Boosting with missing-class safe probabilities
# ============================================================

def device_layer_boosting(devices_data, residuals_devices, device_models, le, num_classes, X_finetune=None,
                          y_finetune=None):
    """
    Device-level sequential boosting (classification version).

    Returns:
        residuals_devices: list of np.ndarray
            Final residuals per device
        device_models: list
            Trained device models per device
        device_embeddings: list of np.ndarray
            One-hot leaf embeddings per device
    """

    if not isinstance(config, dict):
        raise TypeError(f"Expected `config` to be a dict, got {type(config)}. "
                        f"Make sure you pass the full config dictionary, not a single value.")

    if "device_boosting_rounds" not in config:
        raise KeyError("`config` must contain key 'device_boosting_rounds'")

    device_embeddings = []

    for idx, dev_tuple in enumerate(devices_data):
        X_train, _, y_train, _ = dev_tuple
        n_samples = X_train.shape[0]

        # Initialize residuals
        if residuals_devices[idx] is None:
            residual = np.zeros((n_samples, num_classes), dtype=float)
            residual[np.arange(n_samples), le.transform(y_train)] = 1.0
        else:
            residual = residuals_devices[idx].copy()

        models_per_device = device_models[idx] if device_models[idx] else []

        # Sequential boosting per device
        for t in range(config["device_boosting_rounds"]):
            y_pseudo = residual.argmax(axis=1)
            if len(np.unique(y_pseudo)) < 2:
                logging.debug(f"Device {idx}, round {t}: single class, skipping.")
                break

            model = train_lightgbm(X_train, y_pseudo, X_finetune, y_finetune,
                                   config["early_stopping_rounds_device"], num_classes=num_classes)

            pred_proba = predict_proba_fixed(model, X_train, num_classes, le=le)

            # Update residuals
            residual -= pred_proba
            residual = np.clip(residual, 0.0, None)

            models_per_device.append(model)

        residuals_devices[idx] = residual
        device_models[idx] = models_per_device

        # Compute leaf embeddings (one-hot of leaf indices from all trees)
        # Here we assume `get_leaf_indices(model, X)` returns leaf indices per sample
        leaf_indices_list = [get_leaf_indices(mdl, X_train) for mdl in models_per_device]
        leaf_indices_concat = np.hstack(leaf_indices_list)
        leaf_embeddings = np.zeros((n_samples, np.max(leaf_indices_concat) + 1))
        leaf_embeddings[np.arange(n_samples)[:, None], leaf_indices_concat] = 1.0
        device_embeddings.append(leaf_embeddings)

    return residuals_devices, device_models, device_embeddings


import numpy as np
import logging


def edge_layer_boosting(edge_groups, device_embeddings, residuals_devices, le, num_classes,
                        X_finetune=None, y_finetune=None):
    """
    Edge-level sequential boosting.

    Parameters
    ----------
    edge_groups : list of lists
        Each sublist contains indices of devices in that edge.
    device_embeddings : list of np.ndarray
        Device-level one-hot leaf embeddings per device.
    residuals_devices : list of np.ndarray
        Device-level residuals.
    le : LabelEncoder
        For transforming labels.
    num_classes : int

    X_finetune, y_finetune : optional
        Data for early stopping / fine-tuning.

    Returns
    -------
    edge_models : list
        Trained models per edge.
    residuals_edges : list
        Residuals after edge-level boosting per edge.
    edge_embeddings : list of np.ndarray
        One-hot leaf embeddings per edge (stacked from devices).
    weighted_acc : float
        Weighted accuracy across all edges.
    """
    edge_models = []
    residuals_edges = []
    edge_embeddings = []
    edge_acc_list = []

    for edge_idx, edge_devices in enumerate(edge_groups):
        if len(edge_devices) == 0:
            edge_models.append([])
            residuals_edges.append([])
            edge_embeddings.append(np.array([]))
            edge_acc_list.append((0.0, 0))
            continue

        # Stack device embeddings and residuals
        X_edge = np.vstack([device_embeddings[i] for i in edge_devices])
        residual_edge = np.vstack([residuals_devices[i] for i in edge_devices])
        edge_embeddings.append(X_edge.copy())  # save hot embeddings for global layer

        models_per_edge = []

        for t in range(config["edge_boosting_rounds"]):
            y_pseudo = residual_edge.argmax(axis=1)
            if len(np.unique(y_pseudo)) < 2:
                logging.debug(f"Edge {edge_idx}, round {t}: single class, skipping.")
                break

            # Train edge model
            model_e = train_lightgbm(X_edge, y_pseudo, X_finetune, y_finetune,
                                     early_stopping_rounds=20, num_classes=num_classes)
            pred_proba = predict_proba_fixed(model_e, X_edge, num_classes, le=le)

            # Variance pruning if enabled
            if config.get("variance_prune", False):
                var = np.var(pred_proba, axis=0)
                mask = var >= config.get("variance_threshold", 0.0)
                pred_proba[:, ~mask] = 0.0
                row_sums = pred_proba.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1.0
                pred_proba /= row_sums

            # Update residuals
            residual_edge -= pred_proba
            residual_edge = np.clip(residual_edge, 0.0, None)

            models_per_edge.append(model_e)

        edge_models.append(models_per_edge)
        residuals_edges.append(residual_edge)

        # Optional: compute weighted edge accuracy
        # (requires device test data, not included in this version)

    return edge_models, residuals_edges, edge_embeddings


def global_layer_bayesian_aggregation(edge_outputs, edge_embeddings, y_true, num_classes):
    """
    Global (server) Bayesian aggregation using edge embeddings.

    Parameters
    ----------
    edge_outputs : list of np.ndarray
        Final predictions from each edge after edge-level boosting.
        Each element: shape (n_samples_edge, num_classes)
    edge_embeddings : list of np.ndarray
        Leaf embeddings from each edge. Each element: shape (n_samples_edge, embedding_dim)
    y_true : np.ndarray
        True labels (integer indices or one-hot)
    num_classes : int
        Number of classes

    Returns
    -------
    y_global_pred : np.ndarray
        Global predictions (n_samples, num_classes)
    global_residuals : np.ndarray
        Residuals y_true - y_global_pred
    theta_global : dict
        Posterior parameters of the global Bayesian model
    """
    # Stack all edges' embeddings and outputs
    X_global = np.vstack(edge_embeddings)  # shape: (total_samples, embedding_dim)
    # Before stacking edge outputs
    all_edge_outputs = fix_proba_shape(edge_outputs, num_classes)
    H_global = np.stack(all_edge_outputs, axis=-1)  # shape: (n_samples, num_classes, n_edges)

    n_samples = X_global.shape[0]

    # Initialize global Bayesian parameters (Normal prior)
    alpha = np.zeros(num_classes)
    beta = np.ones((H_global.shape[1], num_classes))

    # Weighted sum of edge outputs
    logits = alpha + H_global @ beta  # shape: (n_samples, num_classes)

    # Softmax to get probabilities
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    y_global_pred = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # Convert y_true to one-hot if needed
    if y_true.ndim == 1:
        y_onehot = np.zeros_like(y_global_pred)
        y_onehot[np.arange(n_samples), y_true] = 1
    else:
        y_onehot = y_true

    # Compute residuals for feedback
    global_residuals = y_onehot - y_global_pred

    # Posterior update placeholder (mean-based)
    theta_global = {
        "alpha": alpha + np.mean(H_global, axis=0),
        "beta": beta + np.mean(global_residuals, axis=0)
    }

    return y_global_pred, global_residuals, theta_global


# ============================================================
# Forward pass with Device → Edge → Gossip
# ============================================================

def forward_pass(devices_data, edge_groups, num_classes, X_finetune, y_finetune,
                 residuals_devices=None, device_models=None):
    """
    Perform a single forward pass through:
    - Device-level boosting
    - Edge-level boosting
    - Gossip/global aggregation
    Returns models, residuals, accuracies
    """
    if residuals_devices is None:
        residuals_devices = [None] * len(devices_data)
    if device_models is None:
        device_models = [None] * len(devices_data)
    # -----------------------------
    # Device Layer
    # -----------------------------
    residuals_devices, device_models, device_embeddings = device_layer_boosting(
        devices_data, residuals_devices, device_models, le, num_classes, X_finetune, y_finetune
    )
    # Ensure embeddings are not None
    assert device_embeddings is not None, "Device embeddings returned as None!"

    # -----------------------------
    # Edge Layer
    # -----------------------------
    edge_models, residuals_edges, edge_embeddings = edge_layer_boosting(
        edge_groups, device_embeddings, residuals_devices, le, num_classes, X_finetune, y_finetune)

    # -----------------------------
    # Gossip / Global Layer
    # -----------------------------
    global_residual, gossip_summary, global_acc = global_layer_bayesian_aggregation(
        devices_data, edge_embeddings, le, num_classes
    )

    return device_models, edge_models, gossip_summary, residuals_devices, residuals_edges, edge_acc, global_acc


# ============================================================
# Evaluate final accuracy
# ============================================================

def evaluate_final_accuracy(devices_data, device_models, edge_groups, le, num_classes, gossip_summary=None):
    """
    Evaluate final accuracy at:
    - Device-level
    - Edge-level
    - Global/Gossip-level (if gossip_summary provided)
    """
    # -----------------------------
    # Device-level
    # -----------------------------
    device_acc_list = []
    for idx, (X_test, _, y_test, _) in enumerate(devices_data):
        X_test_np, y_test_np = safe_array(X_test), safe_array(y_test)
        device_preds = np.zeros((X_test_np.shape[0], num_classes))
        for mdl in device_models[idx]:
            device_preds += predict_proba_fixed(mdl, X_test_np, num_classes, le)
        device_preds = device_preds.argmax(axis=1)
        device_acc_list.append(compute_accuracy(y_test_np, device_preds))
    device_acc = np.mean(device_acc_list)

    # -----------------------------
    # Edge-level
    # -----------------------------
    edge_acc_list = []
    for edge_devices in edge_groups:
        if len(edge_devices) == 0:
            continue
        X_edge = np.vstack([safe_array(devices_data[i][0]) for i in edge_devices])
        y_edge = np.hstack([safe_array(devices_data[i][2]) for i in edge_devices])
        edge_preds = np.zeros((X_edge.shape[0], num_classes))
        for i in edge_devices:
            for mdl in device_models[i]:
                edge_preds += predict_proba_fixed(mdl, X_edge, num_classes, le)
        edge_preds = edge_preds.argmax(axis=1)
        edge_acc_list.append(compute_accuracy(y_edge, edge_preds))
    edge_acc = np.mean(edge_acc_list) if edge_acc_list else 0.0

    # -----------------------------
    # Global/Gossip-level
    # -----------------------------
    global_acc = None
    if gossip_summary is not None:
        X_global = np.vstack([safe_array(devices_data[i][0]) for i in range(len(devices_data))])
        y_global = np.hstack([safe_array(devices_data[i][2]) for i in range(len(devices_data))])
        betas = np.asarray(gossip_summary["betas"])
        alpha = np.asarray(gossip_summary["alpha"])
        if betas.shape[0] != X_global.shape[1]:
            # Pad or truncate to match feature dimension
            if betas.shape[0] < X_global.shape[1]:
                pad_width = X_global.shape[1] - betas.shape[0]
                betas = np.vstack([betas, np.zeros((pad_width, betas.shape[1]))])
            else:
                betas = betas[:X_global.shape[1], :]
        if alpha.ndim != 1 or alpha.shape[0] != betas.shape[1]:
            alpha = alpha.flatten()
        logits = X_global @ betas + alpha
        global_preds = sp_softmax(logits, axis=1).argmax(axis=1)
        global_acc = compute_accuracy(y_global, global_preds)

    return device_acc, edge_acc, global_acc


# -----------------------------
# Step 2: Hierarchical Dirichlet Partition
# -----------------------------
def dirichlet_partition_for_devices_edges_non_iid(X, y, num_devices, device_per_edge, n_edges, alpha=0.5, seed=42):
    """
    Hierarchical non-IID partition.
    Each device gets its own local train/test split for device-level training.
    """
    X_np = X.to_numpy() if isinstance(X, (pd.DataFrame, pd.Series)) else X
    y_np = y.to_numpy() if isinstance(y, (pd.DataFrame, pd.Series)) else y
    rng = np.random.default_rng(seed)

    unique_classes = np.unique(y_np)
    device_indices = [[] for _ in range(num_devices)]

    # Step 1: Non-IID class partition across devices
    for c in unique_classes:
        class_idx = np.where(y_np == c)[0]
        rng.shuffle(class_idx)
        proportions = rng.dirichlet(alpha * np.ones(num_devices))
        min_count = max(int(min(proportions) * len(class_idx)), 1)  # ensure at least 1
        proportions = np.maximum((proportions * len(class_idx)).astype(int), min_count)
        diff = len(class_idx) - proportions.sum()
        proportions[np.argmax(proportions)] += diff
        splits = np.split(class_idx, np.cumsum(proportions)[:-1])
        for device_id, split in enumerate(splits):
            device_indices[device_id].extend(split.tolist())

    # Step 2: Build device-level data with local train/test
    devices_data = []
    for device_id, idxs in enumerate(device_indices):
        X_dev, y_dev = X_np[idxs], y_np[idxs]
        # n_samples = len(X_dev)

        # Split each device into mini-devices if needed
        # mini_idxs = np.array_split(np.arange(n_samples), device_per_edge)
        # device_subdata = [(X_dev[i], y_dev[i]) for i in mini_idxs]
        # hierarchical_data[device_id] = [(X_dev, y_dev)]

        # Local train/test split per device
        X_train, X_test_local, y_train, y_test_local = train_test_split(
            X_dev, y_dev, test_size=0.3, random_state=seed
        )
        devices_data.append((X_train, X_test_local, y_train, y_test_local))

    # Step 3: Create edge groups
    edge_groups = make_edge_groups(num_devices, n_edges, random_state=seed)

    return devices_data, edge_groups


# ============================================================
# Helper functions
# ============================================================

def safe_array(X):
    """Convert input to NumPy array if it is a DataFrame or Series."""
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return X.to_numpy()
    return np.asarray(X)


def fix_proba_shape(H_list, num_classes):
    """
    Ensure each H in H_list has shape (n_samples, num_classes).
    Pads missing columns with zeros or truncates extra columns.
    """
    fixed_list = []
    for H in H_list:
        H = np.atleast_2d(np.array(H))  # ensure 2D array
        n_samples, n_cols = H.shape

        if n_cols < num_classes:
            pad_width = num_classes - n_cols
            H = np.hstack([H, np.zeros((n_samples, pad_width))])
        elif n_cols > num_classes:
            H = H[:, :num_classes]

        fixed_list.append(H)
    return fixed_list

def compute_accuracy(y_true, y_pred):
    y_true_np = safe_array(y_true)
    y_pred_np = safe_array(y_pred)
    print("DEBUG compute_accuracy:")
    return float(np.mean(y_true_np == y_pred_np))


def make_edge_groups(n_devices: int, n_edges: int, random_state: int = 42) -> List[List[int]]:
    idxs = np.arange(n_devices)
    rng_local = np.random.default_rng(random_state)
    rng_local.shuffle(idxs)
    return [list(g) for g in np.array_split(idxs, n_edges)]


# ============================================================
# Dirichlet Partitioning
# ============================================================

def dirichlet_partition(X, y, num_devices, alpha=0.3, seed=42):
    """Partition dataset indices using Dirichlet distribution."""
    y_np = y.values if hasattr(y, 'values') else y
    np.random.seed(seed)
    unique_classes = np.unique(y_np)
    device_indices = [[] for _ in range(num_devices)]

    for c in unique_classes:
        class_idx = np.where(y_np == c)[0]
        np.random.shuffle(class_idx)
        proportions = np.random.dirichlet(alpha * np.ones(num_devices))
        proportions = (proportions * len(class_idx)).astype(int)
        diff = len(class_idx) - proportions.sum()
        proportions[np.argmax(proportions)] += diff
        splits = np.split(class_idx, np.cumsum(proportions)[:-1])
        for device_id, split in enumerate(splits):
            device_indices[device_id].extend(split.tolist())

    X_np = X.values if hasattr(X, 'values') else X
    device_data_dict = {}
    for i, idxs in enumerate(device_indices):
        device_data_dict[i] = (X_np[idxs], y_np[idxs])
    return device_data_dict


def dirichlet_partition_for_devices_edges(X, y, num_devices, device_per_edge, n_edges):
    """Return device_data and hierarchical_data for devices and edges."""
    device_data_dict = dirichlet_partition(X, y, num_devices, config["alpha"], config["random_seed"])
    devices_data = []
    hierarchical_data = {}
    for cid, (X_c, y_c) in device_data_dict.items():
        n_samples = len(X_c)
        device_idxs = np.array_split(np.arange(n_samples), device_per_edge)
        device_data = [(X_c[d], y_c[d]) for d in device_idxs]
        hierarchical_data[cid] = device_data
        X_train, X_test, y_train, y_test = train_test_split(X_c, y_c, test_size=0.2, random_state=config["random_seed"])
        devices_data.append((X_train, X_test, y_train, y_test))
    edge_groups = make_edge_groups(num_devices, n_edges)
    return devices_data, hierarchical_data, edge_groups



# ============================================================
# Training Loop Example
# ============================================================

if __name__ == "__main__":

    folder_path = "CIC_IoT_DIAD_2024"
    today_str = datetime.now().strftime("%Y-%m-%d")
    log_path_str = os.path.join("logs", f"{folder_path}_{today_str}")
    os.makedirs(log_path_str, exist_ok=True)

    # 1. Preprocess
    X_final, y_final, X_pretrain, y_pretrain, X_finetune, y_finetune, X_test, y_test = safe_preprocess_data(
        log_path_str, folder_path, scaler_type='minmax'
    )

    # 2. Encode labels
    le = LabelEncoder()
    le.fit(y_pretrain)
    num_classes = len(le.classes_)

    # 3. Partition fine-tune data for devices & edges
    devices_data, edge_groups = dirichlet_partition_for_devices_edges_non_iid(
        X_pretrain, y_pretrain,  # ✅ use X_finetune & y_finetune
        num_devices=config["n_device"],
        device_per_edge=config["device_per_edge"],
        n_edges=config["n_edges"],
        alpha=0.5,
        seed=42
    )
    device_accs, edge_accs, global_accs = [], [], []
    residuals_devices = [None] * len(devices_data)
    device_models = [None] * len(devices_data)
    residuals_edges = None
    edge_models = None
    gossip_summary = None
    # -----------------------------
    # Training epochs
    # -----------------------------
    for epoch in range(config["epoch"]):
        device_models, edge_models, gossip_summary, residuals_devices, residuals_edges, edge_acc, global_acc = \
            forward_pass(devices_data, edge_groups, num_classes, X_finetune, y_finetune,
                         residuals_devices=residuals_devices,  # pass in residuals
                         device_models=device_models)  # pass in previous models

        # Compute device-level accuracy
        device_epoch_acc_list = []
        for i, c in enumerate(devices_data):
            X_test, _, y_test, _ = c
            device_preds = np.zeros((safe_array(X_test).shape[0], num_classes))
            for mdl in device_models[i]:
                device_preds += predict_proba_fixed(mdl, X_test, num_classes, le)
            device_preds = device_preds.argmax(axis=1)
            device_epoch_acc_list.append(compute_accuracy(y_test, device_preds))
        device_acc_epoch = np.mean(device_epoch_acc_list)

        # Store accuracies
        device_accs.append(device_acc_epoch)
        edge_accs.append(edge_acc)
        global_accs.append(global_acc)

        print(f"[Epoch {epoch + 1}] Device Acc: {device_acc_epoch:.4f}, "
              f"Edge Acc: {edge_acc:.4f}, Global Acc: {global_acc:.4f}")

        # Evaluate final accuracy (sanity check)
        device_acc, edge_acc, global_acc = evaluate_final_accuracy(
            devices_data, device_models, edge_groups, le, num_classes, gossip_summary
        )
        logging.info(f"Final Test Accuracy - Device: {device_acc:.4f}, "
                     f"Edge: {edge_acc:.4f}, Global: {global_acc:.4f}")

    # -----------------------------
    # Plot accuracy trends
    # -----------------------------

    # --- Plot at the end ---
    plt.figure(figsize=(10, 6))
    plt.plot(device_accs, label="Device-level Acc", marker='o')
    plt.plot(edge_accs, label="Edge-level Acc", marker='s')
    plt.plot(global_accs, label="Global Acc", marker='^')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Hierarchical PFL Accuracy Trends")
    plt.ylim(0, 1)  # Optional: fix y-axis
    plt.legend()
    plt.grid(True)
    plt.show()





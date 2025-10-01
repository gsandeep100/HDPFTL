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

from hdpftl_training.hdpftl_data.preprocess import preprocess_data_safe

logging.basicConfig(level=logging.INFO)

os.environ["OMP_NUM_THREADS"] = "4"

# -------------------------------------------------------------
# Config
# -------------------------------------------------------------
config = {
    "random_seed": 42,
    "n_edges": 5,
    "n_device": 5,
    "device_per_edge": 1,
    "epoch": 5,
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
    "learning_rate_device": 0.05,
    "learning_rate_edge": 0.1,
    "learning_rate_backward": 0.1,
    "max_depth": 5,
    "min_data_in_leaf": 5,
    "feature_fraction": 0.8,
    "early_stopping_rounds_device": 20,
    "early_stopping_rounds_edge": 20,
    "early_stopping_rounds_backward": 10

}

rng = np.random.default_rng(config["random_seed"])
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series]
DeviceData = Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]


# ============================================================
# LightGBM Training
# ============================================================

def train_lightgbm(X_train, y_train, X_valid=None, y_valid=None, early_stopping_rounds=None, learning_rate = None, num_classes=None):
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
        learning_rate= learning_rate,
        max_depth=config["max_depth"],
        min_data_in_leaf=config["min_data_in_leaf"],
        min_child_samples=config["min_data_in_leaf"],

        feature_fraction=config["feature_fraction"],
        device="cpu",
        # gpu_platform_id=0,
        # gpu_device_id=0
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

            model = train_lightgbm(X_train, y_pseudo,X_finetune, y_finetune,
                                   config["early_stopping_rounds_device"], config["learning_rate_device"],num_classes=num_classes)

            pred_proba = predict_proba_fixed(model, X_train, num_classes, le=le)

            # Update residuals
            residual -= pred_proba
            residual = np.clip(residual, 0.0, None)

            models_per_device.append(model)
            print(model)

        residuals_devices[idx] = residual
        device_models[idx] = models_per_device

        # Compute leaf embeddings (one-hot of leaf indices from all trees)
        # Here we assume `get_leaf_indices(model, X)` returns leaf indices per sample
        leaf_indices_list = [get_leaf_indices(mdl, X_train) for mdl in models_per_device]
        if leaf_indices_list:
            leaf_indices_concat = np.hstack(leaf_indices_list)
            leaf_embeddings = np.zeros((n_samples, np.max(leaf_indices_concat) + 1))
            leaf_embeddings[np.arange(n_samples)[:, None], leaf_indices_concat] = 1.0
        else:
            # No models trained, return a zero embedding
            leaf_embeddings = np.zeros((n_samples, 1))
        device_embeddings.append(leaf_embeddings)

    return residuals_devices, device_models, device_embeddings


def edge_layer_boosting(edge_groups, device_embeddings, residuals_devices, le, num_classes,
                        X_finetune=None, y_finetune=None):
    """
    Edge-level sequential boosting.

    Args:
        edge_groups: list of lists
            Each element is a list of device indices belonging to that edge
        device_embeddings: list of np.ndarray
            Leaf embeddings from device-level boosting
        residuals_devices: list of np.ndarray
            Residuals from device-level boosting
        le: LabelEncoder
        num_classes: int
        X_finetune, y_finetune: optional finetuning datasets

    Returns:
        edge_outputs: list of np.ndarray
            Predictions (n_samples_edge, num_classes) per edge
        edge_models: list of list
            Trained models per edge
        residuals_edges: list of np.ndarray
            Final residuals per edge
        edge_embeddings: list of np.ndarray
            One-hot leaf embeddings per edge
    """

    edge_models = []
    residuals_edges = []
    edge_embeddings_list = []
    edge_outputs = []

    for edge_idx, edge_devices in enumerate(edge_groups):
        if len(edge_devices) == 0:
            edge_models.append([])
            residuals_edges.append(None)
            edge_embeddings_list.append(None)
            edge_outputs.append(None)
            continue

        # -----------------------------
        # Stack device embeddings and residuals with padding
        # -----------------------------
        embeddings_list = [device_embeddings[i] for i in edge_devices]
        residual_list = [residuals_devices[i] for i in edge_devices]

        max_cols = max(e.shape[1] for e in embeddings_list)
        padded_embeddings = [
            np.pad(e, ((0, 0), (0, max_cols - e.shape[1])), mode="constant")
            for e in embeddings_list
        ]

        X_edge = np.vstack(padded_embeddings)
        residual_edge = np.vstack(residual_list)

        models_per_edge = []

        # Sequential boosting per edge
        for t in range(config.get("edge_boosting_rounds", 1)):
            y_pseudo = residual_edge.argmax(axis=1)

            # Stop if all pseudo-labels are the same
            if len(np.unique(y_pseudo)) < 2:
                break

            # Train LightGBM on pseudo-labels
            model = train_lightgbm(
                X_edge, y_pseudo, X_finetune, y_finetune,
                config["early_stopping_rounds_edge"], config["learning_rate_edge"], num_classes=num_classes
            )

            pred_proba = predict_proba_fixed(model, X_edge, num_classes, le=le)

            # Update residuals (sequential boosting)
            residual_edge -= pred_proba
            residual_edge = np.clip(residual_edge, 0.0, None)

            models_per_edge.append(model)

        # Store results
        edge_models.append(models_per_edge)
        residuals_edges.append(residual_edge)
        edge_embeddings_list.append(X_edge)

        # Compute edge outputs using ensemble of models
        if len(models_per_edge) > 0:
            y_preds = []
            for m in models_per_edge:
                y_pred = m.predict_proba(X_edge)
                # Pad predictions if num_classes mismatch
                if y_pred.shape[1] < num_classes:
                    y_pred = np.pad(y_pred, ((0, 0), (0, num_classes - y_pred.shape[1])), mode='constant')
                y_preds.append(y_pred)

            # Stack and average
            y_preds_stack = np.stack(y_preds, axis=0)  # (n_models, n_samples_edge, num_classes)
            y_pred_edge = np.mean(y_preds_stack, axis=0)  # average over models
            edge_outputs.append(y_pred_edge)
        else:
            edge_outputs.append(None)

    return edge_outputs, edge_models, residuals_edges, edge_embeddings_list


def global_layer_bayesian_aggregation(edge_outputs, edge_embeddings, y_true_per_edge,
                                      residuals_edges=None, num_classes=2, verbose=True):
    """
    Global (server) Bayesian aggregation using edge embeddings,
    weighting edges by their residual errors, with explicit alpha/beta calculation.
    """

    # Step 1: Filter valid edges
    valid_edges = [
        i for i, (out, emb) in enumerate(zip(edge_outputs, edge_embeddings))
        if out is not None and emb is not None
    ]
    if len(valid_edges) == 0:
        raise ValueError("No valid edges found!")

    # Step 2: Pad edge_embeddings to same dimension
    max_embed_cols = max(edge_embeddings[i].shape[1] for i in valid_edges)
    X_global_list = [
        np.pad(edge_embeddings[i], ((0, 0), (0, max_embed_cols - edge_embeddings[i].shape[1])), mode='constant')
        for i in valid_edges
    ]
    X_global = np.vstack(X_global_list)

    # Step 3: Pad edge_outputs to same number of classes
    H_global_list = []
    for i in valid_edges:
        out = edge_outputs[i]
        if out.shape[1] < num_classes:
            out_padded = np.pad(out, ((0, 0), (0, num_classes - out.shape[1])), mode='constant')
        else:
            out_padded = out
        H_global_list.append(out_padded)
    H_global = np.vstack(H_global_list)

    # Step 4: Align/pad true labels to match number of samples per edge
    y_global_list = []
    for idx, i in enumerate(valid_edges):
        n_rows = H_global_list[idx].shape[0]
        labels = y_true_per_edge[i]

        # Convert scalar to 1D array
        if np.isscalar(labels):
            labels = np.array([labels])
        else:
            labels = np.array(labels)

        if len(labels) < n_rows:
            labels = np.pad(labels, (0, n_rows - len(labels)), mode='edge')
        elif len(labels) > n_rows:
            labels = labels[:n_rows]  # truncate if too long

        y_global_list.append(labels)
    y_global = np.hstack(y_global_list)

    n_samples = X_global.shape[0]

    # Step 5: Prepare edge weights
    if residuals_edges is not None:
        edge_weights = np.hstack([
            1.0 / (np.mean(residuals_edges[i] ** 2, axis=1) + 1e-6)
            for i in valid_edges
        ])
    else:
        edge_weights = np.ones(n_samples)

    edge_weights_norm = edge_weights / np.sum(edge_weights)
    if verbose:
        print("Normalized edge weights per sample:", edge_weights_norm)

    # Step 6: Calculate alpha (weighted mean of edge outputs)
    alpha = np.sum(H_global * edge_weights_norm[:, None], axis=0)
    if verbose:
        print("Alpha (weighted mean of H_global per class):", alpha)

    # Step 7: Convert true labels to one-hot
    y_onehot = np.zeros((n_samples, num_classes))
    y_onehot[np.arange(n_samples), y_global] = 1

    # Step 8: Calculate global residuals
    global_residuals = y_onehot - H_global

    # Step 9: Calculate beta (weighted mean of residuals)
    beta = np.sum(global_residuals * edge_weights_norm[:, None], axis=0)
    if verbose:
        print("Beta (weighted mean of residuals per class):", beta)

    # Step 10: Compute global predictions using alpha + beta
    logits = H_global + beta
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    y_global_pred = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # Step 11: Prepare posterior parameters
    theta_global = {"alpha": alpha, "beta": beta}

    return y_global_pred, global_residuals, theta_global


# ============================================================
# Forward pass with Device → Edge → Gossip
# ============================================================

def forward_pass(devices_data, edge_groups, le, num_classes, X_finetune, y_finetune,
                 residuals_devices=None, device_models=None):
    """
    Perform a single forward pass through:
    - Device-level boosting
    - Edge-level boosting
    - Gossip/global aggregation

    Returns:
        device_models: list of trained device models per device
        edge_models: list of trained edge models per edge
        theta_global: dict of global Bayesian parameters
        residuals_devices: list of residuals per device
        residuals_edges: list of residuals per edge
        y_global_pred: np.ndarray of global predictions
        device_embeddings: list of device embeddings
        y_global_true: np.ndarray of true labels aligned with global predictions
        global_acc: float, global accuracy
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
    assert device_embeddings is not None, "Device embeddings returned as None!"

    # -----------------------------
    # Edge Layer
    # -----------------------------
    edge_outputs, edge_models, residuals_edges, edge_embeddings = edge_layer_boosting(
        edge_groups, device_embeddings, residuals_devices, le, num_classes, X_finetune, y_finetune
    )

    y_true_per_edge = []

    for edge_idx, edge_devices in enumerate(edge_groups):
        labels_edge_list = []
        for dev_idx in edge_devices:
            # Original labels for this device
            y_dev = np.array(y_finetune[dev_idx])

            # Number of samples in this device's embeddings
            num_samples = device_embeddings[dev_idx].shape[0]

            if y_dev.size == 1:
                # Repeat single label
                y_dev = np.full((num_samples,), y_dev.item())
            elif y_dev.size == num_samples:
                pass  # Already matches
            else:
                raise ValueError(f"Mismatch in device {dev_idx}: y_dev has {y_dev.size} samples, "
                                 f"but device_embeddings has {num_samples} samples.")

            labels_edge_list.append(y_dev)

        # Stack labels for all devices in this edge
        labels_edge = np.hstack(labels_edge_list)
        y_true_per_edge.append(labels_edge)

    # Flatten all edges to match global predictions
    y_global_true = np.hstack(y_true_per_edge)

    # -----------------------------
    # Gossip / Global Layer
    # -----------------------------
    y_global_pred, global_residuals, theta_global = global_layer_bayesian_aggregation(
        edge_outputs=edge_outputs,
        edge_embeddings=edge_embeddings,
        y_true_per_edge=y_true_per_edge,
        residuals_edges=residuals_edges,
        num_classes=num_classes
    )

    # Compute global accuracy
    # global_acc = compute_accuracy(y_global_true, y_global_pred.argmax(axis=1))

    return device_models, edge_models, edge_outputs, theta_global, residuals_devices, residuals_edges, \
        y_global_pred, device_embeddings, edge_embeddings, y_global_true, y_true_per_edge


def backward_pass(edge_groups, device_models, edge_models,
                  device_embeddings, edge_embeddings,
                  y_true_per_edge, residuals_devices,
                  residuals_edges, num_classes,
                  max_iter=3, verbose=True):
    """
    HPFL Backward Pass: propagate residuals from global -> edge -> device
    and refine models iteratively (residual feedback mechanism).

    Args:
        edge_groups: list of lists, devices per edge
        device_models: list of lists of device-level models
        edge_models: list of lists of edge-level models
        device_embeddings: list of np.ndarray, device embeddings
        edge_embeddings: list of np.ndarray, edge embeddings
        y_true_per_edge: list of np.ndarray, true labels per edge
        residuals_devices: list of np.ndarray
        residuals_edges: list of np.ndarray
        num_classes: int
        max_iter: int, number of feedback iterations
        verbose: bool

    Returns:
        Updated residuals_devices, device_models, residuals_edges, edge_models
    """

    for iteration in range(max_iter):
        if verbose:
            print(f"\n--- Backward Pass Iteration {iteration + 1} ---")

        # 1️⃣ Compute global residuals
        y_global_pred, global_residuals, _ = global_layer_bayesian_aggregation(
            edge_outputs=[
                np.mean([m.predict_proba(edge_embeddings[i])
                         for m in edge_models[i]], axis=0)
                if edge_models[i] else None
                for i in range(len(edge_groups))
            ],
            edge_embeddings=edge_embeddings,
            y_true_per_edge=y_true_per_edge,
            residuals_edges=residuals_edges,
            num_classes=num_classes,
            verbose=verbose
        )

        # 2️⃣ Propagate residuals downward to edges
        for edge_idx, devices in enumerate(edge_groups):
            if not devices or edge_embeddings[edge_idx] is None:
                continue

            # Edge-level residuals
            r_e = global_residuals[:edge_embeddings[edge_idx].shape[0], :]
            r_e_scaled = config["learning_rate_edge"] * r_e

            # Sequentially update edge models
            for model in edge_models[edge_idx]:
                pseudo_labels = r_e_scaled.argmax(axis=1)
                if len(np.unique(pseudo_labels)) > 1:
                    train_lightgbm(edge_embeddings[edge_idx], pseudo_labels, config["early_stopping_rounds_backward"],
                                   config["learning_rate_device_backward"],num_classes=num_classes)

            # 3️⃣ Propagate residuals to devices within this edge
            start_idx = 0
            for d_idx in devices:
                if device_embeddings[d_idx] is None:
                    continue

                n_samples = device_embeddings[d_idx].shape[0]
                r_d = r_e_scaled[start_idx:start_idx + n_samples, :]
                r_d_scaled = config["learning_rate_device"] * r_d

                # Sequentially update device models
                for model in device_models[d_idx]:
                    pseudo_labels = r_d_scaled.argmax(axis=1)
                    if len(np.unique(pseudo_labels)) > 1:
                        train_lightgbm(device_embeddings[d_idx], pseudo_labels,config["early_stopping_rounds_backward"],
                                       config["learning_rate_device_backward"],num_classes=num_classes)

                # Update device residuals
                residuals_devices[d_idx] -= r_d_scaled
                residuals_devices[d_idx] = np.clip(residuals_devices[d_idx], 0.0, None)

                start_idx += n_samples

            # Update edge residuals
            residuals_edges[edge_idx] -= r_e_scaled
            residuals_edges[edge_idx] = np.clip(residuals_edges[edge_idx], 0.0, None)

    return residuals_devices, device_models, residuals_edges, edge_models


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


def safe_edge_output(edge_outputs, num_classes):
    """
    Ensures each edge output is 2D with shape (n_samples, num_classes).
    If outputs are missing columns, pad with zeros.
    If outputs are 1D, convert to one-hot.
    """
    safe_list = []
    for i, H in enumerate(edge_outputs):
        H = np.asarray(H)

        # Case 1: 1D vector of labels -> convert to one-hot
        if H.ndim == 1:
            H_onehot = np.zeros((H.shape[0], num_classes))
            H_onehot[np.arange(H.shape[0]), H.astype(int)] = 1
            H = H_onehot

        # Case 2: Single row (e.g. (num_classes,)) -> reshape
        elif H.ndim == 1 and H.shape[0] == num_classes:
            H = H.reshape(1, -1)

        # Case 3: Already 2D but wrong num_classes -> pad or truncate
        elif H.ndim == 2:
            if H.shape[1] < num_classes:
                # pad missing columns with zeros
                pad_width = num_classes - H.shape[1]
                H = np.pad(H, ((0, 0), (0, pad_width)), mode='constant')
            elif H.shape[1] > num_classes:
                # truncate extra columns
                H = H[:, :num_classes]

        else:
            raise ValueError(f"[safe_edge_output] Unexpected shape {H.shape} for edge {i}")

        safe_list.append(H)

    return safe_list


def compute_accuracy(y_true, y_pred):
    """
    Compute accuracy safely with debug info.

    Args:
        y_true: array-like, true labels
        y_pred: array-like, predicted labels

    Returns:
        float: accuracy
    """
    y_true_np = np.array(y_true).flatten()
    y_pred_np = np.array(y_pred).flatten()

    # Debug info
    print("DEBUG compute_accuracy:")
    print(f"  y_true shape: {y_true_np.shape}")
    print(f"  y_pred shape: {y_pred_np.shape}")
    print(f"  y_true sample: {y_true_np[:10]}")
    print(f"  y_pred sample: {y_pred_np[:10]}")

    # Ensure same length
    if y_true_np.shape[0] != y_pred_np.shape[0]:
        raise ValueError(f"Shape mismatch: y_true has {y_true_np.shape[0]} samples, "
                         f"but y_pred has {y_pred_np.shape[0]} samples.")

    acc = np.mean(y_true_np == y_pred_np)
    return float(acc)


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


def hpfl_train_with_accuracy(devices_data, edge_groups, le, num_classes,
                             X_finetune, y_finetune, verbose=True):
    """
    HPFL training loop with forward/backward passes and accuracy tracking.

    Args:
        devices_data: list of tuples (X_train, _, y_train, _)
        edge_groups: list of lists, devices per edge
        le: LabelEncoder
        num_classes: int
        X_finetune, y_finetune: optional fine-tuning data
        verbose: bool

    Returns:
        device_models: list of device-level models per device
        edge_models: list of edge-level models per edge
        residuals_devices: list of residuals per device
        residuals_edges: list of residuals per edge
        y_global_pred: final global predictions
        device_embeddings: list of device embeddings
        edge_embeddings: list of edge embeddings
        device_accs: list of mean device accuracies per epoch
        edge_accs: list of mean edge accuracies per epoch
        global_accs: list of global accuracies per epoch
    """

    device_accs, edge_accs, global_accs = [], [], []
    residuals_devices = [None] * len(devices_data)
    device_models = [None] * len(devices_data)
    residuals_edges = [None] * len(edge_groups)
    edge_models = [None] * len(edge_groups)

    device_embeddings = [None] * len(devices_data)
    edge_embeddings = [None] * len(edge_groups)

    for epoch in range(config["epoch"]):
        if verbose:
            print(f"\n=== Epoch {epoch + 1}/{config['epoch']} ===")

        # -----------------------------
        # Forward Pass
        # -----------------------------
        device_models, edge_models, edge_outputs, theta_global, \
            residuals_devices, residuals_edges, y_global_pred, \
            device_embeddings, edge_embeddings, y_global_true, y_true_per_edge = forward_pass(
            devices_data, edge_groups, le, num_classes,
            X_finetune, y_finetune,
            residuals_devices=residuals_devices,
            device_models=device_models
        )

        # -----------------------------
        # Backward Pass (Residual Feedback)
        # -----------------------------
        residuals_devices, device_models, residuals_edges, edge_models = backward_pass(
            edge_groups, device_models, edge_models,
            device_embeddings, edge_embeddings,
            y_true_per_edge, residuals_devices,
            residuals_edges, num_classes, verbose=verbose)

        # -----------------------------
        # Compute device-level accuracy
        # -----------------------------
        device_epoch_acc_list = []
        for i, c in enumerate(devices_data):
            X_test, _, y_test, _ = c
            device_preds = np.zeros((safe_array(X_test).shape[0], num_classes))
            for mdl in device_models[i]:
                device_preds += predict_proba_fixed(mdl, X_test, num_classes, le)
            device_preds = device_preds.argmax(axis=1)
            device_epoch_acc_list.append(compute_accuracy(y_test, device_preds))
        device_acc_epoch = np.mean(device_epoch_acc_list)

        # -----------------------------
        # Compute edge-level accuracy
        # -----------------------------
        edge_acc_list = []
        for edge_idx, edge_devices in enumerate(edge_groups):
            if len(edge_devices) == 0 or edge_outputs[edge_idx] is None:
                continue
            # Stack true labels from all devices in this edge
            y_edge_true = np.hstack([np.array(devices_data[d_idx][2]) for d_idx in edge_devices])
            y_edge_pred = edge_outputs[edge_idx].argmax(axis=1)
            edge_acc_list.append(compute_accuracy(y_edge_true, y_edge_pred))
        edge_acc_epoch = np.mean(edge_acc_list) if edge_acc_list else 0.0

        # -----------------------------
        # Compute global accuracy
        # -----------------------------
        global_acc = compute_accuracy(y_global_true, y_global_pred.argmax(axis=1))

        # -----------------------------
        # Store accuracies
        # -----------------------------
        device_accs.append(device_acc_epoch)
        edge_accs.append(edge_acc_epoch)
        global_accs.append(global_acc)

        if verbose:
            print(f"[Epoch {epoch + 1}] Device Acc: {device_acc_epoch:.4f}, "
                  f"Edge Acc: {edge_acc_epoch:.4f}, Global Acc: {global_acc:.4f}")

    return device_models, edge_models, residuals_devices, residuals_edges, \
        y_global_pred, device_embeddings, edge_embeddings, \
        device_accs, edge_accs, global_accs


# ============================================================
# Training Loop Example
# ============================================================

if __name__ == "__main__":
    folder_path = "CIC_IoT_IDAD_Dataset_2024"
    today_str = datetime.now().strftime("%Y-%m-%d")
    log_path_str = os.path.join("logs", f"{folder_path}_{today_str}")
    os.makedirs(log_path_str, exist_ok=True)

    # 1. Preprocess
    X_final, y_final, X_pretrain, y_pretrain, X_finetune, y_finetune, X_test, y_test = preprocess_data_safe(
        log_path_str, folder_path, scaler_type='minmax'
    )

    # 2. Encode labels
    le = LabelEncoder()
    le.fit(y_pretrain)
    num_classes = len(np.unique(y_pretrain))

    # 3. Partition fine-tune data for devices & edges (non-IID)
    devices_data, edge_groups = dirichlet_partition_for_devices_edges_non_iid(
        X_finetune, y_finetune,  # use fine-tune data
        num_devices=config["n_device"],
        device_per_edge=config["device_per_edge"],
        n_edges=config["n_edges"],
        alpha=0.5,
        seed=42
    )

    # 4. Train HPFL model with accuracy tracking
    device_models, edge_models, residuals_devices, residuals_edges, \
        y_global_pred, device_embeddings, edge_embeddings, \
        device_accs, edge_accs, global_accs = hpfl_train_with_accuracy(
        devices_data=devices_data,
        edge_groups=edge_groups,
        le=le,
        num_classes=num_classes,
        X_finetune=X_finetune,
        y_finetune=y_finetune,
        verbose=True
    )

    # 5. Plot accuracy trends
    plt.figure(figsize=(10, 6))
    plt.plot(device_accs, label="Device-level Acc", marker='o')
    plt.plot(edge_accs, label="Edge-level Acc", marker='s')
    plt.plot(global_accs, label="Global Acc", marker='^')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Hierarchical PFL Accuracy Trends")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.show()

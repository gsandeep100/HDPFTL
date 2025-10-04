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
import torch
from lightgbm import early_stopping, LGBMClassifier, LGBMRegressor, log_evaluation
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
    "boosting": "gbdt",
    "random_seed": 42,
    "n_edges": 5,
    "n_device": 5,
    "device_per_edge": 1,
    "epoch": 5,
    "device_boosting_rounds": 5,
    "edge_boosting_rounds": 5,
    "num_iterations_device": 100,
    "num_iterations_edge": 100,
    "variance_prune": True,
    "variance_threshold": 1e-4,
    "bayes_n_samples": 500,
    "bayes_n_tune": 500,
    "save_results": True,
    "results_path": "results",
    "isotonic_min_positives": 5,
    "max_cores": 2,
    "n_estimators": 50,
    "num_leaves": 64,
    "alpha": 1.0,
    "min_child_samples": 10,
    "learning_rate_device": 0.01,
    "learning_rate_edge": 0.1,
    "learning_rate_backward": 0.1,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "early_stopping_rounds_device": 20,
    "early_stopping_rounds_edge": 100,
    "early_stopping_rounds_backward": 10,
    "class_weight": "balanced",
    "lambda_l1": 1.0,
    "lambda_l2": 1.0,
    "device": "cpu"

}

rng = np.random.default_rng(config["random_seed"])
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series]
DeviceData = Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]


# ============================================================
# LightGBM Training
# ============================================================

def train_lightgbm(X_train, y_train, X_valid=None, y_valid=None, early_stopping_rounds=None,
                   learning_rate=None, num_iteration=None, num_classes=None, init_model=None):
    """
    Train a LightGBM classifier safely with both training and validation monitoring.
    Supports continuing from a previous model via `init_model`.

    Args:
        X_train (array-like): Training features.
        y_train (array-like): Training labels.
        X_valid (array-like, optional): Validation features (default: None).
        y_valid (array-like, optional): Validation labels (default: None).
        early_stopping_rounds (int, optional): Stop if validation score does not improve
            after this many rounds (default: None).
        learning_rate (float, optional): Learning rate for boosting (default: 0.01).
        num_iteration (int, optional): Maximum number of boosting rounds (default: 5000).
        num_classes (int, optional): Number of target classes. If None, inferred from labels.
        init_model (LGBMClassifier, optional): Previous model to continue training from.

        Note:- Compare multi_logloss for train vs. valid:

        If train logloss keeps going down but valid stops improving → overfitting.

        If neither improves much → weak features or wrong parameters.

        OR in other Words

        If train ↓ and valid ↑ → overfitting (trees too complex, regularize more).

        If train flat + valid flat → weak features (model can’t learn).

        If train ↓ and valid ↓ together → good learning.

    Returns:
        model (lightgbm.LGBMClassifier): Trained LightGBM model.
    """

    fit_kwargs = {}

    # Convert inputs to NumPy arrays
    X_np = safe_array(X_train)
    y_np = safe_array(y_train)
    X_valid_np = safe_array(X_valid) if X_valid is not None else None
    y_valid_np = safe_array(y_valid) if y_valid is not None else None

    if num_classes is None:
        num_classes = len(np.unique(y_np))

    objective = "multiclass" if num_classes > 2 else "binary"
    num_class = num_classes if num_classes > 2 else None

    model = lgb.LGBMClassifier(
        # Objective & classes
        boosting=config["boosting"],
        objective=objective,
        num_class=num_class,

        # Boosting setup
        n_estimators=config["n_estimators"],
        learning_rate=learning_rate,
        num_iterations=num_iteration,

        # Tree complexity
        num_leaves=config["num_leaves"],
        max_depth=config["max_depth"],

        # Data constraints & regularization
        min_child_samples=config["min_child_samples"],
        class_weight=config["class_weight"],
        lambda_l1=config["lambda_l1"] * 2,
        lambda_l2=config["lambda_l2"] * 2,
        feature_fraction=config["feature_fraction"],

        random_state=config["random_seed"],
        device=config["device"],
        gpu_platform_id=0,
        gpu_device_id=0,
        verbose=-1
    )

    if X_valid_np is not None and y_valid_np is not None and early_stopping_rounds:
        # Filter validation set to match training classes
        mask = np.isin(y_valid_np, np.unique(y_np))
        if mask.sum() < 10:  # avoid too few samples after filtering
            mask = np.ones_like(y_valid_np, dtype=bool)

        X_valid_filtered = X_valid_np[mask]
        y_valid_filtered = y_valid_np[mask]

        fit_kwargs.update({
            "eval_set": [(X_np, y_np), (X_valid_filtered, y_valid_filtered)],
            "eval_metric": "multi_logloss" if num_classes > 2 else "logloss",
            "eval_names": ["train", "valid"],
            "callbacks": [
                early_stopping(early_stopping_rounds),
                log_evaluation(10)
            ],
        })

    # Pass previous model to continue boosting
    if init_model is not None:
        fit_kwargs["init_model"] = init_model

    # Train the model
    model.fit(safe_array(X_np), safe_array(y_np), **fit_kwargs)

    return model


# ============================================================
# Predict probabilities safely, filling missing classes
# ============================================================

def predict_proba_fixed(model, X, num_classes, le=None):
    """
    Safe wrapper for LightGBM's predict_proba:
    - Ensures correct feature count (pads/slices X if needed).
    - Ensures correct class count (fills missing classes if needed).
    - Always returns shape (n_samples, num_classes).
    """
    X_np = safe_array(X)

    # --- Feature alignment ---
    n_features_model = getattr(model, "n_features_in_", X_np.shape[1])
    if X_np.shape[1] > n_features_model:
        X_np = X_np[:, :n_features_model]  # slice down
    elif X_np.shape[1] < n_features_model:
        pad_width = n_features_model - X_np.shape[1]
        X_np = np.pad(X_np, ((0, 0), (0, pad_width)), mode="constant")

    # --- Prediction ---
    pred = model.predict_proba(X_np)
    pred = np.atleast_2d(pred)

    # --- Class alignment ---
    if pred.shape[1] == num_classes:
        return pred

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
            if cls < num_classes:
                full[:, int(cls)] = pred[:, i]

    # --- Normalize ---
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

def device_layer_boosting(devices_data, residuals_devices, device_models, le, num_classes,
                          X_finetune=None, y_finetune=None):
    """
    Device-level sequential boosting (classification version) with safe incremental boosting.

    Returns:
        residuals_devices: list of np.ndarray, final residuals per device
        device_models: list of lists, trained models per device
        device_embeddings: list of np.ndarray, one-hot leaf embeddings per device
        y_true_devices: list of np.ndarray, y_finetune per device
    """
    print("""
    ************************************************************
    *                                                          *
    *                  STARTING DEVICE LAYER                   *
    *                                                          *
    ************************************************************
    """)

    device_embeddings = []

    # Loop through each device (each device gets its own mini boosting process)
    for idx, dev_tuple in enumerate(devices_data):
        X_train, _, y_train, _ = dev_tuple
        n_samples = X_train.shape[0]

        # ----------------------------
        # Initialize residuals per device
        # ----------------------------
        if residuals_devices[idx] is None:
            # One-hot encode labels → residuals start as perfect truth signal
            residual = np.zeros((n_samples, num_classes), dtype=float)
            residual[np.arange(n_samples), le.transform(y_train)] = 1.0
        else:
            # If we already had previous round residuals, reuse them
            residual = residuals_devices[idx].copy()

        # Retrieve models already trained for this device (if incremental learning is on)
        models_per_device = device_models[idx] if device_models[idx] else []

        # ----------------------------
        # Sequential boosting for this device
        # ----------------------------
        for t in range(config["device_boosting_rounds"]):
            # Pseudo-labels from residuals: take the class with max residual
            y_pseudo = residual.argmax(axis=1)

            # If pseudo-labels collapse into one class → skip (no useful training signal)
            if len(np.unique(y_pseudo)) < 2:
                logging.debug(f"Device {idx}, round {t}: single class, skipping.")
                break

            # Warm-start: only reuse previous model if input shapes match
            init_model = models_per_device[-1] if models_per_device else None

            # Train LightGBM on pseudo-labels (residual-driven labels)
            model = train_lightgbm(
                X_train, y_pseudo,
                X_valid=X_finetune[:, :X_train.shape[1]] if X_finetune is not None else None,
                y_valid=y_finetune,
                early_stopping_rounds=config["early_stopping_rounds_device"],
                learning_rate=config["learning_rate_device"],
                num_iteration=config["num_iterations_device"],
                num_classes=num_classes,
                init_model=init_model
            )

            # Get probability predictions from the model
            pred_proba = predict_proba_fixed(model, X_train, num_classes, le=le)

            # ----------------------------
            # Update residuals
            # ----------------------------
            residual -= pred_proba          # reduce mass where model is confident
            residual = np.clip(residual, 0.0, None)  # keep residuals non-negative

            # Store model for this device
            models_per_device.append(model)

        # Save final residuals + trained models
        residuals_devices[idx] = residual
        device_models[idx] = models_per_device

        # ----------------------------
        # Compute device embedding (leaf indices one-hot encoding)
        # ----------------------------
        leaf_indices_list = [get_leaf_indices(mdl, X_train) for mdl in models_per_device]
        if leaf_indices_list:
            # Concatenate leaf indices from all boosting rounds
            leaf_indices_concat = np.hstack(leaf_indices_list)

            # One-hot encode leaf positions across all trees
            leaf_embeddings = np.zeros((n_samples, np.max(leaf_indices_concat) + 1))
            leaf_embeddings[np.arange(n_samples)[:, None], leaf_indices_concat] = 1.0
        else:
            # If no models trained, fallback to dummy embedding
            leaf_embeddings = np.zeros((n_samples, 1))

        device_embeddings.append(leaf_embeddings)

    # ----------------------------
    # Convert y_finetune into per-device arrays (for eval/finetune later)
    # ----------------------------
    y_true_devices = [np.array(y_finetune[d]) for d in range(len(devices_data))]

    return residuals_devices, device_models, device_embeddings, y_true_devices


def edge_layer_boosting(edge_groups, device_embeddings, residuals_devices,
                        le, num_classes, X_finetune=None, y_finetune=None,
                        device_weights=None):
    """
    Vectorized edge-level sequential boosting with index tracking.

    Returns:
        edge_outputs: list of final boosted predictions per edge
        edge_models: list of models trained per edge
        residuals_edges: list of residuals per edge
        edge_embeddings_list: list of embeddings per edge
        global_pred_matrix: stacked predictions from all edges (n_samples, num_classes)
        edge_sample_slices: dict {device_idx: global_indices}
    """

    print("""
    ************************************************************
    *                  STARTING EDGE LAYER                     *
    *   Devices -> Edge-level boosted ensemble -> Edge output   *
    ************************************************************
    """)

    edge_models = []
    residuals_edges = []
    edge_embeddings_list = []
    edge_outputs = []
    global_pred_blocks = []
    edge_sample_slices = {}

    global_offset = 0  # tracks row position in global prediction matrix

    for edge_idx, edge_devices in enumerate(edge_groups):
        if len(edge_devices) == 0:
            edge_models.append([])
            residuals_edges.append(None)
            edge_embeddings_list.append(None)
            edge_outputs.append(None)
            print(f"Edge {edge_idx}: no devices, skipping.")
            continue

        # -----------------------------
        # 1. Gather embeddings + residuals
        # -----------------------------
        embeddings_list = [device_embeddings[i] for i in edge_devices]
        residual_list = [residuals_devices[i] for i in edge_devices]

        if device_weights is None:
            weights = np.ones(len(edge_devices))
        else:
            weights = np.array([device_weights[i] for i in edge_devices])

        # pad embeddings to same width
        max_cols = max(e.shape[1] for e in embeddings_list)
        padded_embeddings = [
            np.pad(e, ((0, 0), (0, max_cols - e.shape[1])), mode="constant")
            for e in embeddings_list
        ]
        X_edge = np.vstack(padded_embeddings)

        # weighted residual stacking
        residual_edge = np.vstack([r * w for r, w in zip(residual_list, weights)])
        residual_edge = np.clip(residual_edge, 0.0, None)

        models_per_edge = []

        # -----------------------------
        # 2. Sequential boosting
        # -----------------------------
        boosting_rounds = config["edge_boosting_rounds"]
        for t in range(boosting_rounds):
            y_pseudo = np.argmax(residual_edge, axis=1)
            unique_classes = np.unique(y_pseudo)
            if len(unique_classes) < 2:
                print(f"Edge {edge_idx}, round {t}: only {len(unique_classes)} class(es), skipping boosting.")
                break

            init_model = models_per_edge[-1] if models_per_edge else None
            X_valid_slice = X_finetune[:, :X_edge.shape[1]] if X_finetune is not None else None

            model = train_lightgbm(
                X_edge, y_pseudo,
                X_valid=X_valid_slice,
                y_valid=y_finetune,
                early_stopping_rounds=config["early_stopping_rounds_edge"],
                learning_rate=config["learning_rate_edge"],
                num_iteration=config["num_iterations_edge"],
                num_classes=num_classes,
                init_model=init_model
            )

            pred_proba = predict_proba_fixed(model, X_edge, num_classes, le=le)
            residual_edge = np.clip(residual_edge - pred_proba, 0.0, None)
            models_per_edge.append(model)

        # -----------------------------
        # 3. Store results
        # -----------------------------
        edge_models.append(models_per_edge)
        residuals_edges.append(residual_edge)
        edge_embeddings_list.append(X_edge)

        if models_per_edge:
            # Average predictions over boosting rounds
            model_preds = np.array([m.predict_proba(X_edge) for m in models_per_edge])
            edge_pred_avg = np.mean(model_preds, axis=0)
        else:
            edge_pred_avg = np.zeros((X_edge.shape[0], num_classes))

        edge_outputs.append(edge_pred_avg)

        # --- assign global indices per device ---
        row_start = 0
        for dev_idx, emb in zip(edge_devices, embeddings_list):
            n_dev = emb.shape[0]
            row_end = row_start + n_dev
            global_idx = np.arange(global_offset + row_start, global_offset + row_end)
            edge_sample_slices[dev_idx] = global_idx
            row_start = row_end

        global_pred_blocks.append(edge_pred_avg)
        global_offset += X_edge.shape[0]

    # -----------------------------
    # Final global prediction matrix
    # -----------------------------
    if global_pred_blocks:
        global_pred_matrix = np.vstack(global_pred_blocks)
    else:
        global_pred_matrix = np.empty((0, num_classes))

    return edge_outputs, edge_models, residuals_edges, edge_embeddings_list, global_pred_matrix, edge_sample_slices



def global_layer_bayesian_aggregation(edge_outputs, edge_embeddings, y_true_per_edge,
                                       edge_groups,
                                      residuals_edges=None, num_classes=2,
                                      edge_sample_slices=None, verbose=True):
    """
    Global Bayesian aggregation using edge outputs with proper index mapping.
    """

    # Collect only valid edges
    valid_edges = [i for i, (out, emb) in enumerate(zip(edge_outputs, edge_embeddings))
                   if out is not None and emb is not None]

    if not valid_edges:
        raise ValueError("No valid edges found!")

    # Compute total number of samples from edge slices
    if edge_sample_slices is not None:
        n_samples_total = max([idxs[-1] + 1 for idxs in edge_sample_slices.values()])
    else:
        n_samples_total = sum(edge_outputs[i].shape[0] for i in valid_edges)

    # Initialize global predictions
    H_global = np.zeros((n_samples_total, num_classes))
    y_global = np.full((n_samples_total,), -1, dtype=int)
    global_residuals = np.zeros_like(H_global)

    # Scatter edge outputs into global matrix
    for edge_idx in valid_edges:
        edge_out = edge_outputs[edge_idx]
        if edge_sample_slices is None:
            idxs = np.arange(H_global.shape[0])  # fallback
        else:
            # concatenate all device indices for this edge
            idxs = np.hstack([edge_sample_slices[d] for d in edge_groups[edge_idx]])
        H_global[idxs] = edge_out

        # Flatten true labels
        y_edge = np.hstack([y_true_per_edge[edge_idx]])  # already aligned per edge
        y_global[idxs] = y_edge

    # Compute edge weights using residuals
    if residuals_edges is not None:
        W = np.zeros((n_samples_total,))
        for edge_idx in valid_edges:
            resid = residuals_edges[edge_idx]
            idxs = np.hstack([edge_sample_slices[d] for d in edge_groups[edge_idx]])
            W[idxs] = 1.0 / (np.mean(resid**2, axis=1) + 1e-6)
        W /= np.sum(W)
    else:
        W = np.ones(n_samples_total) / n_samples_total

    # Bayesian alpha and beta
    alpha = np.sum(H_global * W[:, None], axis=0)
    y_onehot = np.zeros_like(H_global)
    y_onehot[np.arange(n_samples_total), y_global] = 1
    global_residuals = y_onehot - H_global
    beta = np.sum(global_residuals * W[:, None], axis=0)

    # Softmax predictions
    logits = H_global + beta
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    y_global_pred = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # Device slices
    device_sample_slices = []
    for edge_idx, edge_devices in enumerate(edge_groups):
        for d in edge_devices:
            device_sample_slices.append(edge_sample_slices[d])

    theta_global = {"alpha": alpha, "beta": beta}

    if verbose:
        print(f"Global alpha: {alpha}")
        print(f"Global beta: {beta}")
        print(f"Total samples: {n_samples_total}")

    return y_global_pred, global_residuals, theta_global, edge_sample_slices, device_sample_slices



# ============================================================
# Forward pass with Device → Edge → Gossip
# ============================================================

def forward_pass(devices_data, edge_groups, le, num_classes, X_finetune, y_finetune,
                 residuals_devices=None, device_models=None):
    """
    Perform a single forward pass through:
    - Device-level boosting
    - Edge-level boosting
    - Global Bayesian aggregation

    Returns:
        device_models, edge_models, edge_outputs, theta_global,
        residuals_devices, residuals_edges,
        y_global_pred, device_embeddings, edge_embeddings_list,
        y_global_true, y_true_per_edge,
        global_residuals, edge_sample_slices, y_true_devices, device_sample_slices
    """

    if residuals_devices is None:
        residuals_devices = [None] * len(devices_data)
    if device_models is None:
        device_models = [None] * len(devices_data)

    # -----------------------------
    # 1. Device Layer
    # -----------------------------
    residuals_devices, device_models, device_embeddings, y_true_devices = device_layer_boosting(
        devices_data, residuals_devices, device_models,
        le, num_classes, X_finetune, y_finetune
    )
    assert device_embeddings is not None, "Device embeddings returned as None!"

    # -----------------------------
    # 2. Edge Layer
    # -----------------------------
    # Track total samples for global prediction matrix
    n_samples = sum(e.shape[0] for e in device_embeddings)
    edge_outputs, edge_models, residuals_edges, edge_embeddings_list, global_pred_matrix, edge_sample_slices = edge_layer_boosting(
        edge_groups=edge_groups,
        device_embeddings=device_embeddings,
        residuals_devices=residuals_devices,
        le=le,
        num_classes=num_classes,
        X_finetune=X_finetune,
        y_finetune=y_finetune
    )

    # -----------------------------
    # 3. Build Global Ground Truth
    # -----------------------------
    y_global_true = np.full((n_samples,), fill_value=-1, dtype=int)
    y_true_per_edge = []

    for edge_idx, edge_devices in enumerate(edge_groups):
        labels_edge_list = []

        for dev_idx in edge_devices:
            y_dev = np.array(y_finetune[dev_idx])
            num_samples_dev = device_embeddings[dev_idx].shape[0]

            # Expand single label if needed
            if y_dev.size == 1:
                y_dev = np.full((num_samples_dev,), y_dev.item())
            elif y_dev.size != num_samples_dev:
                raise ValueError(
                    f"Device {dev_idx}: y_dev has {y_dev.size} samples, "
                    f"but embeddings has {num_samples_dev}"
                )

            labels_edge_list.append(y_dev)

            # Assign labels to global vector safely
            idxs = edge_sample_slices.get(dev_idx, np.arange(num_samples_dev))
            y_global_true[idxs] = y_dev

        # Stack per-edge labels
        labels_edge = np.hstack(labels_edge_list)
        y_true_per_edge.append(labels_edge)

    # -----------------------------
    # 4. Global Layer Aggregation
    # -----------------------------
    y_global_pred, global_residuals, theta_global, edge_sample_slices, device_sample_slices = global_layer_bayesian_aggregation(
        edge_outputs=edge_outputs,
        edge_embeddings=edge_embeddings_list,
        edge_groups=edge_groups,
        residuals_edges=residuals_edges,
        num_classes=num_classes
    )

    # -----------------------------
    # 5. Return
    # -----------------------------
    return (
        device_models,
        edge_models,
        edge_outputs,
        theta_global,
        residuals_devices,
        residuals_edges,
        y_global_pred,
        device_embeddings,
        edge_embeddings_list,
        y_global_true,
        y_true_per_edge,
        global_residuals,
        edge_sample_slices,
        y_true_devices,
        device_sample_slices
    )



def backward_pass(edge_models, device_models, edge_embeddings, device_embeddings,
                  y_true_per_edge, y_true_devices, y_global_pred=None,
                  n_classes=8, use_classification=True, verbose=True):
    """
    Hierarchical residual feedback for HPFL using LightGBM.

    - use_classification=True : fit LGBMClassifier on integer labels
    - use_classification=False: fit LGBMRegressor on residuals (continuous)

    Returns updated edge_models and device_models.
    """
    print("""
    ************************************************************
    *                                                          *
    *                  BACKWARD PASS                           *
    *                                                          *
    ************************************************************
    """)

    updated_edge_preds = []

    # -----------------------------
    # 1. Update Edge Models
    # -----------------------------
    for i, models in enumerate(edge_models):
        X_edge = edge_embeddings[i]
        y_edge_true = y_true_per_edge[i]

        if not models or X_edge is None:
            continue

        # Convert classification labels if needed
        if use_classification:
            y_edge_labels = (
                y_edge_true.argmax(axis=1).astype(int) if y_edge_true.ndim > 1 else y_edge_true.ravel().astype(int)
            )
        else:
            # Use continuous residuals (2D)
            if y_global_pred is not None:
                # Ensure shapes match
                residuals_edge = y_edge_true - y_global_pred[i][:y_edge_true.shape[0]]
            else:
                residuals_edge = y_edge_true
            y_edge_labels = np.atleast_2d(residuals_edge).astype(np.float32)

        regressors_per_edge = []
        preds_list = []

        for m in models:
            if use_classification:
                clf = LGBMClassifier(
                    n_estimators=m.n_estimators,
                    learning_rate=m.learning_rate,
                    max_depth=m.max_depth,
                    random_state=m.random_state,
                    num_class=n_classes
                )
                clf.fit(X_edge, y_edge_labels)
                preds = clf.predict_proba(X_edge)
            else:
                reg = LGBMRegressor(
                    n_estimators=m.n_estimators,
                    learning_rate=m.learning_rate,
                    max_depth=m.max_depth,
                    random_state=m.random_state
                )
                reg.fit(X_edge, y_edge_labels)
                preds = reg.predict(X_edge)
                preds = np.atleast_2d(preds).reshape(X_edge.shape[0], -1)

            regressors_per_edge.append(reg if not use_classification else clf)
            preds_list.append(preds)

        # Average predictions for this edge
        if preds_list:
            preds_avg = np.mean(preds_list, axis=0)
            updated_edge_preds.append(preds_avg)

        edge_models[i] = regressors_per_edge

    # Stack all edge predictions (num_samples_total, n_classes or 1)
    updated_edge_preds = np.vstack(updated_edge_preds) if updated_edge_preds else np.empty((0, n_classes))

    # -----------------------------
    # 2. Update Device Models
    # -----------------------------
    for i, models in enumerate(device_models):
        X_device = device_embeddings[i]
        y_device_true = y_true_devices[i]

        if not models or X_device is None:
            continue

        if use_classification:
            y_device_labels = (
                y_device_true.argmax(axis=1).astype(int) if y_device_true.ndim > 1 else y_device_true.ravel().astype(
                    int)
            )
        else:
            # Compute residuals for device using edge predictions
            # If shapes match, subtract corresponding edge predictions
            if updated_edge_preds.shape[0] >= y_device_true.shape[0]:
                residuals_device = y_device_true - updated_edge_preds[:y_device_true.shape[0], :]
            else:
                residuals_device = y_device_true
            y_device_labels = np.atleast_2d(residuals_device).astype(np.float32)

        regressors_per_device = []

        for m in models:
            if use_classification:
                clf = LGBMClassifier(
                    n_estimators=m.n_estimators,
                    learning_rate=m.learning_rate,
                    max_depth=m.max_depth,
                    random_state=m.random_state,
                    num_class=n_classes
                )
                clf.fit(X_device, y_device_labels)
                regressors_per_device.append(clf)
            else:
                reg = LGBMRegressor(
                    n_estimators=m.n_estimators,
                    learning_rate=m.learning_rate,
                    max_depth=m.max_depth,
                    random_state=m.random_state
                )
                reg.fit(X_device, y_device_labels)
                regressors_per_device.append(reg)

        device_models[i] = regressors_per_device

    if verbose:
        print("Backward hierarchical feedback completed safely.")

    return edge_models, device_models


# ======================================================================
#                           HELPER FUNCTIONS
# ======================================================================
# Utility functions for HPFL training, including:
#   • Padding and averaging predictions safely
#   • Converting inputs to NumPy arrays
#   • Ensuring edge outputs are consistent (one-hot / padded)
#   • Computing accuracy with debug support
#   • Generating edge groups
#   • Dirichlet-based partitioning for devices and edges
#   • HPFL training loop with forward/backward passes
#   • Final evaluation of device, edge, and global accuracy
# ======================================================================


def pad_predictions(pred_list, num_samples=None, num_classes=None):
    """
    Pad a list of predictions to a uniform shape for stacking.
    Any None predictions are replaced with zeros.
    """
    # Determine max dimensions if not provided
    if num_samples is None:
        num_samples = max(pred.shape[0] for pred in pred_list if pred is not None)
    if num_classes is None:
        num_classes = max(pred.shape[1] for pred in pred_list if pred is not None)

    padded_preds = []
    for pred in pred_list:
        if pred is None:
            padded = np.zeros((num_samples, num_classes), dtype=float)
        else:
            # Ensure pred has at least 2 dimensions
            pred = np.atleast_2d(pred)
            # Pad rows
            if pred.shape[0] < num_samples:
                pred = np.pad(pred, ((0, num_samples - pred.shape[0]), (0, 0)), mode='constant')
            # Pad columns
            if pred.shape[1] < num_classes:
                pred = np.pad(pred, ((0, 0), (0, num_classes - pred.shape[1])), mode='constant')
            # Truncate if necessary
            if pred.shape[0] > num_samples:
                pred = pred[:num_samples, :]
            if pred.shape[1] > num_classes:
                pred = pred[:, :num_classes]
            padded = pred
        padded_preds.append(padded)

    return np.stack(padded_preds, axis=0)


def average_predictions(pred_list, num_samples=None, num_classes=None):
    """
    Safely average predictions across models, with padding.
    """
    if not pred_list:
        return None

    # Ensure all predictions are numpy arrays
    pred_list = [np.array(p) for p in pred_list if p is not None]

    if len(pred_list) == 0:
        return None

    padded = pad_predictions(pred_list, num_samples, num_classes)
    return np.mean(padded, axis=0)  # (num_samples, num_classes)


def safe_array(X):
    """Convert input to numeric NumPy array for LightGBM."""
    if X is None:
        return None
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return X.to_numpy(dtype=np.float32)
    return np.asarray(X, dtype=np.float32)


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


def  compute_accuracy(y_true, y_pred):
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


# -----------------------------
# Step 2: Hierarchical Dirichlet Partition
# -----------------------------
def dirichlet_partition_for_devices_edges_non_iid(X, y, num_devices, n_edges, alpha=0.5, seed=42):
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


def compute_multilevel_accuracy(devices_data,
                                device_models,
                                edge_groups,
                                edge_models,
                                y_global_true,
                                y_global_pred,
                                num_classes,
                                le=None,
                                verbose=True):
    """
    Compute accuracy at device, edge, and global levels.

    Returns:
        metrics: dict with full details:
            - device_accs: list of per-device accuracies
            - device_mean, device_std
            - edge_accs: list of per-edge accuracies
            - edge_mean, edge_std
            - global_acc
    """
    # -----------------------------
    # 1. Device-Level Accuracy
    # -----------------------------
    device_accs = []
    for i, (X_test, _, y_test, _) in enumerate(devices_data):
        X_test_np = safe_array(X_test)
        device_preds = np.zeros((X_test_np.shape[0], num_classes))

        if device_models[i] is not None:
            for mdl in device_models[i]:
                device_preds += predict_proba_fixed(mdl, X_test_np, num_classes, le=le)
            device_preds /= max(len(device_models[i]), 1)

        device_accs.append(compute_accuracy(y_test, device_preds.argmax(axis=1)))

    device_mean = float(np.mean(device_accs)) if device_accs else 0.0
    device_std = float(np.std(device_accs)) if device_accs else 0.0

    # -----------------------------
    # 2. Edge-Level Accuracy
    # -----------------------------
    edge_accs = []
    for edge_idx, devices_in_edge in enumerate(edge_groups):
        if not devices_in_edge or edge_models[edge_idx] is None:
            continue

        X_edge = np.vstack([safe_array(devices_data[d][0]) for d in devices_in_edge])
        y_edge = np.hstack([devices_data[d][2] for d in devices_in_edge])

        edge_preds = np.zeros((X_edge.shape[0], num_classes))
        for mdl in edge_models[edge_idx]:
            edge_preds += predict_proba_fixed(mdl, X_edge, num_classes, le=le)
        edge_preds /= max(len(edge_models[edge_idx]), 1)

        edge_accs.append(compute_accuracy(y_edge, edge_preds.argmax(axis=1)))

    edge_mean = float(np.mean(edge_accs)) if edge_accs else 0.0
    edge_std = float(np.std(edge_accs)) if edge_accs else 0.0

    # -----------------------------
    # 3. Global Accuracy
    # -----------------------------
    global_acc = compute_accuracy(y_global_true, y_global_pred.argmax(axis=1))

    # -----------------------------
    # 4. Verbose Logging
    # -----------------------------
    if verbose:
        print(f"[Acc] Device Mean: {device_mean:.4f} ± {device_std:.4f} | "
              f"Edge Mean: {edge_mean:.4f} ± {edge_std:.4f} | "
              f"Global: {global_acc:.4f}")

    # -----------------------------
    # 5. Return Dict
    # -----------------------------
    return {
        "device_accs": device_accs,
        "device_mean": device_mean,
        "device_std": device_std,
        "edge_accs": edge_accs,
        "edge_mean": edge_mean,
        "edge_std": edge_std,
        "global_acc": float(global_acc)
    }


# ============================================================
#                  HPFL TRAINING AND EVALUATION
# ============================================================
# - hpfl_train_with_accuracy: hierarchical training loop
#   with device-level and edge-level sequential boosting.
# - evaluate_final_accuracy: compute final accuracy metrics at
#   device, edge, and global (gossip-summary) levels.
# ============================================================

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
        device_models, edge_models,
        residuals_devices, residuals_edges,
        y_global_pred, device_embeddings, edge_embeddings,
        metrics_history: dict with per-epoch means & stds
    """

    residuals_devices = [None] * len(devices_data)
    device_models = [None] * len(devices_data)
    residuals_edges = [None] * len(edge_groups)
    edge_models = [None] * len(edge_groups)

    device_embeddings = [None] * len(devices_data)
    edge_embeddings = [None] * len(edge_groups)

    num_epochs = config["epoch"]

    # --- history tracking ---
    history = {
        "device_means": [],
        "device_stds": [],
        "edge_means": [],
        "edge_stds": [],
        "global_accs": []
    }

    for epoch in range(num_epochs):
        # Pretty epoch header
        print(f"""
        ************************************************************
        *                                                          *
        *                  === Epoch {epoch + 1}/{num_epochs} ===  *
        *                                                          *
        ************************************************************
        """)

        # -----------------------------
        # 1. Forward Pass
        # -----------------------------
        device_models, edge_models, edge_outputs, theta_global, \
            residuals_devices, residuals_edges, y_global_pred, \
            device_embeddings, edge_embeddings, y_global_true, \
            y_true_per_edge, global_residuals, edge_sample_slices, \
            y_true_devices, device_sample_slices = forward_pass(
                devices_data, edge_groups, le, num_classes,
                X_finetune, y_finetune,
                residuals_devices=residuals_devices,
                device_models=device_models
        )


        # -----------------------------
        # 2. Compute Multi-Level Accuracy
        # -----------------------------
        metrics = compute_multilevel_accuracy(
            devices_data,
            device_models,
            edge_groups,
            edge_models,
            y_global_true,
            y_global_pred,
            num_classes,
            le=le,
            verbose=True
        )

        # -----------------------------
        # 3. Backward Pass using global residuals
        # -----------------------------
        """
        edge_models, device_models = backward_pass(
            edge_models=edge_models,
            device_models=device_models,
            edge_embeddings=edge_embeddings,
            device_embeddings=device_embeddings,
            y_true_per_edge=y_true_per_edge,
            y_true_devices=y_true_devices,
            y_global_pred=y_global_pred,  # can be None if you don’t have global predictions
            n_classes=8,
            use_classification=True,
            verbose=True
        )  
        """

        # Store results
        history["device_means"].append(metrics["device_mean"])
        history["device_stds"].append(metrics["device_std"])
        history["edge_means"].append(metrics["edge_mean"])
        history["edge_stds"].append(metrics["edge_std"])
        history["global_accs"].append(metrics["global_acc"])

    return (device_models, edge_models,
            residuals_devices, residuals_edges,
            y_global_pred, device_embeddings, edge_embeddings,
            history)



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
        global_preds = np.argmax(sp_softmax(logits, axis=1), axis=1)
        global_acc = compute_accuracy(y_global, global_preds)

    return device_acc, edge_acc, global_acc


# ============================================================
# Training Loop Example
# ============================================================

if __name__ == "__main__":
    folder_path = "CIC_IoT_DIAD_2024"
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
        X_pretrain, y_pretrain,
        num_devices=config["n_device"],
        n_edges=config["n_edges"],
        alpha=config["alpha"],
        seed=config["random_seed"]
    )

    # 4. Train HPFL model with accuracy tracking
    device_models, edge_models, residuals_devices, residuals_edges, \
        y_global_pred, device_embeddings, edge_embeddings, \
        history = hpfl_train_with_accuracy(
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
    plt.plot(history["device_means"], label="Device-level Mean Acc", marker='o')
    plt.plot(history["edge_means"], label="Edge-level Mean Acc", marker='s')
    plt.plot(history["global_accs"], label="Global Acc", marker='^')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Hierarchical PFL Accuracy Trends")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.show()

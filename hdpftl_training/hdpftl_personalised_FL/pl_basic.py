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
    "early_stopping_rounds_edge": 20,
    "early_stopping_rounds_backward": 10,
    "class_weight": "balanced",
    "lambda_l1": 1.0,
    "lambda_l2": 1.0

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
        device="gpu",
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
    Device-level sequential boosting (classification version) with safe incremental boosting.
    Returns:
        residuals_devices: list of np.ndarray, final residuals per device
        device_models: list of lists, trained models per device
        device_embeddings: list of np.ndarray, one-hot leaf embeddings per device
        y_true_devices: list of np.ndarray, y_finetune per device
    """

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

            # Only pass previous model if features match
            init_model = models_per_device[-1] if models_per_device else None

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

            pred_proba = predict_proba_fixed(model, X_train, num_classes, le=le)

            # Update residuals
            residual -= pred_proba
            residual = np.clip(residual, 0.0, None)

            models_per_device.append(model)

        residuals_devices[idx] = residual
        device_models[idx] = models_per_device

        # Compute leaf embeddings for all trained models
        leaf_indices_list = [get_leaf_indices(mdl, X_train) for mdl in models_per_device]
        if leaf_indices_list:
            leaf_indices_concat = np.hstack(leaf_indices_list)
            leaf_embeddings = np.zeros((n_samples, np.max(leaf_indices_concat) + 1))
            leaf_embeddings[np.arange(n_samples)[:, None], leaf_indices_concat] = 1.0
        else:
            leaf_embeddings = np.zeros((n_samples, 1))
        device_embeddings.append(leaf_embeddings)

    # y_true_devices: just convert y_finetune to list of arrays per device
    y_true_devices = [np.array(y_finetune[d]) for d in range(len(devices_data))]

    return residuals_devices, device_models, device_embeddings, y_true_devices


def edge_layer_boosting(edge_groups, device_embeddings, residuals_devices, le, num_classes,
                        X_finetune=None, y_finetune=None):
    """
    Edge-level sequential boosting with fully safe incremental boosting.

    Handles:
    - Missing classes in pseudo-labels
    - Feature mismatches for init_model
    - Safe validation handling
    - Robust averaging of predictions

    Returns:
        edge_outputs: list of np.ndarray, predictions per edge
        edge_models: list of list, trained models per edge
        residuals_edges: list of np.ndarray, final residuals per edge
        edge_embeddings_list: list of np.ndarray, padded leaf embeddings per edge
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

            unique_classes = np.unique(y_pseudo)
            # Stop if less than 2 classes
            if len(unique_classes) < 2:
                print(f"Edge {edge_idx}, round {t}: only {len(unique_classes)} class(es), skipping boosting.")
                break

            # Only use init_model if feature shapes match and all classes exist
            if models_per_edge:
                prev_model = models_per_edge[-1]
                if prev_model.n_features_in_ == X_edge.shape[1] and len(unique_classes) == num_classes:
                    init_model = prev_model
                else:
                    init_model = None
            else:
                init_model = None

            # Slice validation features to match X_edge if necessary
            X_valid_slice = X_finetune[:, :X_edge.shape[1]] if X_finetune is not None else None

            # Train LightGBM safely
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

            # Update residuals safely
            residual_edge -= pred_proba
            residual_edge = np.clip(residual_edge, 0.0, None)

            models_per_edge.append(model)

        # Store results
        edge_models.append(models_per_edge)
        residuals_edges.append(residual_edge)
        edge_embeddings_list.append(X_edge)

        # Compute edge outputs robustly
        if models_per_edge:
            model_preds = [np.atleast_2d(m.predict_proba(X_edge)) for m in models_per_edge]
            edge_pred_avg = average_predictions(
                model_preds,
                num_samples=X_edge.shape[0],
                num_classes=num_classes
            )
            edge_outputs.append(edge_pred_avg)
        else:
            edge_outputs.append(None)

    return edge_outputs, edge_models, residuals_edges, edge_embeddings_list


def global_layer_bayesian_aggregation(edge_outputs, edge_embeddings, y_true_per_edge,
                                      device_embeddings, residuals_edges=None,
                                      num_classes=2, verbose=True):
    """
    Global (server) Bayesian aggregation using edge embeddings,
    weighting edges by their residual errors, with explicit alpha/beta calculation.
    Returns NumPy arrays only, no PyTorch.

    Args:
        edge_outputs: list of np.ndarray, outputs from edge models
        edge_embeddings: list of np.ndarray, embeddings per edge
        y_true_per_edge: list of np.ndarray, true labels per edge
        device_embeddings: list of np.ndarray, for slicing
        residuals_edges: list of np.ndarray, optional residuals for weighting
        num_classes: int
        verbose: bool

    Returns:
        y_global_pred: np.ndarray, global predictions (softmax)
        global_residuals: np.ndarray, residuals for backward pass
        theta_global: dict with 'alpha' and 'beta'
        edge_sample_slices: list of tuples (start_idx, end_idx) per edge
        device_sample_slices: list of slice per device
    """

    # -----------------------------
    # Step 1: Filter valid edges
    # -----------------------------
    valid_edges = [i for i, (out, emb) in enumerate(zip(edge_outputs, edge_embeddings))
                   if out is not None and emb is not None]
    if not valid_edges:
        raise ValueError("No valid edges found!")

    # -----------------------------
    # Step 2: Collect edge predictions and labels
    # -----------------------------
    edge_preds_list = []
    y_global_list = []
    edge_sample_slices = []
    start_idx = 0
    for i in valid_edges:
        out = edge_outputs[i]
        n_samples = out.shape[0]

        # Pad classes if necessary
        if out.shape[1] < num_classes:
            out = np.pad(out, ((0, 0), (0, num_classes - out.shape[1])), mode='constant')
        edge_preds_list.append(out)

        # True labels
        labels = np.atleast_1d(y_true_per_edge[i])
        if len(labels) < n_samples:
            labels = np.pad(labels, (0, n_samples - len(labels)), mode='edge')
        elif len(labels) > n_samples:
            labels = labels[:n_samples]
        y_global_list.append(labels)

        # Slice for this edge
        edge_sample_slices.append((start_idx, start_idx + n_samples))
        start_idx += n_samples

    # -----------------------------
    # Step 3: Stack predictions and labels
    # -----------------------------
    H_global = np.vstack(edge_preds_list)  # (n_samples_total, num_classes)
    y_global = np.hstack(y_global_list)
    n_samples = H_global.shape[0]

    # -----------------------------
    # Step 4: Compute edge weights
    # -----------------------------
    if residuals_edges is not None:
        weights_list = [1.0 / (np.mean(residuals_edges[i] ** 2, axis=1) + 1e-6)
                        for i in valid_edges]
        edge_weights = np.hstack(weights_list)
    else:
        edge_weights = np.ones(n_samples)
    edge_weights /= np.sum(edge_weights)  # normalize

    if verbose:
        print("Normalized edge weights:", edge_weights)

    # -----------------------------
    # Step 5: Alpha and beta
    # -----------------------------
    alpha = np.sum(H_global * edge_weights[:, None], axis=0)

    # One-hot true labels
    y_onehot = np.zeros((n_samples, num_classes))
    y_onehot[np.arange(n_samples), y_global] = 1

    # Residuals and beta
    global_residuals = y_onehot - H_global
    beta = np.sum(global_residuals * edge_weights[:, None], axis=0)

    # -----------------------------
    # Step 6: Compute softmax global predictions
    # -----------------------------
    logits = H_global + beta
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    y_global_pred = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # -----------------------------
    # Step 7: Generate device sample slices
    # -----------------------------
    device_sample_slices = []
    for edge_idx, edge_devices in enumerate(edge_groups):
        start_idx_edge = edge_sample_slices[edge_idx][0]
        for d in edge_devices:
            n_samples_device = device_embeddings[d].shape[0]
            device_sample_slices.append(slice(start_idx_edge, start_idx_edge + n_samples_device))
            start_idx_edge += n_samples_device

    theta_global = {"alpha": alpha, "beta": beta}

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
    residuals_devices, device_models, device_embeddings, y_true_devices = device_layer_boosting(
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
    # Compute averaged edge predictions safely
    # -----------------------------

    # -----------------------------
    # Gossip / Global Layer
    # -----------------------------
    y_global_pred, global_residuals, theta_global, edge_sample_slices, device_sample_slices = global_layer_bayesian_aggregation(
        edge_outputs=edge_outputs,
        edge_embeddings=edge_embeddings,
        y_true_per_edge=y_true_per_edge,
        device_embeddings=device_embeddings,
        residuals_edges=residuals_edges,
        num_classes=num_classes
    )

    # Compute global accuracy
    # global_acc = compute_accuracy(y_global_true, y_global_pred.argmax(axis=1))

    return (device_models, edge_models, edge_outputs, theta_global, residuals_devices, residuals_edges, \
            y_global_pred, device_embeddings, edge_embeddings, y_global_true, y_true_per_edge, global_residuals,
            edge_sample_slices,
            y_true_devices, device_sample_slices)


def backward_pass(edge_models, device_models, edge_embeddings, device_embeddings,
                  y_true_per_edge, y_true_devices, y_global_pred=None,
                  n_classes=8, use_classification=True, verbose=True):
    """
    Hierarchical residual feedback for HPFL using LightGBM.

    - use_classification=True : fit LGBMClassifier on integer labels
    - use_classification=False: fit LGBMRegressor on residuals (continuous)

    Returns updated edge_models and device_models.
    """

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


# ============================================================
#                  HPFL TRAINING AND EVALUATION
# ============================================================
# - hpfl_train_with_accuracy: hierarchical training loop
#   with device-level and edge-level sequential boosting.
# - evaluate_final_accuracy: compute final accuracy metrics at
#   device, edge, and global (gossip-summary) levels.
# ============================================================

def hpfl_train_with_accuracy(devices_data, edge_groups, le, num_classes,
                             X_finetune, y_finetune, verbose=True, device='gpu'):
    """
    HPFL training loop with forward/backward passes and accuracy tracking.

    Args:
        devices_data: list of tuples (X_train, _, y_train, _)
        edge_groups: list of lists, devices per edge
        le: LabelEncoder
        num_classes: int
        X_finetune, y_finetune: optional fine-tuning data
        verbose: bool
        device: 'cpu' or 'cuda'

    Returns:
        device_models, edge_models, residuals_devices, residuals_edges,
        y_global_pred, device_embeddings, edge_embeddings,
        device_accs, edge_accs, global_accs
    """

    device_accs, edge_accs, global_accs = [], [], []
    residuals_devices = [None] * len(devices_data)
    device_models = [None] * len(devices_data)
    residuals_edges = [None] * len(edge_groups)
    edge_models = [None] * len(edge_groups)

    device_embeddings = [None] * len(devices_data)
    edge_embeddings = [None] * len(edge_groups)

    num_epochs = config["epoch"]

    for epoch in range(num_epochs):
        if verbose:
            print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")

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

        # Move residuals to device for backward pass
        global_residuals_torch = torch.tensor(global_residuals, dtype=torch.float32, device=device)

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
        # -----------------------------
        # 4. Compute Device-Level Accuracy
        # -----------------------------
        device_acc_epoch = []
        for i, (X_test, _, y_test, _) in enumerate(devices_data):
            X_test_np = safe_array(X_test)
            device_preds = np.zeros((X_test_np.shape[0], num_classes))

            if device_models[i] is not None:
                for mdl in device_models[i]:
                    device_preds += predict_proba_fixed(mdl, X_test_np, num_classes, le=le)
                device_preds /= max(len(device_models[i]), 1)

            device_acc_epoch.append(compute_accuracy(y_test, device_preds.argmax(axis=1)))
        device_accs.append(np.mean(device_acc_epoch))

        # -----------------------------
        # 5. Compute Edge-Level Accuracy
        # -----------------------------
        edge_acc_epoch = []
        for edge_idx, devices_in_edge in enumerate(edge_groups):
            if not devices_in_edge or edge_models[edge_idx] is None:
                continue

            X_edge = np.vstack([safe_array(devices_data[d][0]) for d in devices_in_edge])
            y_edge = np.hstack([devices_data[d][2] for d in devices_in_edge])

            edge_preds = np.zeros((X_edge.shape[0], num_classes))
            for mdl in edge_models[edge_idx]:
                edge_preds += predict_proba_fixed(mdl, X_edge, num_classes, le=le)
            edge_preds /= max(len(edge_models[edge_idx]), 1)

            edge_acc_epoch.append(compute_accuracy(y_edge, edge_preds.argmax(axis=1)))
        edge_accs.append(np.mean(edge_acc_epoch) if edge_acc_epoch else 0.0)

        # -----------------------------
        # 6. Compute Global Accuracy
        # -----------------------------
        global_acc = compute_accuracy(y_global_true, y_global_pred.argmax(axis=1))
        global_accs.append(global_acc)

        if verbose:
            print(f"[Epoch {epoch + 1}] Device Acc: {np.mean(device_acc_epoch):.4f}, "
                  f"Edge Acc: {np.mean(edge_acc_epoch) if edge_acc_epoch else 0.0:.4f}, "
                  f"Global Acc: {global_acc:.4f}")

    return device_models, edge_models, residuals_devices, residuals_edges, \
        y_global_pred, device_embeddings, edge_embeddings, \
        device_accs, edge_accs, global_accs


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
        X_pretrain, y_pretrain,  # use fine-tune data
        num_devices=config["n_device"],
        n_edges=config["n_edges"],
        alpha=config["alpha"],
        seed=config["random_seed"]
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

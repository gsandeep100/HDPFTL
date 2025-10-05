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
from matplotlib.animation import FuncAnimation
from scipy.special import softmax as sp_softmax
from sklearn.metrics import accuracy_score
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

def train_lightgbm(
        X_train,
        y_train,
        X_valid=None,
        y_valid=None,
        early_stopping_rounds = config["num_iterations_edge"],
        learning_rate = config["learning_rate_device"],
        init_model = None,
        verbose=-1
):
    """
    Train a LightGBM classifier safely with optional validation and continuation from a previous model.

    Args:
        X_train, y_train: training data
        X_valid, y_valid: optional validation data
        config: dict of hyperparameters
        learning_rate: float
        num_iterations: int
        early_stopping_rounds: int
        init_model: existing LGBMClassifier to continue training
        verbose: int, verbosity level (-1 to suppress)

    Returns:
        Trained LGBMClassifier
    """

    # Safe conversion
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_valid = np.array(X_valid) if X_valid is not None else None
    y_valid = np.array(y_valid) if y_valid is not None else None

    # Handle one-hot labels
    if y_train.ndim > 1 and y_train.shape[1] > 1:
        y_train = np.argmax(y_train, axis=1)
    if y_valid is not None and y_valid.ndim > 1 and y_valid.shape[1] > 1:
        y_valid = np.argmax(y_valid, axis=1)

    # Determine number of classes
    num_classes = len(np.unique(y_train))
    objective = "binary" if num_classes == 2 else "multiclass"

    # Build new model only if init_model is not provided
    if init_model is None:
        model_params = {
            "boosting_type": config.get("boosting", "gbdt"),
            "objective": objective,
            "num_class": num_classes if num_classes > 2 else None,
            "n_estimators": config.get("n_estimators", 5000),
            "learning_rate": learning_rate,
            "num_iterations": config.get("num_iterations_device", 100),
            "num_leaves": config.get("num_leaves", 31),
            "max_depth": config.get("max_depth", -1),
            "min_child_samples": config.get("min_child_samples", 20),
            "class_weight": config.get("class_weight", None),
            "lambda_l1": config.get("lambda_l1", 0.0) * 2,
            "lambda_l2": config.get("lambda_l2", 0.0) * 2,
            "feature_fraction": config.get("feature_fraction", 1.0),
            "random_state": config.get("random_seed", 42),
            "device": config.get("device", "cpu"),
            "verbose": verbose
        }
        model = LGBMClassifier(**model_params)
    else:
        model = init_model  # continue training previous model

    # Prepare fit kwargs
    fit_kwargs = {}
    if X_valid is not None and y_valid is not None and early_stopping_rounds:
        mask = np.isin(y_valid, np.unique(y_train))
        if mask.sum() < 10:
            mask = np.ones_like(y_valid, dtype=bool)
        X_valid_filtered = X_valid[mask]
        y_valid_filtered = y_valid[mask]

        fit_kwargs.update({
            "eval_set": [(X_train, y_train), (X_valid_filtered, y_valid_filtered)],
            "eval_metric": "multi_logloss" if num_classes > 2 else "logloss",
            "eval_names": ["train", "valid"],
            "callbacks": [
                early_stopping(early_stopping_rounds),
                log_evaluation(10)
            ]
        })

    # Continue from previous model only if classes match
    if init_model is not None:
        train_classes = np.unique(y_train)
        model_classes = init_model.classes_

        # Check if training labels match the init_model classes
        if len(train_classes) == len(model_classes) and np.all(np.isin(train_classes, model_classes)):
            fit_kwargs["init_model"] = init_model
        else:
            print(
                "Skipping init_model because training labels do not match previous model classes. "
                f"Train classes: {train_classes}, Model classes: {model_classes}"
            )
            init_model = None  # safely skip continuation

    # Train
    print("Training shapes:", X_train.shape, y_train.shape)
    print("Unique labels:", np.unique(y_train))
    model.fit(X_train, y_train, **fit_kwargs)

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
    # --- Ensure X is 2D numpy array ---
    X_np = np.atleast_2d(np.array(X))

    # --- Feature alignment ---
    n_features_model = getattr(model, "n_features_in_", X_np.shape[1])
    if X_np.shape[1] > n_features_model:
        X_np = X_np[:, :n_features_model]  # slice down
    elif X_np.shape[1] < n_features_model:
        pad_width = n_features_model - X_np.shape[1]
        X_np = np.pad(X_np, ((0, 0), (0, pad_width)), mode="constant")

    # --- Predict ---
    pred = model.predict_proba(X_np)
    pred = np.atleast_2d(pred)

    # --- Class alignment ---
    full = np.zeros((pred.shape[0], num_classes), dtype=float)
    model_classes = getattr(model, "classes_", np.arange(pred.shape[1]))

    if le is not None:
        # Map model classes to label encoder positions
        class_pos = {cls: i for i, cls in enumerate(le.classes_)}
        for i, cls in enumerate(model_classes):
            pos = class_pos.get(cls)
            if pos is not None and pos < num_classes:
                full[:, pos] = pred[:, i]
    else:
        # Map integer class labels directly
        for i, cls in enumerate(model_classes):
            if 0 <= cls < num_classes:
                full[:, int(cls)] = pred[:, i]

    # --- Normalize rows ---
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
        padded_blocks = []
        for block in global_pred_blocks:
            if block.shape[1] < num_classes:
                # Pad missing columns (new classes) with zeros
                pad_width = num_classes - block.shape[1]
                block = np.hstack([block, np.zeros((block.shape[0], pad_width))])
            padded_blocks.append(block)
        global_pred_matrix = np.vstack(padded_blocks)
    else:
        global_pred_matrix = np.empty((0, num_classes))

    return edge_outputs, edge_models, residuals_edges, edge_embeddings_list, global_pred_matrix, edge_sample_slices



def global_layer_bayesian_aggregation(edge_outputs, edge_embeddings, y_true_per_edge,
                                      residuals_edges=None, num_classes=2, verbose=True):
    """
    Global Bayesian aggregation using edge outputs, weighting edges by residuals.

    Args:
        edge_outputs: list of np.ndarray
            Edge-level boosted outputs (probability matrices per edge)
        edge_embeddings: list of np.ndarray
            Embeddings per edge (used for alignment)
        y_true_per_edge: list of np.ndarray
            True labels per edge
        residuals_edges: list of np.ndarray, optional
            Residuals from edge boosting (used for reliability weighting)
        num_classes: int
            Number of classes in classification task
        verbose: bool
            Whether to print debug info

    Returns:
        y_global_pred: np.ndarray
            Global softmax predictions (n_samples_total × num_classes)
        global_residuals: np.ndarray
            Residuals after global aggregation (for analysis/backprop)
        theta_global: dict
            Parameters of Bayesian fusion: {"alpha": α, "beta": β}
    """

    print("""
    ************************************************************
    *                                                          *
    *                  STARTING GLOBAL LAYER                   *
    *  Edges -> Bayesian fusion (weighted) -> Global ensemble  *
    *                                                          *
    ************************************************************
    """)

    # -----------------------------
    # Step 1: Filter valid edges
    # -----------------------------
    valid_edges = [i for i, (out, emb) in enumerate(zip(edge_outputs, edge_embeddings))
                   if out is not None and emb is not None]
    if not valid_edges:
        raise ValueError("No valid edges found!")

    # -----------------------------
    # Step 2: Stack predictions and labels across edges
    # -----------------------------
    edge_preds_list = []
    y_global_list = []
    for i in valid_edges:
        out = edge_outputs[i]
        n_samples = out.shape[0]

        # Pad missing classes if needed
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

    H_global = np.vstack(edge_preds_list)   # (total_samples, num_classes)
    y_global = np.hstack(y_global_list)     # Flattened true labels
    n_samples_total = H_global.shape[0]

    # -----------------------------
    # Step 3: Compute edge weights (Bayesian reliability)
    # -----------------------------
    if residuals_edges is not None:
        weights_list = [1.0 / (np.mean(residuals_edges[i] ** 2, axis=1) + 1e-6)
                        for i in valid_edges]
        edge_weights = np.hstack(weights_list)
    else:
        edge_weights = np.ones(n_samples_total)

    edge_weights /= np.sum(edge_weights)
    if verbose:
        print(f"Normalized edge weights sum to {np.sum(edge_weights):.6f}")

    # -----------------------------
    # Step 4: Bayesian parameters α and β
    # -----------------------------
    alpha = np.sum(H_global * edge_weights[:, None], axis=0)  # weighted average of predictions

    y_onehot = np.zeros((n_samples_total, num_classes))
    y_onehot[np.arange(n_samples_total), y_global] = 1
    global_residuals = y_onehot - H_global
    beta = np.sum(global_residuals * edge_weights[:, None], axis=0)  # correction term

    # -----------------------------
    # Step 5: Compute global softmax predictions
    # -----------------------------
    logits = H_global + beta
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    y_global_pred = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    theta_global = {"alpha": alpha, "beta": beta}

    if verbose:
        print(f"Global alpha: {alpha}")
        print(f"Global beta: {beta}")
        print(f"Total samples: {n_samples_total}")

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
    y_global_pred, global_residuals, theta_global = global_layer_bayesian_aggregation(
        edge_outputs=edge_outputs,
        edge_embeddings=edge_embeddings_list,
        y_true_per_edge=y_true_per_edge,
        residuals_edges=residuals_edges,
        num_classes=num_classes
    )

    y_global_true = np.hstack(y_true_per_edge)  # <- ensures labels match stacked preds

    # Accuracy
    y_pred_labels = y_global_pred.argmax(axis=1)
    global_acc = compute_accuracy(y_global_true, y_pred_labels)

    print(f"Global Accuracy: {global_acc:.4f}")

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
        y_true_devices,
        global_acc
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


def safe_labels(y):
    """Convert labels to integer NumPy array for LightGBM."""
    if y is None:
        return None
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = y.to_numpy()
    y = np.asarray(y)
    # Convert one-hot to class indices if needed
    if y.ndim > 1 and y.shape[1] > 1:
        y = np.argmax(y, axis=1)
    return y.astype(np.int32)


def safe_fit(X, y, *, model=None, **fit_kwargs):
    X_safe = safe_array(X)
    y_safe = safe_labels(y)

    # Determine number of classes
    n_classes = len(np.unique(y_safe))
    objective = 'binary' if n_classes == 2 else 'multiclass'

    # Handle verbose safely
    verbose = fit_kwargs.pop("verbose", None)

    # Use given model or create a new one
    if model is None:
        model = LGBMClassifier(objective=objective)
        if verbose is not None:
            model.set_params(verbose=verbose)

    # Set num_class for multiclass if needed
    if n_classes > 2:
        model.set_params(num_class=n_classes)

    # Fit the model with valid kwargs only
    model.fit(X_safe, y_safe, **fit_kwargs)
    return model


"""
def safe_edge_output(edge_outputs, num_classes):
    \"""
    Ensures each edge output is 2D with shape (n_samples, num_classes).
    If outputs are missing columns, pad with zeros.
    If outputs are 1D, convert to one-hot.
    \"""
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
"""

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


def compute_multilevel_accuracy(
    device_models,
    edge_models,
    edge_groups,
    X_test,
    y_test,
    y_global_pred_test,
    num_classes=2,
    verbose=True
):
    """
    Compute device-level, edge-level, and global accuracy on test data,
    handling scalar labels and class/feature alignment.

    Args:
        device_models: list of lists of trained models per device
        edge_models: list of lists of trained models per edge
        edge_groups: list of lists, devices per edge
        X_test: list of np.ndarray per device
        y_test: list/array of true labels per device (can be scalar)
        y_global_pred_test: np.ndarray from forward_pass (global predictions)
        num_classes: int
        verbose: bool

    Returns:
        metrics: dict containing device, edge, and global accuracies
    """

    # -----------------------------
    # 1. Device-level predictions
    # -----------------------------
    device_preds = []
    device_accs = []

    for dev_idx, X_dev in enumerate(X_test):
        n_samples = X_dev.shape[0]
        pred_accum = np.zeros((n_samples, num_classes), dtype=float)
        trained_models = device_models[dev_idx] if dev_idx < len(device_models) else []

        # Predict
        for mdl in trained_models:
            if mdl is not None:
                pred_accum += predict_proba_fixed(mdl, X_dev, num_classes)

        # Average if multiple models
        if len(trained_models) > 0:
            pred_accum /= len([m for m in trained_models if m is not None])

        device_preds.append(pred_accum)

        # --- Fix scalar labels ---
        y_true_dev = np.atleast_1d(y_test[dev_idx])
        if y_true_dev.shape[0] != n_samples:
            if y_true_dev.size == 1:
                y_true_dev = np.full(n_samples, y_true_dev[0])
            else:
                raise ValueError(f"Device {dev_idx}: y_true length {y_true_dev.shape[0]} "
                                 f"does not match X_dev samples {n_samples}")

        device_labels = pred_accum.argmax(axis=1)
        device_accs.append(accuracy_score(y_true_dev, device_labels))

    # -----------------------------
    # 2. Edge-level predictions
    # -----------------------------
    edge_preds = []
    edge_accs = []

    for edge_idx, edge_devices in enumerate(edge_groups):
        # Combine device test data for this edge
        X_edge = np.vstack([X_test[d] for d in edge_devices])
        pred_accum = np.zeros((X_edge.shape[0], num_classes), dtype=float)
        trained_models = edge_models[edge_idx] or []

        # Predict
        for mdl in trained_models:
            if mdl is not None:
                pred_accum += predict_proba_fixed(mdl, X_edge, num_classes)

        # Average
        if len(trained_models) > 0:
            pred_accum /= len([m for m in trained_models if m is not None])

        edge_preds.append(pred_accum)

        # --- Fix scalar/mismatched labels per device ---
        y_edge_true_list = []
        for d in edge_devices:
            y_d = np.atleast_1d(y_test[d])
            if y_d.shape[0] == 1:
                y_d = np.full(X_test[d].shape[0], y_d[0])
            elif y_d.shape[0] != X_test[d].shape[0]:
                print(f"[Warning] Edge {edge_idx}, Device {d}: y_true length {y_d.shape[0]} "
                      f"!= X_test samples {X_test[d].shape[0]}; trimming to match")
                y_d = y_d[:X_test[d].shape[0]]  # trim if longer
            y_edge_true_list.append(y_d)

        y_edge_true = np.hstack(y_edge_true_list)

        # Compute accuracy
        edge_labels = pred_accum.argmax(axis=1)
        edge_accs.append(accuracy_score(y_edge_true, edge_labels))

    # -----------------------------
    # 3. Global predictions
    # -----------------------------
    global_labels = y_global_pred_test.argmax(axis=1)
    y_global_true = np.hstack([
        np.full(X_test[d].shape[0], y_test[d]) if np.atleast_1d(y_test[d]).size == 1
        else np.atleast_1d(y_test[d])
        for d in range(len(X_test))
    ])
    global_acc = accuracy_score(y_global_true, global_labels)

    # -----------------------------
    # 4. Device / Edge vs Global
    # -----------------------------
    device_vs_global = []
    start_idx = 0
    for dev_idx, X_dev in enumerate(X_test):
        n_samples = X_dev.shape[0]
        slice_idx = slice(start_idx, start_idx + n_samples)
        if slice_idx.stop <= len(global_labels):
            device_vs_global.append(
                accuracy_score(global_labels[slice_idx], device_preds[dev_idx].argmax(axis=1))
            )
        else:
            print(f"[Warning] Device {dev_idx}: slice exceeds global size")
        start_idx += n_samples

    edge_vs_global = []
    start_idx = 0
    for edge_idx, edge_devices in enumerate(edge_groups):
        n_samples_edge = sum(X_test[d].shape[0] for d in edge_devices)
        slice_idx = slice(start_idx, start_idx + n_samples_edge)
        if slice_idx.stop <= len(global_labels):
            edge_vs_global.append(
                accuracy_score(global_labels[slice_idx], edge_preds[edge_idx].argmax(axis=1))
            )
        else:
            print(f"[Warning] Edge {edge_idx}: slice exceeds global size")
        start_idx += n_samples_edge

    # -----------------------------
    # 5. Return metrics
    # -----------------------------
    metrics = {
        "device_accs": device_accs,
        "edge_accs": edge_accs,
        "global_acc": global_acc,
        "device_vs_global": device_vs_global,
        "edge_vs_global": edge_vs_global
    }

    if verbose:
        print(f"Device mean acc: {np.mean(device_accs):.4f}, "
              f"Edge mean acc: {np.mean(edge_accs):.4f}, "
              f"Global acc: {global_acc:.4f}")

    return metrics


# ============================================================
#                  HPFL TRAINING AND EVALUATION
# ============================================================
# - hpfl_train_with_accuracy: hierarchical training loop
#   with device-level and edge-level sequential boosting.
# - evaluate_final_accuracy: compute final accuracy metrics at
#   device, edge, and global (gossip-summary) levels.
# ============================================================

def hpfl_train_with_accuracy(devices_data, edge_groups, le, num_classes,
                             X_finetune, y_finetune, X_test, y_test, verbose=True):
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
    y_true_per_epoch = []

    # --- history tracking ---
    history = {
        "device_means": [],  # mean accuracy across all devices per epoch
        "device_stds": [],  # std dev of device accuracies per epoch
        "edge_means": [],  # mean accuracy across all edges per epoch
        "edge_stds": [],  # std dev of edge accuracies per epoch
        "global_accs": [],  # global model accuracy per epoch
        "device_vs_global": [],  # mean device accuracy compared with global per epoch
        "edge_vs_global": [],  # mean edge accuracy compared with global per epoch
        "y_true_per_epoch": []  # optional: true labels of test set per epoch
    }

    for epoch in range(num_epochs):
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
            y_true_per_edge, global_residuals, \
            y_true_devices,global_acc = forward_pass(
                devices_data, edge_groups, le, num_classes,
                X_finetune, y_finetune,
                residuals_devices=residuals_devices,
                device_models=device_models
        )
        y_true_per_epoch.append(y_global_true)

        # -----------------------------
        # 2. Compute Multi-Level Accuracy
        # -----------------------------
        metrics_test = compute_multilevel_accuracy(
            device_models=device_models,
            edge_models=edge_models,
            edge_groups=edge_groups,
            y_test=y_test,
            X_test=X_test,
            y_global_pred_test=y_global_pred,  # from forward pass
            num_classes=num_classes,
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


        # Update history for plotting
        history["device_means"].append(np.mean(metrics_test["device_accs"]))
        history["device_stds"].append(np.std(metrics_test["device_accs"]))
        history["edge_means"].append(np.mean(metrics_test["edge_accs"]))
        history["edge_stds"].append(np.std(metrics_test["edge_accs"]))
        history["global_accs"].append(metrics_test["global_acc"])
        history["device_vs_global"].append(np.mean(metrics_test["device_vs_global"]))
        history["edge_vs_global"].append(np.mean(metrics_test["edge_vs_global"]))
        history["y_true_per_epoch"].append(y_test)  # optional reference

    return (device_models,
        edge_models,
        residuals_devices,
        residuals_edges,
        y_global_pred,
        y_true_per_epoch,
        device_embeddings,
        edge_embeddings,
        history)


"""
def evaluate_final_accuracy(devices_data, device_models, edge_groups, le, num_classes, gossip_summary=None):
    \"""
    Evaluate final accuracy at:
    - Device-level
    - Edge-level
    - Global/Gossip-level (if gossip_summary provided)
    \"""
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
"""

# Create dated folder inside the main save_dir
def get_dated_save_dir(base_dir="hdpftl_plot_outputs"):
    today_str = datetime.now().strftime("%Y-%m-%d")
    dated_dir = os.path.join(base_dir, today_str)
    os.makedirs(dated_dir, exist_ok=True)
    return dated_dir


def plot_hpfl_all(history, save_root_dir="hdpftl_plot_outputs"):
    """
    Generate all Hierarchical PFL plots with layer-wise contributions and comparisons
    against global accuracy.

    Args:
        history (dict): Output from compute_multilevel_accuracy for each epoch, e.g.:
            history[epoch_idx] = {
                'device_accs': [...],
                'edge_accs': [...],
                'global_acc': ...,
                'device_vs_global': [...],
                'edge_vs_global': [...]
            }
        save_root_dir (str): Base directory to save plots
    """

    # -----------------------------
    # Create dated folder automatically
    # -----------------------------
    today_str = datetime.now().strftime("%Y-%m-%d")
    save_dir = os.path.join(save_root_dir, f"{today_str}")
    os.makedirs(save_dir, exist_ok=True)

    num_epochs = len(history)
    layers = ["Device", "Edge", "Global"]
    colors = ['skyblue', 'orange', 'green']

    # -----------------------------
    # 1. Per-epoch stacked bar charts & comparisons
    # -----------------------------
    for epoch_idx in range(num_epochs):
        metrics = history[epoch_idx]

        device_accs = np.array(metrics['device_accs'])
        edge_accs = np.array(metrics['edge_accs'])
        global_acc = metrics['global_acc']
        device_vs_global = np.array(metrics['device_vs_global'])
        edge_vs_global = np.array(metrics['edge_vs_global'])

        # -----------------------------
        # Stacked contribution vs global
        # -----------------------------
        contributions = [np.mean(device_accs), np.mean(edge_accs) - np.mean(device_accs), global_acc - np.mean(edge_accs)]
        contributions = np.clip(contributions, 0, None)

        plt.figure(figsize=(6, 4))
        bars = plt.bar(layers, contributions, color=colors)
        plt.ylim(0, 1)
        plt.ylabel("Contribution to Final Accuracy")
        plt.title(f"Epoch {epoch_idx+1} Layer Contributions")
        for bar, val in zip(bars, contributions):
            plt.text(bar.get_x() + bar.get_width()/2, float(val) + 0.02, f"{float(val):.3f}", ha='center', va='bottom')
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"epoch_contribution_{epoch_idx+1}.png"))
        plt.close()

        # -----------------------------
        # Device vs Global and Edge vs Global plots
        # -----------------------------
        plt.figure(figsize=(6, 4))
        plt.bar([f"Device vs Global", f"Edge vs Global"],
                [np.mean(device_vs_global), np.mean(edge_vs_global)],
                color=['skyblue', 'orange'])
        plt.ylim(0, 1)
        plt.ylabel("Accuracy compared to Global")
        plt.title(f"Epoch {epoch_idx+1} Layer vs Global Accuracy")
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"epoch_vs_global_{epoch_idx+1}.png"))
        plt.close()

    # -----------------------------
    # 2. Accuracy trends across epochs
    # -----------------------------
    mean_device_accs = [np.mean(history[e]['device_accs']) for e in range(num_epochs)]
    mean_edge_accs = [np.mean(history[e]['edge_accs']) for e in range(num_epochs)]
    global_accs = [history[e]['global_acc'] for e in range(num_epochs)]
    mean_device_vs_global = [np.mean(history[e]['device_vs_global']) for e in range(num_epochs)]
    mean_edge_vs_global = [np.mean(history[e]['edge_vs_global']) for e in range(num_epochs)]

    # Layer Accuracy Trends
    plt.figure(figsize=(10, 6))
    plt.plot(mean_device_accs, label="Device Accuracy", marker='o', color='skyblue')
    plt.plot(mean_edge_accs, label="Edge Accuracy", marker='s', color='orange')
    plt.plot(global_accs, label="Global Accuracy", marker='^', color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Hierarchical PFL Accuracy Trends")
    plt.ylim(0, 1)
    plt.xticks(np.arange(num_epochs), labels=[f"Epoch {i+1}" for i in range(num_epochs)])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "overall_accuracy_trends.png"))
    plt.show()
    plt.close()

    # Device/Edge vs Global Trends
    plt.figure(figsize=(10, 6))
    plt.plot(mean_device_vs_global, label="Device vs Global", marker='o', color='skyblue')
    plt.plot(mean_edge_vs_global, label="Edge vs Global", marker='s', color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy compared to Global")
    plt.title("Device/Edge Accuracy vs Global Across Epochs")
    plt.ylim(0, 1)
    plt.xticks(np.arange(num_epochs), labels=[f"Epoch {i+1}" for i in range(num_epochs)])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "device_edge_vs_global_trends.png"))
    plt.show()
    plt.close()

    print(f"All plots saved to folder: {save_dir}")


# ============================================================
#                     Main
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
        X_pretrain, y_pretrain,
        num_devices=config["n_device"],
        n_edges=config["n_edges"],
        alpha=config["alpha"],
        seed=config["random_seed"]
    )

    # 4. Train HPFL model with accuracy tracking
    device_models, edge_models, residuals_devices, residuals_edges, \
        y_global_pred, y_true_per_epoch, device_embeddings, edge_embeddings, \
        history = hpfl_train_with_accuracy(
        devices_data=devices_data,
        edge_groups=edge_groups,
        le=le,
        num_classes=num_classes,
        X_finetune=X_finetune,
        y_finetune=y_finetune,
        X_test = X_test,
        y_test = y_test,
        verbose=True
    )

    plot_hpfl_all(history)


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
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymc import logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

from lightgbm import early_stopping, LGBMClassifier, LGBMRegressor, log_evaluation
from sklearn.metrics import accuracy_score, log_loss
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
    "epoch": 3,
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
    "device": "cpu",
    "eps_threshold": 0.001,

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
            "max_cores": 2,
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

def predict_proba_fixed(model, X, num_classes):
    """
    Universal safe prediction wrapper for LightGBM / sklearn models.
    Handles:
      - Binary & multiclass
      - Degenerate one-hot predictions
      - Missing/mismatched classes
      - Prevents NaN in logloss
    """
    X_np = np.atleast_2d(np.asarray(X, dtype=float))
    n_features_model = getattr(model, "n_features_in_", X_np.shape[1])

    # Align input features with model
    if X_np.shape[1] > n_features_model:
        X_np = X_np[:, :n_features_model]
    elif X_np.shape[1] < n_features_model:
        X_np = np.pad(X_np, ((0, 0), (0, n_features_model - X_np.shape[1])), mode="constant")

    # Feature names
    columns = getattr(model, "feature_name_", [f"f{i}" for i in range(X_np.shape[1])])
    X_df = pd.DataFrame(X_np, columns=columns)

    # Predict probabilities
    if isinstance(model, lgb.Booster):
        pred = model.predict(X_np, raw_score=False)
    elif hasattr(model, "predict_proba"):
        pred = model.predict_proba(X_df)
    else:
        pred = model.predict(X_df, raw_score=False)

    pred = np.atleast_2d(pred)

    # Flatten binary predictions
    if pred.shape[1] == 1:
        pred = np.hstack([1 - pred, pred])

    # Clip to avoid extreme values
    eps = 1e-6
    pred = np.clip(pred, eps, 1 - eps)

    # Align with total number of classes
    full = np.zeros((pred.shape[0], num_classes))
    model_classes = np.asarray(getattr(model, "classes_", np.arange(pred.shape[1]))).astype(int)

    for i, cls in enumerate(model_classes):
        if 0 <= cls < num_classes:
            full[:, cls] = pred[:, i]

    # Normalize rows
    row_sums = full.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    full /= row_sums

    return full

def get_leaf_indices(model, X):
    """
    Extract leaf indices for all samples in all trees of a trained LightGBM model.

    Args:
        model: LightGBM model (Booster or LGBMClassifier/LGBMRegressor)
        X: np.ndarray or pandas DataFrame, input samples

    Returns:
        np.ndarray: shape (n_samples, n_trees)
            Leaf index of each sample in each tree
    """
    X_np = np.atleast_2d(np.asarray(X, dtype=float))

    # Align features to model
    n_features_model = getattr(model, "n_features_in_", X_np.shape[1])
    if X_np.shape[1] > n_features_model:
        X_np = X_np[:, :n_features_model]
    elif X_np.shape[1] < n_features_model:
        X_np = np.pad(X_np, ((0, 0), (0, n_features_model - X_np.shape[1])), mode="constant")

    columns = getattr(model, "feature_name_", [f"f{i}" for i in range(X_np.shape[1])])
    X_df = pd.DataFrame(X_np, columns=columns)

    # Predict leaf indices
    if isinstance(model, lgb.Booster):
        leaf_indices = model.predict(X_np, pred_leaf=True)
    elif hasattr(model, "predict"):
        leaf_indices = model.predict(X_df, pred_leaf=True)
    else:
        raise ValueError("Model does not support leaf index prediction.")

    return np.asarray(leaf_indices, dtype=np.int32)

# ============================================================
# Device Layer Boosting with missing-class safe probabilities
# ============================================================

def device_layer_boosting(d_data, d_residuals, d_models, le, n_classes, use_true_labels=False):
    """
    Device-level sequential boosting (classification version) with safe incremental boosting.
    Optionally uses true labels for residual updates instead of pseudo-labels.
    Includes final true-label residual correction to align residuals with real accuracy.

    Args:
        d_data: list of tuples per device (X_train, _, y_train, _, _, _, X_device_finetune, y_device_finetune)
        d_residuals: list of np.ndarray, previous residuals per device (or None)
        d_models: list of lists, previously trained models per device (or empty)
        le: LabelEncoder for y labels
        n_classes: number of classes
        use_true_labels: bool, if True, use true labels instead of pseudo-labels

    Returns:
        d_residuals: final residuals per device
        d_models: trained models per device
        device_embeddings: leaf embeddings per device
        device_weights: per-device weights based on validation
    """
    print("\n" + "*"*60)
    print("*" + " " * 28 + "STARTING DEVICE LAYER" + " " * 28 + "*")
    print("*" * 60 + "\n")

    device_embeddings = []
    device_val_scores = []
    eps_residual = 1e-6  # small epsilon to prevent degenerate residuals

    for idx, dev_tuple in enumerate(d_data):
        X_train, _, y_train, _, _, _, X_device_finetune, y_device_finetune = dev_tuple
        n_samples = X_train.shape[0]
        prev_y_pseudo = None

        # ----------------------------
        # Initialize residuals per device
        # ----------------------------
        if d_residuals[idx] is None:
            residual = np.zeros((n_samples, n_classes), dtype=float)
            y_encoded = le.transform(y_train)
            residual[np.arange(n_samples), y_encoded] = 1.0
        else:
            residual = d_residuals[idx].copy()

        models_per_device = d_models[idx] if d_models[idx] else []

        # ----------------------------[44144
        # Sequential boosting
        # ----------------------------
        for t in range(config["device_boosting_rounds"]):
            if use_true_labels:
                y_pseudo = le.transform(y_train)  # true labels mode
            else:
                sample_sign = np.sign(residual.sum(axis=1))  # +1 if net positive, -1 if net negative

                y_pseudo = np.where(
                    sample_sign >= 0,
                    np.argmax(residual, axis=1),
                    np.argmin(residual, axis=1)
                )

            if len(np.unique(y_pseudo)) < 2:
                print(f"Device {idx}, round {t}: only single class left, skipping further boosting.")
                break

            if np.sum(np.abs(residual)) < config["eps_threshold"]:
                print(f"Device {idx}, round {t}: residuals below threshold, stopping.")
                break

            if prev_y_pseudo is not None:
                changes = np.mean(prev_y_pseudo != y_pseudo)
                if changes < 0.01:
                    print(f"Device {idx}, round {t}: labels stabilized, stopping.")
                    break
            prev_y_pseudo = y_pseudo.copy()

            init_model = models_per_device[-1] if models_per_device else None

            model = train_lightgbm(
                X_train, y_pseudo,
                X_valid=X_device_finetune[:, :X_train.shape[1]] if y_device_finetune is not None else None,
                y_valid=y_device_finetune,
                early_stopping_rounds=config["early_stopping_rounds_device"],
                learning_rate=config["learning_rate_device"],
                init_model=init_model
            )

            pred_proba = predict_proba_fixed(model, X_train, n_classes)
            residual -= pred_proba
            residual = np.clip(residual, -1 + eps_residual, 1 - eps_residual)

            if models_per_device:
                models_per_device[-1] = model  # update last one
            else:
                models_per_device.append(model)

        # ----------------------------
        # ✅ Final residual correction using true labels
        # ----------------------------
        if y_train is not None and len(models_per_device) > 0:
            y_true_enc = le.transform(y_train)
            y_onehot = np.zeros_like(residual)
            y_onehot[np.arange(len(y_true_enc)), y_true_enc] = 1.0

            # Predictions from final model
            y_pred_proba_final = predict_proba_fixed(models_per_device[-1], X_train, n_classes)

            alpha = 0.5  # correction strength (0.0 = off, 1.0 = full overwrite)
            res_before = np.linalg.norm(residual)
            residual = (1 - alpha) * residual + alpha * (y_onehot - y_pred_proba_final)
            residual = np.clip(residual, -1 + eps_residual, 1 - eps_residual)
            res_after = np.linalg.norm(residual)

            print(f"Device {idx}: residual norm before={res_before:.4f}, after true-label correction={res_after:.4f}")

        # Save residuals & models
        d_residuals[idx] = residual
        d_models[idx] = models_per_device

        # ----------------------------
        # Device embeddings
        # ----------------------------
        leaf_indices_list = [get_leaf_indices(mdl, X_train) for mdl in models_per_device]
        if leaf_indices_list:
            leaf_indices_concat = np.hstack(leaf_indices_list)
            leaf_embeddings = np.zeros((n_samples, np.max(leaf_indices_concat) + 1))
            leaf_embeddings[np.arange(n_samples)[:, None], leaf_indices_concat] = 1.0
        else:
            leaf_embeddings = np.zeros((n_samples, 1))
        device_embeddings.append(leaf_embeddings)

        # ----------------------------
        # Validation-based per-model weights
        # ----------------------------
        if X_device_finetune is not None and y_device_finetune is not None and models_per_device:
            scores_per_model = []
            y_val_encoded = le.transform(np.atleast_1d(y_device_finetune))
            for mdl in models_per_device:
                X_val = X_device_finetune
                if X_val is None:
                    y_pred_proba = np.ones((y_val_encoded.shape[0], n_classes)) / n_classes
                else:
                    X_val_slice = X_val[:, :X_train.shape[1]] if X_val.ndim == 2 and X_val.shape[1] > X_train.shape[1] else X_val
                    y_pred_proba = predict_proba_fixed(mdl, X_val_slice, n_classes)
                try:
                    loss = log_loss(y_val_encoded, y_pred_proba)
                except ValueError:
                    loss = 1.0
                scores_per_model.append(1.0 / (loss + 1e-7))

            scores_per_model = np.array(scores_per_model)
            if scores_per_model.sum() > 0:
                scores_per_model /= scores_per_model.sum()
            else:
                scores_per_model = np.ones_like(scores_per_model) / len(scores_per_model)
            device_val_scores.append(scores_per_model)
        else:
            device_val_scores.append(np.ones(len(models_per_device)) / len(models_per_device)
                                     if models_per_device else np.array([1.0]))

    # Collapse per-model weights into scalar per-device weights
    device_weights = np.array([
        np.mean(w) if isinstance(w, np.ndarray) else float(w)
        for w in device_val_scores
    ], dtype=float)

    return d_residuals, d_models, device_embeddings, device_weights


def edge_layer_boosting(e_groups, d_embeddings, d_residuals,
                        n_classes, X_ftune=None, y_ftune=None,
                        device_weights=None):
    """
    Vectorized edge-level sequential boosting with index tracking.
    Preserves signed device residuals (does NOT force non-negative),
    uses predict_proba_fixed for all probability predictions,
    and applies learning-rate scaled residual subtraction.
    """
    print("""
    ************************************************************
    *                  STARTING EDGE LAYER                     *
    *   Devices -> Edge-level boosted ensemble -> Edge output   *
    ************************************************************
    """)

    e_models = []
    e_residuals = []
    edge_embeddings_list = []
    edge_outputs = []
    global_pred_blocks = []
    edge_sample_slices = {}

    global_offset = 0  # tracks row position in global prediction matrix
    eps_residual = 1e-6

    for edge_idx, edge_devices in enumerate(e_groups):
        prev_y_pseudo_edge = None  # initialize before edge boosting loop

        # select finetune data for this edge (if provided)
        if X_ftune is not None and y_ftune is not None:
            X_valid_edge = X_ftune[edge_idx]
            y_valid_edge = y_ftune[edge_idx]
        else:
            X_valid_edge, y_valid_edge = None, None

        if len(edge_devices) == 0:
            e_models.append([])
            e_residuals.append(None)
            edge_embeddings_list.append(None)
            edge_outputs.append(None)
            print(f"Edge {edge_idx}: no devices, skipping.")
            continue

        # -----------------------------
        # 1. Gather embeddings + residuals
        # -----------------------------
        embeddings_list = [d_embeddings[i] for i in edge_devices]
        residual_list = [d_residuals[i] if d_residuals[i] is not None else
                         np.zeros((emb.shape[0], n_classes)) for emb, i in zip(embeddings_list, edge_devices)]

        # Device weights (per-device scalar) -> normalized across devices for this edge
        if device_weights is None:
            weights = np.ones(len(edge_devices), dtype=float)
        else:
            weights = np.array([device_weights[i] for i in edge_devices], dtype=float)

        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones_like(weights) / len(weights)

        # pad embeddings to same width (column dimension)
        max_cols = max(e.shape[1] for e in embeddings_list)
        padded_embeddings = [
            np.pad(e, ((0, 0), (0, max_cols - e.shape[1])), mode="constant")
            for e in embeddings_list
        ]
        # Stack device embeddings vertically so rows = sum(dev_rows)
        X_edge = np.vstack(padded_embeddings)

        # Stack residuals with per-device weights (weighted per-sample residuals)
        # Note: residual_list[i] shape must match embeddings_list[i] rows
        weighted_residual_rows = []
        for r, w in zip(residual_list, weights):
            # ensure r has correct width n_classes (pad/truncate if needed)
            if r.shape[1] < n_classes:
                r = np.pad(r, ((0, 0), (0, n_classes - r.shape[1])), mode="constant")
            elif r.shape[1] > n_classes:
                r = r[:, :n_classes]
            weighted_residual_rows.append(r * w)
        residual_edge = np.vstack(weighted_residual_rows)

        # Keep sign information; clip to safe range
        residual_edge = np.clip(residual_edge, -1 + eps_residual, 1 - eps_residual)

        models_per_edge = []

        # -----------------------------
        # 2. Sequential boosting (edge-level uses pseudo-labels)
        # -----------------------------
        boosting_rounds = config["edge_boosting_rounds"]
        for t in range(boosting_rounds):
            # Stabilize residuals without changing magnitude scaling
            y_pseudo = np.argmax(residual_edge + 1e-9 * np.random.randn(*residual_edge.shape), axis=1)
            unique_classes = np.unique(y_pseudo)

            if len(unique_classes) < 2:
                print(f"Edge {edge_idx}, round {t}: only {len(unique_classes)} class(es), skipping boosting.")
                break
            if np.sum(np.abs(residual_edge)) < config["eps_threshold"]:
                print(f"Edge {edge_idx}, round {t}: residuals below threshold, stopping.")
                break

            if prev_y_pseudo_edge is not None:
                changes = np.mean(prev_y_pseudo_edge != y_pseudo)
                if changes < 0.01:
                    print(f"Edge {edge_idx}, round {t}: pseudo-labels stabilized, stopping.")
                    break

            prev_y_pseudo_edge = y_pseudo.copy()  # update for next round

            init_model = models_per_edge[-1] if models_per_edge else None
            X_valid_slice = (X_valid_edge[:, :X_edge.shape[1]] if (X_valid_edge is not None and X_valid_edge.ndim == 2)
                             else X_valid_edge)

            model = train_lightgbm(
                X_edge, y_pseudo,
                X_valid=X_valid_slice,
                y_valid=y_valid_edge,
                early_stopping_rounds=config["early_stopping_rounds_edge"],
                learning_rate=config["learning_rate_edge"],
                init_model=init_model
            )

            # Use the safe wrapper
            pred_proba = predict_proba_fixed(model, X_edge, n_classes)

            # Update residuals with learning rate scaling, preserving sign
            residual_edge = residual_edge - config["learning_rate_edge"] * pred_proba
            residual_edge = np.clip(residual_edge, -1 + eps_residual, 1 - eps_residual)

            models_per_edge.append(model)

        # -----------------------------
        # 3. Store results
        # -----------------------------
        e_models.append(models_per_edge)
        e_residuals.append(residual_edge.copy())
        edge_embeddings_list.append(X_edge)

        if models_per_edge:
            # Average predictions over boosting rounds using predict_proba_fixed
            model_preds = np.array([predict_proba_fixed(m, X_edge, n_classes) for m in models_per_edge])
            e_pred_avg = np.mean(model_preds, axis=0)
        else:
            e_pred_avg = np.zeros((X_edge.shape[0], n_classes))

        edge_outputs.append(e_pred_avg)

        # --- assign global indices per device ---
        row_start = 0
        for dev_idx, emb in zip(edge_devices, embeddings_list):
            n_dev = emb.shape[0]
            row_end = row_start + n_dev
            global_idx = np.arange(global_offset + row_start, global_offset + row_end)
            edge_sample_slices[dev_idx] = global_idx
            row_start = row_end

        global_pred_blocks.append(e_pred_avg)
        global_offset += X_edge.shape[0]

    # -----------------------------
    # Final global prediction matrix (stack edges)
    # -----------------------------
    if global_pred_blocks:
        padded_blocks = []
        for block in global_pred_blocks:
            if block.shape[1] < n_classes:
                pad_width = n_classes - block.shape[1]
                block = np.hstack([block, np.zeros((block.shape[0], pad_width))])
            padded_blocks.append(block)
        global_pred_matrix = np.vstack(padded_blocks)
    else:
        global_pred_matrix = np.empty((0, n_classes))

    return edge_outputs, e_models, e_residuals, edge_embeddings_list, global_pred_matrix, edge_sample_slices


def global_layer_bayesian_aggregation(e_outputs, e_embeddings, e_residuals=None,
                                      n_classes=2, verbose=True):
    """
    Global Bayesian aggregation using edge outputs, weighting edges by residual reliability.
    Uses pseudo-labels internally instead of true labels.
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
    # 1. Filter valid edges
    # -----------------------------
    valid_edges = [i for i, (out, emb) in enumerate(zip(e_outputs, e_embeddings))
                   if out is not None and emb is not None]
    if not valid_edges:
        raise ValueError("No valid edges found!")

    # -----------------------------
    # 2. Stack predictions across edges
    # -----------------------------
    edge_preds_list = []
    for i in valid_edges:
        out = e_outputs[i]
        if out.shape[1] < n_classes:
            out = np.pad(out, ((0, 0), (0, n_classes - out.shape[1])), mode='constant')
        edge_preds_list.append(out)

    H_global = np.vstack(edge_preds_list)  # (total_samples, num_classes)
    n_samples_total = H_global.shape[0]

    # -----------------------------
    # 3. Compute per-sample reliability weights from residuals (if available)
    # -----------------------------
    eps = 1e-6
    if e_residuals is not None:
        # Build a single vector of per-sample variance-based reliabilities matching H_global stacking
        weights_list = []
        for i in valid_edges:
            r = e_residuals[i]
            if r is None:
                # fallback to uniform small weight
                weights_list.append(np.ones((edge_preds_list[len(weights_list)].shape[0],)) * eps)
                continue
            # Ensure r has correct class dim
            if r.shape[1] < n_classes:
                r = np.pad(r, ((0, 0), (0, n_classes - r.shape[1])), mode="constant")
            elif r.shape[1] > n_classes:
                r = r[:, :n_classes]
            # reliability = 1 / (mean squared residual + tiny_eps)
            per_sample_mse = np.mean(r ** 2, axis=1)
            reliability = 1.0 / (per_sample_mse + eps)
            # cap reliability to reasonable range to avoid extreme domination
            reliability = np.clip(reliability, a_min=eps, a_max=1e6)
            weights_list.append(reliability)
        edge_weights = np.hstack(weights_list)
    else:
        edge_weights = np.ones(n_samples_total, dtype=float)

    # Normalize weights
    total = edge_weights.sum()
    if total <= 0:
        edge_weights = np.ones_like(edge_weights) / edge_weights.size
    else:
        edge_weights = edge_weights / total

    if verbose:
        print(f"Normalized edge weights sum to {edge_weights.sum():.6f}")

    # -----------------------------
    # 4. Compute pseudo-labels and residuals internally
    # -----------------------------
    y_pseudo_global = np.argmax(H_global, axis=1)
    y_pseudo_onehot = np.zeros((n_samples_total, n_classes))
    y_pseudo_onehot[np.arange(n_samples_total), y_pseudo_global] = 1

    global_residuals = y_pseudo_onehot - H_global

    # -----------------------------
    # 5. Compute Bayesian correction (weighted)
    # -----------------------------
    # alpha: weighted average predictions per-class
    alpha = np.sum(H_global * edge_weights[:, None], axis=0)
    # beta: weighted average residual correction per-class
    beta = np.sum(global_residuals * edge_weights[:, None], axis=0)

    # -----------------------------
    # 6. Final global softmax predictions (apply correction)
    # -----------------------------
    logits = H_global + beta  # broadcast beta to rows
    # Numerically stable softmax per row
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

def forward_pass(devices_data, edge_groups, le, num_classes,X_edge_finetunes,y_edge_finetunes,residuals_devices=None, device_models=None):
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
    residuals_devices, device_models, device_embeddings, device_weights = device_layer_boosting(
        devices_data, residuals_devices, device_models,
        le, num_classes
    )
    assert device_embeddings is not None, "Device embeddings returned as None!"

    #return (residuals_devices, device_models, device_embeddings, device_val_scores)


    # -----------------------------
    # 2. Edge Layer
    # -----------------------------
    # Track total samples for global prediction matrix
    n_samples = sum(e.shape[0] for e in device_embeddings)
    edge_outputs, edge_models, residuals_edges, edge_embeddings_list, global_pred_matrix, edge_sample_slices = edge_layer_boosting(
        e_groups=edge_groups,
        d_embeddings=device_embeddings,
        d_residuals=residuals_devices,
        n_classes=num_classes,
        X_ftune=X_edge_finetunes,
        y_ftune=y_edge_finetunes,
        device_weights=device_weights
    )

    # -----------------------------
    # Build Global Ground Truth
    # -----------------------------
    y_global_true = np.full((n_samples,), -1, dtype=int)
    written_mask = np.zeros(n_samples, dtype=bool)
    y_true_per_edge = []

    for edge_idx, edge_devices in enumerate(edge_groups):
        edge_labels = []

        for dev_idx in edge_devices:
            idxs = edge_sample_slices.get(dev_idx, np.arange(device_embeddings[dev_idx].shape[0]))
            y_dev = np.array(y_edge_finetunes[dev_idx])

            # Align label size safely
            if y_dev.size < idxs.size:
                y_dev = np.pad(y_dev, (0, idxs.size - y_dev.size), mode="edge")
            elif y_dev.size > idxs.size:
                y_dev = y_dev[:idxs.size]

            # Write to global only once
            not_written = ~written_mask[idxs]
            y_global_true[idxs[not_written]] = y_dev[not_written]
            written_mask[idxs[not_written]] = True

            edge_labels.append(y_dev)

        y_true_per_edge.append(np.hstack(edge_labels))

    # -----------------------------
    # 4. Global Layer Aggregation
    # -----------------------------
    y_global_pred, global_residuals, theta_global = global_layer_bayesian_aggregation(
        e_outputs=edge_outputs,
        e_embeddings=edge_embeddings_list,
        e_residuals=residuals_edges,
        n_classes=num_classes
    )


    #y_global_true = np.hstack(y_true_per_edge)  # <- ensures labels match stacked preds

    # Accuracy
    #y_pred_labels = y_global_pred.argmax(axis=1)
    #global_acc = compute_accuracy(y_global_true, y_pred_labels)

    #print(f"Global Accuracy: {global_acc:.4f}")

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
        edge_sample_slices
    )

def backward_pass(edge_models, device_models,
                  edge_embeddings, device_embeddings,
                  y_true_per_edge,
                  edge_sample_slices=None,
                  global_pred_matrix=None,
                  n_classes=2,
                  use_classification=True,
                  verbose=True,
                  le=None):
    """
    Hierarchical backward feedback that updates edge and device models.

    Args:
        edge_models: list[list(models)]     - models trained per edge (list of models)
        device_models: list[list(models)]   - models trained per device
        edge_embeddings: list[np.ndarray]   - X_edge matrices (stacked per-device embeddings)
        device_embeddings: list[np.ndarray] - device-level embeddings
        y_true_per_edge: list[np.ndarray]   - true labels per edge (1D arrays of ints or 1-hot)
        edge_sample_slices: dict (device_idx -> global_indices) optional
            Mapping from device index to row indices in the forward global prediction matrix.
            Required to align updated edge preds back to devices. If None, device updates
            that require alignment will be skipped.
        global_pred_matrix: np.ndarray or None
            If available, can be used for alignment checks. Not required.
        n_classes: int
        use_classification: bool
            If True -> re-fit classifiers using integer labels.
            If False -> re-fit regressors on residuals (continuous corrections).
        verbose: bool
        le: optional LabelEncoder used in forward (if labels are strings)

    Returns:
        updated_edge_models, updated_device_models, updated_edge_preds_stacked

    Notes:
        - This function replaces each existing model with a freshly trained model
          with the same basic hyperparameters (n_estimators, learning_rate, max_depth).
        - If edge_sample_slices is provided it will map updated edge predictions back
          to device rows and compute device residuals for device-level updates.
    """
    if verbose:
        print("\n" + "*"*60)
        print("*" + " " * 26 + "STARTING BACKWARD PASS" + " " * 26 + "*")
        print("*" * 60 + "\n")

    # Defensive checks
    num_edges = len(edge_models)
    assert len(edge_embeddings) == num_edges, "edge_embeddings and edge_models length mismatch"
    assert len(y_true_per_edge) == num_edges, "y_true_per_edge must align with edge_models"

    # -----------------------------
    # 1) Update Edge Models (train to predict true per-edge labels)
    # -----------------------------
    updated_edge_preds_list = []   # store per-edge average predictions (for stacking)
    updated_edge_models = []

    for ei in range(num_edges):
        models = edge_models[ei]
        X_edge = edge_embeddings[ei]
        y_edge = y_true_per_edge[ei]

        if X_edge is None or models is None or len(models) == 0:
            # nothing to update; keep placeholder
            updated_edge_models.append([])
            # push zeros so stacking works
            updated_edge_preds_list.append(np.zeros((0, n_classes)))
            if verbose:
                print(f"Edge {ei}: no models or no data, skipping.")
            continue

        # Convert y_edge to integer labels if necessary
        if y_edge.ndim > 1 and y_edge.shape[1] == n_classes:
            # one-hot -> int
            y_edge_labels = np.argmax(y_edge, axis=1)
        else:
            y_edge_labels = np.asarray(y_edge).ravel()
            # If labels are strings and LabelEncoder provided, transform
            if le is not None and y_edge_labels.dtype.kind in {'U', 'S', 'O'}:
                try:
                    y_edge_labels = le.transform(y_edge_labels)
                except Exception:
                    pass

        # Re-train each model for this edge (safe replacement)
        new_models_for_edge = []
        preds_per_model = []
        for m in models:
            # extract simple params with fallbacks
            params = {}
            n_est = getattr(m, "n_estimators", None) or getattr(m, "params", {}).get("n_estimators", None) or 100
            lr = getattr(m, "learning_rate", None) or getattr(m, "params", {}).get("learning_rate", None) or 0.1
            md = getattr(m, "max_depth", None) or getattr(m, "params", {}).get("max_depth", None)

            if use_classification:
                clf = LGBMClassifier(n_estimators=n_est, learning_rate=lr, max_depth=md, random_state=getattr(m, "random_state", 42), num_class=n_classes)
                # fit classifier
                try:
                    clf.fit(X_edge, y_edge_labels)
                    preds = clf.predict_proba(X_edge)
                except Exception as e:
                    if verbose:
                        print(f"Edge {ei}: classifier fit failed with {e}, trying fallback simple fit.")
                    clf = LGBMClassifier(n_estimators=n_est, learning_rate=lr, max_depth=md, random_state=getattr(m, "random_state", 42), num_class=n_classes)
                    clf.fit(X_edge, y_edge_labels)
                    preds = clf.predict_proba(X_edge)
                new_models_for_edge.append(clf)
                preds_per_model.append(preds)
            else:
                # regression: train to predict continuous residuals (here we use one-vs-all residual for each class)
                # Prepare residual targets: one-hot minus current model avg if global_pred_matrix provided
                # Fallback to trying to learn one-hot labels as continuous targets
                if global_pred_matrix is not None:
                    # try to slice rows for this edge from global_pred_matrix by matching X_edge row count
                    try:
                        preds_init = m.predict_proba(X_edge) if hasattr(m, "predict_proba") else np.zeros((X_edge.shape[0], n_classes))
                    except Exception:
                        preds_init = np.zeros((X_edge.shape[0], n_classes))
                    if preds_init.shape[1] < n_classes:
                        preds_init = np.pad(preds_init, ((0,0),(0,n_classes - preds_init.shape[1])), mode="constant")
                    y_edge_onehot = np.zeros((X_edge.shape[0], n_classes))
                    y_edge_idx = np.asarray(y_edge_labels).ravel()
                    y_edge_onehot[np.arange(X_edge.shape[0]), y_edge_idx] = 1.0
                    residual_targets = (y_edge_onehot - preds_init).astype(np.float32)
                else:
                    # fallback: use one-hot as continuous target
                    y_edge_onehot = np.zeros((X_edge.shape[0], n_classes))
                    y_edge_idx = np.asarray(y_edge_labels).ravel()
                    y_edge_onehot[np.arange(X_edge.shape[0]), y_edge_idx] = 1.0
                    residual_targets = y_edge_onehot.astype(np.float32)

                # train one regressor per original model (we'll flatten outputs into NxC)
                reg = LGBMRegressor(n_estimators=n_est, learning_rate=lr, max_depth=md, random_state=getattr(m, "random_state", 42))
                # Flatten target to shape (N, C) -> for sklearn API, we train C separate regressors or train vector target if supported.
                try:
                    # Some versions support multioutput directly
                    reg.fit(X_edge, residual_targets)
                    preds = reg.predict(X_edge)
                    preds = np.atleast_2d(preds)
                    if preds.ndim == 2 and preds.shape[1] == n_classes:
                        preds_per_model.append(preds)
                    else:
                        # reshape if necessary
                        preds_per_model.append(np.reshape(preds, (X_edge.shape[0], -1)))
                except Exception as e:
                    if verbose:
                        print(f"Edge {ei}: regressor fit failed ({e}), falling back to per-class regressors.")
                    # per-class regressors
                    per_class_preds = []
                    for c in range(n_classes):
                        rc = LGBMRegressor(n_estimators=n_est, learning_rate=lr, max_depth=md, random_state=getattr(m, "random_state", 42))
                        rc.fit(X_edge, residual_targets[:, c])
                        per_class_preds.append(rc.predict(X_edge))
                    preds = np.vstack(per_class_preds).T
                    preds_per_model.append(preds)
                    reg = None  # we replaced with per-class regressors (not stored)
                # store reg if training succeeded
                if reg is not None:
                    new_models_for_edge.append(reg)
                else:
                    # if we used per-class regressors, store placeholders (not ideal but safe)
                    new_models_for_edge.append(None)

        # Average preds across models
        if preds_per_model:
            preds_avg = np.mean(np.array(preds_per_model), axis=0)
        else:
            preds_avg = np.zeros((X_edge.shape[0], n_classes))

        updated_edge_models.append(new_models_for_edge)
        updated_edge_preds_list.append(preds_avg)

        if verbose:
            print(f"Edge {ei}: updated {len(new_models_for_edge)} model(s), preds shape {preds_avg.shape}")

    # Stack updated edge preds vertically (matching forward stacking)
    if updated_edge_preds_list:
        updated_edge_preds_stacked = np.vstack(updated_edge_preds_list)
    else:
        updated_edge_preds_stacked = np.empty((0, n_classes))

    # -----------------------------
    # 2) Map updated edge preds back to devices and update device models
    # -----------------------------
    updated_device_models = list(device_models)  # shallow copy
    # Build a global label array from y_true_per_edge for device label retrieval
    try:
        y_global_labels = np.hstack([np.asarray(y).ravel() for y in y_true_per_edge])
    except Exception:
        y_global_labels = None

    for dev_idx, models in enumerate(device_models):
        X_dev = device_embeddings[dev_idx]
        if X_dev is None or models is None or len(models) == 0:
            if verbose:
                print(f"Device {dev_idx}: no models or no embeddings, skipping.")
            continue

        if edge_sample_slices is None:
            if verbose:
                print(f"Device {dev_idx}: edge_sample_slices not provided -> skipping device update.")
            continue

        # get global indices for this device
        global_idxs = edge_sample_slices.get(dev_idx, None)
        if global_idxs is None:
            if verbose:
                print(f"Device {dev_idx}: no global indices found in edge_sample_slices, skipping.")
            continue

        # Ensure updated_edge_preds_stacked has rows for these indices
        if updated_edge_preds_stacked.shape[0] <= global_idxs.max():
            if verbose:
                print(f"Device {dev_idx}: updated edge preds do not cover requested indices -> skipping.")
            continue

        # Predicted probs for this device (from edges after update)
        preds_for_device = updated_edge_preds_stacked[global_idxs, :]
        # Device true labels from stacked y_true_per_edge if available
        if y_global_labels is not None and y_global_labels.size > 0 and global_idxs.max() < y_global_labels.size:
            y_dev_true = y_global_labels[global_idxs]
        else:
            # fallback: produce pseudo-labels from preds_for_device
            y_dev_true = np.argmax(preds_for_device, axis=1)

        # If classification: train devices to predict integer labels
        if use_classification:
            # ensure integer labels
            y_dev_labels = np.asarray(y_dev_true).ravel()
            new_models_for_device = []
            for m in models:
                n_est = getattr(m, "n_estimators", None) or 100
                lr = getattr(m, "learning_rate", None) or 0.1
                md = getattr(m, "max_depth", None)
                clf = LGBMClassifier(n_estimators=n_est, learning_rate=lr, max_depth=md, random_state=getattr(m, "random_state", 42), num_class=n_classes)
                try:
                    clf.fit(X_dev, y_dev_labels)
                except Exception as e:
                    if verbose:
                        print(f"Device {dev_idx}: classifier fit failed ({e}), retrying default params.")
                    clf = LGBMClassifier(n_estimators=n_est, learning_rate=lr, max_depth=md, random_state=getattr(m, "random_state", 42), num_class=n_classes)
                    clf.fit(X_dev, y_dev_labels)
                new_models_for_device.append(clf)
            updated_device_models[dev_idx] = new_models_for_device
            if verbose:
                print(f"Device {dev_idx}: updated {len(new_models_for_device)} classifier(s).")
        else:
            # regression: targets = (one-hot true) - (edge preds) => residual corrections to learn
            y_dev_onehot = np.zeros((preds_for_device.shape[0], n_classes))
            y_dev_onehot[np.arange(preds_for_device.shape[0]), np.asarray(y_dev_true).ravel()] = 1.0
            residual_targets = (y_dev_onehot - preds_for_device).astype(np.float32)

            new_models_for_device = []
            for m in models:
                n_est = getattr(m, "n_estimators", None) or 100
                lr = getattr(m, "learning_rate", None) or 0.1
                md = getattr(m, "max_depth", None)
                reg = LGBMRegressor(n_estimators=n_est, learning_rate=lr, max_depth=md, random_state=getattr(m, "random_state", 42))
                try:
                    reg.fit(X_dev, residual_targets)
                    new_models_for_device.append(reg)
                except Exception as e:
                    if verbose:
                        print(f"Device {dev_idx}: regressor fit failed ({e}), trying per-class regressors.")
                    # fallback per-class regressors
                    per_class_regs = []
                    for c in range(n_classes):
                        rc = LGBMRegressor(n_estimators=n_est, learning_rate=lr, max_depth=md, random_state=getattr(m, "random_state", 42))
                        rc.fit(X_dev, residual_targets[:, c])
                        per_class_regs.append(rc)
                    new_models_for_device.append(per_class_regs)  # nested list: indicates multi-output trained separately
            updated_device_models[dev_idx] = new_models_for_device
            if verbose:
                print(f"Device {dev_idx}: updated {len(new_models_for_device)} regressor(s).")

    if verbose:
        print("Backward hierarchical feedback completed safely.\n")

    return updated_edge_models, updated_device_models, updated_edge_preds_stacked



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
    Hierarchical non-IID partition for devices and edges.

    Each device:
        - X_train, y_train → for device-level training
        - X_device_finetune, y_device_finetune → optional local finetune
        - X_edge_finetune, y_edge_finetune → used to train edge models

    Each edge:
        - Aggregates X_edge_finetune / y_edge_finetune from its devices

    Returns:
        devices_data: list of tuples per device
        edge_groups: list of device index lists per edge
        edge_data: list of tuples per edge
    """

    # Convert to numpy arrays
    X_np = X.to_numpy() if isinstance(X, (pd.DataFrame, pd.Series)) else np.array(X)
    y_np = y.to_numpy() if isinstance(y, (pd.DataFrame, pd.Series)) else np.array(y)

    rng = np.random.default_rng(seed)
    unique_classes = np.unique(y_np)

    # -------------------------
    # Step 1: Dirichlet Non-IID split for devices
    # -------------------------
    device_indices = [[] for _ in range(num_devices)]
    for c in unique_classes:
        class_idx = np.where(y_np == c)[0]
        rng.shuffle(class_idx)

        # Dirichlet proportions
        proportions = rng.dirichlet(alpha * np.ones(num_devices))
        counts = np.maximum((proportions * len(class_idx)).astype(int), 1)
        diff = len(class_idx) - counts.sum()
        counts[np.argmax(counts)] += diff  # fix rounding
        splits = np.split(class_idx, np.cumsum(counts)[:-1])
        for dev_id, split in enumerate(splits):
            device_indices[dev_id].extend(split.tolist())

    # -------------------------
    # Step 2: Device-level train / finetune splits
    # -------------------------
    devices_data = []
    for dev_id, idxs in enumerate(device_indices):
        X_dev, y_dev = X_np[idxs], y_np[idxs]

        # Device train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_dev, y_dev, test_size=0.1, random_state=seed
        )

        # Split training into device finetune + edge finetune
        X_train, X_edge_finetune, y_train, y_edge_finetune = train_test_split(
            X_train, y_train, test_size=0.1, random_state=seed
        )

        X_train, X_device_finetune, y_train, y_device_finetune = train_test_split(
            X_train, y_train, test_size=0.1, random_state=seed
        )

        devices_data.append((
            X_train, X_test, y_train, y_test,
            X_edge_finetune, y_edge_finetune,
            X_device_finetune, y_device_finetune
        ))

    # -------------------------
    # Step 3: Assign devices to edges
    # -------------------------
    def make_edge_groups(num_devices, n_edges, random_state=None):
        rng_local = np.random.default_rng(random_state)
        devices = np.arange(num_devices)
        rng_local.shuffle(devices)
        return np.array_split(devices, n_edges)

    edge_groups = make_edge_groups(num_devices, n_edges, random_state=seed)

    # -------------------------
    # Step 4: Aggregate edge finetune data
    # -------------------------
    edge_finetune_data = []
    for edge_idx, devices_in_edge in enumerate(edge_groups):
        X_edge_all, y_edge_all = [], []
        for dev_id in devices_in_edge:
            # Take only the device's edge-finetune subset
            X_edge_all.append(devices_data[dev_id][4])  # X_edge_finetune
            y_edge_all.append(devices_data[dev_id][5])  # y_edge_finetune
        X_edge_all = np.vstack(X_edge_all)
        y_edge_all = np.concatenate(y_edge_all)
        edge_finetune_data.append((X_edge_all, y_edge_all))

    return devices_data, edge_groups, edge_finetune_data

"""
def dirichlet_partition_for_devices_edges(X, y, num_devices, device_per_edge, n_edges):
    \"""Return device_data and hierarchical_data for devices and edges.\"""
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
    y_global_pred_test=None,
    num_classes=2,
    verbose=True,
    device_model_weights=None,
    edge_model_weights=None,
    device_val_scores=None,
    edge_val_scores=None,
    calibrate=True,
    device_weight_mode="samples",
):
    \"""
    Compute hierarchical accuracies (device, edge, global) with optional calibration
    and weighted averaging. Also computes metrics comparing device vs global and
    edge vs global predictions.

    Parameters:
        y_global_pred_test: Precomputed global predictions [n_samples_total x num_classes]
    \"""

    # ---------------------------
    # Suppress runtime warnings locally
    # ---------------------------
    warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names.*")
    old_err_settings = np.seterr(divide='ignore', over='ignore', invalid='ignore')

    # ---------------------------
    # Helper functions
    # ---------------------------
    def expand_labels(y, n_samples):
        y = np.atleast_1d(y)
        if y.size == 1:
            return np.full(n_samples, y[0])
        elif y.size != n_samples:
            y_new = np.zeros(n_samples, dtype=y.dtype)
            n_copy = min(y.size, n_samples)
            y_new[:n_copy] = y[:n_copy]
            if n_samples > y.size:
                y_new[n_copy:] = y[-1]
            return y_new
        return y

    def normalize_weights(weights):
        w = np.asarray(weights, dtype=float)
        w[np.isnan(w)] = 0.0
        s = w.sum()
        if s <= 0:
            return np.ones_like(w) / len(w)
        return w / s

    def calibrate_probs(y_true, probs):
        y_true = np.atleast_1d(y_true)
        probs = np.atleast_2d(probs)
        if probs.shape[0] == 1 and y_true.shape[0] > 1:
            probs = np.tile(probs, (y_true.shape[0], 1))
        if probs.shape[1] == 2:
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(probs[:, 1], y_true)
            calibrated_p1 = ir.transform(probs[:, 1])
            return np.column_stack([1 - calibrated_p1, calibrated_p1])
        return probs

    def weighted_average(preds, weights):
        w = normalize_weights(weights)
        pred_accum = np.zeros_like(preds[0])
        for wi, pi in zip(w, preds):
            pred_accum += wi * pi
        return pred_accum

    def predict_proba_safe(model, X, num_classes):
        X = np.atleast_2d(X)
        n_feat = len(model.feature_name_)
        if X.shape[1] < n_feat:
            X = np.hstack([X, np.zeros((X.shape[0], n_feat - X.shape[1]))])
        elif X.shape[1] > n_feat:
            X = X[:, :n_feat]
        return model.predict_proba(X)

    # ---------------------------
    # 1️⃣ DEVICE LEVEL
    # ---------------------------
    device_preds = []
    device_accs = []

    for dev_idx, X_dev in enumerate(X_test):
        n_samples = X_dev.shape[0]
        trained_models = device_models[dev_idx] if dev_idx < len(device_models) else []

        preds, w_list = [], []
        for m_idx, mdl in enumerate(trained_models):
            if mdl is None:
                continue
            p = predict_proba_safe(mdl, X_dev, num_classes)
            if calibrate:
                y_true_dev = expand_labels(y_test[dev_idx], n_samples)
                p = calibrate_probs(y_true_dev, p)
            preds.append(p)

            if device_model_weights and dev_idx < len(device_model_weights) and m_idx < len(device_model_weights[dev_idx]):
                w_list.append(device_model_weights[dev_idx][m_idx])
            elif device_val_scores and dev_idx < len(device_val_scores):
                w_list.append(device_val_scores[dev_idx][m_idx])
            else:
                w_list.append(1.0)

        if len(preds) == 0:
            if y_global_pred_test is not None:
                pred_accum = np.tile(np.clip(np.mean(y_global_pred_test, axis=0), 1e-7, 1-1e-7), (n_samples, 1))
            else:
                pred_accum = np.full((n_samples, num_classes), 1/num_classes)
        else:
            pred_accum = weighted_average(preds, w_list)

        device_preds.append(pred_accum)
        y_true_dev = expand_labels(y_test[dev_idx], n_samples)
        device_accs.append(accuracy_score(y_true_dev, pred_accum.argmax(axis=1)))

    # ---------------------------
    # 2️⃣ EDGE LEVEL
    # ---------------------------
    edge_preds = []
    edge_accs = []

    for edge_idx, edge_devices in enumerate(edge_groups):
        device_edge_preds = [device_preds[d] for d in edge_devices]
        y_edge_true_list = [expand_labels(y_test[d], X_test[d].shape[0]) for d in edge_devices]
        y_edge_true = np.hstack(y_edge_true_list)
        X_edge = np.vstack([np.atleast_2d(X_test[d]) for d in edge_devices])

        if device_weight_mode == "samples":
            w_devices = [X_test[d].shape[0] for d in edge_devices]
        elif device_weight_mode == "val_score" and device_val_scores:
            w_devices = [np.mean(device_val_scores[d]) for d in edge_devices]
        else:
            w_devices = [1.0] * len(edge_devices)
        w_devices = normalize_weights(w_devices)
        device_mix_pred = np.zeros_like(device_edge_preds[0])
        for wi, dp in zip(w_devices, device_edge_preds):
            device_mix_pred += wi * dp

        # edge models
        trained_edge_models = edge_models[edge_idx] if edge_idx < len(edge_models) else []
        edge_model_preds, edge_weights = [], []

        for m_idx, mdl in enumerate(trained_edge_models):
            if mdl is None:
                continue
            p = predict_proba_safe(mdl, X_edge, num_classes)
            if calibrate:
                p = calibrate_probs(y_edge_true, p)
            edge_model_preds.append(p)
            if edge_model_weights and edge_idx < len(edge_model_weights) and m_idx < len(edge_model_weights[edge_idx]):
                edge_weights.append(edge_model_weights[edge_idx][m_idx])
            elif edge_val_scores and edge_idx < len(edge_val_scores):
                edge_weights.append(edge_val_scores[edge_idx][m_idx])
            else:
                edge_weights.append(1.0)

        if len(edge_model_preds) > 0:
            edge_pred_accum = weighted_average(edge_model_preds, edge_weights)
            pred_accum = 0.5 * (edge_pred_accum + device_mix_pred)
        else:
            pred_accum = device_mix_pred

        edge_preds.append(pred_accum)
        edge_accs.append(accuracy_score(y_edge_true, pred_accum.argmax(axis=1)))

    # ---------------------------
    # 3️⃣ GLOBAL LEVEL
    # ---------------------------
    if y_global_pred_test is not None:
        global_pred = np.atleast_2d(y_global_pred_test)
    else:
        # fallback: average across devices
        global_pred = np.vstack([np.clip(np.nan_to_num(dp, nan=1e-7, posinf=1.0, neginf=0.0), 1e-7, 1-1e-7)
                                 for dp in device_preds])

    global_labels = global_pred.argmax(axis=1)
    y_global_true = np.concatenate([expand_labels(y_test[d], X_test[d].shape[0]) for d in range(len(X_test))])
    global_acc = accuracy_score(y_global_true, global_labels)

    # ---------------------------
    # 4️⃣ DEVICE vs GLOBAL
    # ---------------------------
    device_vs_global = []
    idx_offset = 0
    for dev_idx, X_dev in enumerate(X_test):
        n_samples = X_dev.shape[0]
        y_true_dev = expand_labels(y_test[dev_idx], n_samples)
        global_pred_dev = global_pred[idx_offset:idx_offset+n_samples]
        device_vs_global.append(accuracy_score(y_true_dev, global_pred_dev.argmax(axis=1)))
        idx_offset += n_samples

    # ---------------------------
    # 5️⃣ EDGE vs GLOBAL
    # ---------------------------
    edge_vs_global = []
    idx_offset = 0
    for edge_idx, edge_devices in enumerate(edge_groups):
        y_edge_true_list = [expand_labels(y_test[d], X_test[d].shape[0]) for d in edge_devices]
        y_edge_true = np.hstack(y_edge_true_list)
        n_samples_edge = sum([X_test[d].shape[0] for d in edge_devices])
        edge_pred_blocks = []
        row_start = 0
        for d in edge_devices:
            n_dev = X_test[d].shape[0]
            edge_pred_blocks.append(edge_preds[edge_idx][row_start:row_start+n_dev])
            row_start += n_dev
        edge_pred_full = np.vstack(edge_pred_blocks)
        edge_vs_global.append(accuracy_score(y_edge_true, edge_pred_full.argmax(axis=1)))
        idx_offset += n_samples_edge

    if verbose:
        print(f"Device mean acc: {np.mean(device_accs):.4f}, "
              f"Edge mean acc: {np.mean(edge_accs):.4f}, "
              f"Global acc: {global_acc:.4f}")

    return {
        "device_preds": device_preds,
        "edge_preds": edge_preds,
        "global_pred": global_pred,
        "device_accs": device_accs,
        "edge_accs": edge_accs,
        "global_acc": global_acc,
        "device_vs_global": device_vs_global,
        "edge_vs_global": edge_vs_global,
    }

"""
# You must define these helpers externally or above:
#   - predict_proba_fixed(model, X, num_classes)
#   - multi_class_brier(y_true, y_pred_probs)
#   - top_k_accuracy(y_true, y_pred_probs, k)

def evaluate_multilevel_performance(
    device_models,
    edge_models,
    edge_groups,
    X_tests, y_tests,
    le,
    num_classes,
    combine_mode="last",
    device_model_weights=None,
    edge_model_weights=None,
    device_val_scores=None,
    edge_val_scores=None,
    calibrate=True,
    device_weight_mode="samples",
    top_k=3,
):
    """
    Unified evaluation for hierarchical federated models:
    Computes per-device, per-edge, and global metrics.

    Returns a dictionary compatible with plot_hpfl_all.
    """

    # ---------------------------
    # Helper functions
    # ---------------------------
    def normalize_weights(weights):
        w = np.asarray(weights, dtype=float)
        w[np.isnan(w)] = 0.0
        s = w.sum()
        if s <= 0:
            return np.ones_like(w) / len(w)
        return w / s

    def calibrate_probs(y_true, probs):
        y_true = np.atleast_1d(y_true)
        probs = np.atleast_2d(probs)
        if probs.shape[0] == 1 and y_true.shape[0] > 1:
            probs = np.tile(probs, (y_true.shape[0], 1))
        if probs.shape[1] == 2:
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(probs[:, 1], y_true)
            calibrated_p1 = ir.transform(probs[:, 1])
            return np.column_stack([1 - calibrated_p1, calibrated_p1])
        return probs

    # ---------------------------
    # Suppress warnings
    # ---------------------------
    warnings.filterwarnings("ignore", category=UserWarning)
    np.seterr(divide='ignore', over='ignore', invalid='ignore')

    # ---------------------------
    # 1️⃣ Device-level metrics
    # ---------------------------
    device_preds, device_accs, log_losses, brier_scores, topk_accuracies = [], [], [], [], []

    for idx, models_per_device in enumerate(device_models):
        if not models_per_device:
            device_accs.append(0.0)
            log_losses.append(np.nan)
            brier_scores.append(np.nan)
            topk_accuracies.append(np.nan)
            device_preds.append(np.zeros((X_tests[idx].shape[0], num_classes)))
            continue

        X_test = np.atleast_2d(X_tests[idx])
        y_test = np.atleast_1d(y_tests[idx])
        y_true = le.transform(y_test)

        # Stack predictions
        preds_list = [predict_proba_fixed(m, X_test, num_classes) for m in models_per_device]

        # Compute weights
        if combine_mode == "weighted":
            if device_val_scores and idx < len(device_val_scores):
                w = np.array(device_val_scores[idx])
            else:
                w = np.ones(len(preds_list))
            w = w / w.sum()
        else:
            w = None

        # Combine predictions
        if combine_mode == "last":
            y_pred_probs = preds_list[-1]
        elif combine_mode == "average":
            y_pred_probs = np.mean(preds_list, axis=0)
        elif combine_mode == "weighted":
            y_pred_probs = np.zeros_like(preds_list[0])
            for pi, wi in zip(preds_list, w):
                y_pred_probs += wi * pi
        elif combine_mode == "hard_vote":
            votes = np.array([np.argmax(p, axis=1) for p in preds_list])
            y_pred_probs = np.zeros((votes.shape[1], num_classes))
            for c in range(num_classes):
                y_pred_probs[:, c] = np.mean(votes == c, axis=0)
        elif combine_mode == "stacked":
            X_stack = np.hstack(preds_list)
            meta = LogisticRegression(max_iter=500)
            meta.fit(X_stack, y_true)
            y_pred_probs = meta.predict_proba(X_stack)
        else:
            raise ValueError(f"Unknown combine_mode: {combine_mode}")

        # Optional calibration
        if calibrate:
            y_pred_probs = calibrate_probs(y_true, y_pred_probs)

        y_pred = np.argmax(y_pred_probs, axis=1)
        device_preds.append(y_pred_probs)
        device_accs.append(accuracy_score(y_true, y_pred))
        try:
            log_losses.append(log_loss(y_true, y_pred_probs))
        except ValueError:
            log_losses.append(np.nan)
        brier_scores.append(multi_class_brier(y_true, y_pred_probs))
        topk_accuracies.append(top_k_accuracy(y_true, y_pred_probs, k=top_k))

    # ---------------------------
    # 2️⃣ Edge-level metrics
    # ---------------------------
    edge_preds, edge_accs = [], []

    for edge_idx, edge_devices in enumerate(edge_groups):
        device_edge_preds = [device_preds[d] for d in edge_devices if d < len(device_preds)]
        y_edge_true_list = [le.transform(y_tests[d]) for d in edge_devices]
        y_edge_true = np.hstack(y_edge_true_list)
        X_edge = np.vstack([X_tests[d] for d in edge_devices])

        if device_weight_mode == "samples":
            w_devices = [X_tests[d].shape[0] for d in edge_devices]
        else:
            w_devices = [1.0] * len(edge_devices)
        w_devices = normalize_weights(w_devices)

        device_mix_pred = np.zeros_like(device_edge_preds[0])
        for wi, dp in zip(w_devices, device_edge_preds):
            device_mix_pred += wi * dp

        edge_models_list = edge_models[edge_idx] if edge_idx < len(edge_models) else []
        edge_model_preds = []
        for mdl in edge_models_list:
            p = predict_proba_fixed(mdl, X_edge, num_classes)
            if calibrate:
                p = calibrate_probs(y_edge_true, p)
            edge_model_preds.append(p)

        if len(edge_model_preds) > 0:
            edge_pred_accum = np.mean(edge_model_preds, axis=0)
            pred_accum = 0.5 * (edge_pred_accum + device_mix_pred)
        else:
            pred_accum = device_mix_pred

        edge_preds.append(pred_accum)
        edge_accs.append(accuracy_score(y_edge_true, pred_accum.argmax(axis=1)))

    # ---------------------------
    # 3️⃣ Global-level metrics
    # ---------------------------
    global_pred = np.vstack(device_preds)
    global_labels = global_pred.argmax(axis=1)
    y_global_true = np.concatenate([le.transform(y_tests[d]) for d in range(len(X_tests))])
    global_acc = accuracy_score(y_global_true, global_labels)

    # ---------------------------
    # 4️⃣ Prepare structured metrics dictionary
    # ---------------------------
    metrics_test = {
        "device_means": device_accs,  # full list per epoch
        "device_stds": np.std(device_accs),  # scalar
        "edge_means": edge_accs,  # full list per epoch
        "edge_stds": np.std(edge_accs),  # scalar
        "global_accs": global_acc,  # scalar
        "device_vs_global": device_accs,  # full list; subtract global later for plotting
        "edge_vs_global": edge_accs,  # full list; subtract global later
        "metrics": {
            "device": {
                "acc": device_accs,
                "logloss": log_losses,
                "brier": brier_scores,
                "topk": topk_accuracies,
                "preds": device_preds,
            },
            "edge": {
                "acc": edge_accs,
                "preds": edge_preds,
            },
            "global": {
                "acc": global_acc,
                "pred": global_pred,
            },
        },
    }

    return metrics_test




# ============================================================
#                  HPFL TRAINING AND EVALUATION
# ============================================================
# - hpfl_train_with_accuracy: hierarchical training loop
#   with device-level and edge-level sequential boosting.
# - evaluate_final_accuracy: compute final accuracy metrics at
#   device, edge, and global (gossip-summary) levels.
# ============================================================

def hpfl_train_with_accuracy(d_data, e_groups, edge_finetune_data, le, n_classes, verbose=True):
    """
    HPFL training loop with forward/backward passes and accuracy tracking.

    Args:
        d_data: list of tuples (X_train, _, y_train, _)
        e_groups: list of lists, devices per edge
        le: LabelEncoder
        n_classes: int
        X_finetune, y_finetune: optional fine-tuning data
        verbose: bool

    Returns:
        device_models, edge_models,
        residuals_devices, residuals_edges,
        y_global_pred, device_embeddings, edge_embeddings,
        metrics_history: dict with per-epoch means & stds
    """

    # -------------------------
    # 1️⃣ Device-level splits
    # -------------------------
    X_trains, X_tests, y_trains, y_tests = [], [], [], []
    X_device_finetunes, y_device_finetunes = [], []

    for dev_tuple in d_data:
        (
            X_train, X_test, y_train, y_test,
            X_edge_finetune, y_edge_finetune,  # ignored here, already aggregated for edges
            X_device_finetune, y_device_finetune
        ) = dev_tuple

        X_trains.append(X_train)
        X_tests.append(X_test)
        y_trains.append(y_train)
        y_tests.append(y_test)
        X_device_finetunes.append(X_device_finetune)
        y_device_finetunes.append(y_device_finetune)

    # -------------------------
    # 2️⃣ Aggregated edge-level finetune data
    # -------------------------
    X_edges_finetune, y_edges_finetune = zip(*edge_finetune_data)
    X_edges_finetune = list(X_edges_finetune)
    y_edges_finetune = list(y_edges_finetune)

    residuals_devices = [None] * len(d_data)
    device_models = [None] * len(d_data)
    residuals_edges = [None] * len(e_groups)
    edge_models = [None] * len(e_groups)

    device_embeddings = [None] * len(d_data)
    edge_embeddings = [None] * len(e_groups)

    num_epochs = config["epoch"]
    y_true_per_epoch = []

    # Initialize history object before training/evaluation loop
    history = {
        "device_accs_per_epoch": [],
        "edge_accs_per_epoch": [],
        "global_accs": [],

        "device_means": [],
        "device_stds": [],
        "edge_means": [],
        "edge_stds": [],

        "device_vs_global": [],
        "edge_vs_global": [],

        "y_true_per_epoch": [],
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
            y_true_per_edge, global_residuals,edge_sample_slices = forward_pass(
            d_data, e_groups, le, n_classes,
            X_edges_finetune,y_edges_finetune,
            residuals_devices=residuals_devices,
            device_models=device_models
        )

        """
        device_accs_last = plot_device_accuracies(device_models, X_tests, y_tests, le, num_classes, combine_mode="last")
        device_accs_avg = plot_device_accuracies(device_models, X_tests, y_tests, le, num_classes, combine_mode="average")
        device_accs_vote = plot_device_accuracies(device_models, X_tests, y_tests, le, num_classes,
                                                  combine_mode="hard_vote")
        device_accs_weighted = plot_device_accuracies(device_models, X_tests, y_tests, le=le,
                                                      num_classes=num_classes,
                                                      combine_mode="weighted", device_val_scores=device_val_scores)
        """

        #y_true_per_epoch.append(y_global_true)
        """
       # -----------------------------
       # 3. Backward Pass using global residuals
       # -----------------------------
        """
        updated_edge_models, updated_device_models, updated_edge_preds_stacked = backward_pass(
            edge_models=edge_models,
            device_models=device_models,
            edge_embeddings=edge_embeddings,
            device_embeddings=device_embeddings,
            y_true_per_edge=y_true_per_edge,
            edge_sample_slices=edge_sample_slices,
            global_pred_matrix=y_global_pred,
            n_classes=2,
            use_classification=True,
            verbose=True,
            le=le
        )
        # -----------------------------
        # 2. Compute Multi-Level Accuracy

        # -----------------------------

        metrics_test = evaluate_multilevel_performance(
            device_models=updated_device_models,
            edge_models=updated_edge_models,
            edge_groups=edge_groups,
            X_tests=X_tests,
            y_tests=y_tests,
            le=le,
            num_classes=num_classes,
            combine_mode="weighted",
            top_k=3,
        )
        """
        metrics_test = compute_multilevel_accuracy(
            device_models=device_models,
            edge_models=edge_models,
            edge_groups=e_groups,
            y_test=y_tests,
            X_test=X_tests,
            y_global_pred_test=y_global_pred,  # from forward pass
            num_classes=n_classes,
            verbose=True
        )
        """


        # Update history for plotting
        metrics = metrics_test["metrics"]

        history["device_accs_per_epoch"].append(metrics["device"]["acc"])
        history["edge_accs_per_epoch"].append(metrics["edge"]["acc"])
        history["global_accs"].append(metrics["global"]["acc"])

        # Mean/std across devices and edges for reference
        history["device_means"].append(np.mean(metrics["device"]["acc"]))
        history["device_stds"].append(np.std(metrics["device"]["acc"]))
        history["edge_means"].append(np.mean(metrics["edge"]["acc"]))
        history["edge_stds"].append(np.std(metrics["edge"]["acc"]))

        # Differences per device/edge vs global accuracy
        global_acc = metrics["global"]["acc"]

        history["device_vs_global"].append([
            acc - global_acc for acc in metrics["device"]["acc"]
        ])
        history["edge_vs_global"].append([
            acc - global_acc for acc in metrics["edge"]["acc"]
        ])

        # Store y_true per epoch if available
        history["y_true_per_epoch"].append(y_tests)

    return history
        


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


def multi_class_brier(y_true, y_prob):
    """
    Compute multi-class Brier score.
    """
    n_classes = y_prob.shape[1]
    y_one_hot = np.zeros_like(y_prob)
    y_one_hot[np.arange(len(y_true)), y_true] = 1
    return np.mean(np.sum((y_prob - y_one_hot) ** 2, axis=1))


def top_k_accuracy(y_true, y_prob, k=3):
    """
    Compute top-k accuracy for multi-class predictions.
    """
    topk = np.argsort(y_prob, axis=1)[:, -k:]
    correct = np.any(topk == y_true[:, None], axis=1)
    return np.mean(correct)

def plot_device_accuracies(
        d_models,
        X_tests, y_tests,
        le,
        num_classes,
        combine_mode="last",
        device_val_scores=None,
        top_k=3
):
    """
    Compute and plot Accuracy, Log Loss, Brier Score, and Top-K Accuracy per device.
    """
    accuracies, log_losses, brier_scores, topk_accuracies = [], [], [], []

    print("\n================ Device Metrics ================\n")

    for idx, models_per_device in enumerate(d_models):

        if not models_per_device:
            print(f"Device {idx}: ❌ No trained models found.")
            accuracies.append(0.0)
            log_losses.append(np.nan)
            brier_scores.append(np.nan)
            topk_accuracies.append(np.nan)
            continue

        X_test = np.atleast_2d(X_tests[idx])
        y_test = np.atleast_1d(y_tests[idx])
        y_true = le.transform(y_test)

        # Stack predictions
        preds_list = [predict_proba_fixed(m, X_test, num_classes) for m in models_per_device]

        # Compute weights for "weighted" mode
        if combine_mode == "weighted":
            if device_val_scores and idx < len(device_val_scores):
                w = np.array(device_val_scores[idx])
            else:
                w = np.ones(len(preds_list))
            w = w / w.sum()
        else:
            w = None

        # Combine predictions
        if combine_mode == "last":
            y_pred_probs = preds_list[-1]
            y_pred = np.argmax(y_pred_probs, axis=1)
        elif combine_mode == "average":
            y_pred_probs = np.mean(preds_list, axis=0)
            y_pred = np.argmax(y_pred_probs, axis=1)
        elif combine_mode == "weighted":
            y_pred_probs = np.zeros_like(preds_list[0])
            for pi, wi in zip(preds_list, w):
                y_pred_probs += wi * pi
            y_pred = np.argmax(y_pred_probs, axis=1)
        elif combine_mode == "hard_vote":
            votes = np.array([np.argmax(p, axis=1) for p in preds_list])
            y_pred = np.array([np.bincount(votes[:, i], minlength=num_classes).argmax()
                               for i in range(votes.shape[1])])
            y_pred_probs = np.zeros((votes.shape[1], num_classes))
            for c in range(num_classes):
                y_pred_probs[:, c] = np.mean(votes == c, axis=0)
        elif combine_mode == "stacked":
            X_stack = np.hstack(preds_list)
            meta = LogisticRegression(max_iter=500)
            meta.fit(X_stack, y_true)
            y_pred = meta.predict(X_stack)
            y_pred_probs = meta.predict_proba(X_stack)
        else:
            raise ValueError(f"Unknown combine_mode: {combine_mode}")

        # Compute metrics
        acc = accuracy_score(y_true, y_pred)
        try:
            ll = log_loss(y_true, y_pred_probs)
        except ValueError:
            ll = np.nan
        brier = multi_class_brier(y_true, y_pred_probs)
        topk = top_k_accuracy(y_true, y_pred_probs, k=top_k)

        accuracies.append(acc)
        log_losses.append(ll)
        brier_scores.append(brier)
        topk_accuracies.append(topk)

        print(f"✅ Device {idx}: Accuracy={acc:.4f}, LogLoss={ll:.4f}, "
              f"Brier={brier:.4f}, Top-{top_k} Acc={topk:.4f} ({combine_mode})")

    # Plot all metrics
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.bar(range(len(accuracies)), accuracies, color='skyblue', label='Accuracy')
    ax1.set_xlabel("Device Index")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', linestyle='--', alpha=0.6)

    mean_acc = np.mean(accuracies)
    ax1.axhline(mean_acc, color='red', linestyle='--', label=f"Mean Accuracy = {mean_acc:.3f}")

    ax2 = ax1.twinx()
    safe_log_losses = [x for x in log_losses if np.isfinite(x)]
    safe_brier_scores = [x for x in brier_scores if np.isfinite(x)]
    ax2.plot(range(len(log_losses)), log_losses, color='orange', marker='o', label='Log Loss')
    ax2.plot(range(len(brier_scores)), brier_scores, color='green', marker='s', label='Brier Score')
    ax2.plot(range(len(topk_accuracies)), topk_accuracies, color='purple', marker='^', label=f'Top-{top_k} Acc')
    max_log = max(safe_log_losses) if safe_log_losses else 0
    max_brier = max(safe_brier_scores) if safe_brier_scores else 0
    ax2.set_ylim(0, max(max_log, max_brier, 1.0))
    ax2.set_ylabel("Other Metrics")

    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    plt.title(f"Per-Device Metrics ({combine_mode})")
    plt.show()

    print(f"\nOverall Mean Accuracy: {mean_acc:.4f}")
    print(f"Overall Mean Log Loss: {np.nanmean(log_losses):.4f}")
    print(f"Overall Mean Brier Score: {np.nanmean(brier_scores):.4f}")
    print(f"Overall Mean Top-{top_k} Accuracy: {np.nanmean(topk_accuracies):.4f}\n")

    return accuracies, log_losses, brier_scores, topk_accuracies

def plot_hpfl_all(metrics_test, save_root_dir="hdpftl_plot_outputs"):
    """
    Generate Hierarchical PFL plots from structured metrics dictionary.
    Includes per-epoch stacked contributions, device/edge vs global,
    overall accuracy trends, and per-device/edge heatmaps.

    Args:
        metrics_test (dict): Output metrics dictionary per epoch.
        save_root_dir (str): Base directory to save plots.
    """

    # -----------------------------
    # Create dated folder automatically
    # -----------------------------
    today_str = datetime.now().strftime("%Y-%m-%d")
    save_dir = os.path.join(save_root_dir, today_str)
    os.makedirs(save_dir, exist_ok=True)

    # -----------------------------
    # Extract per-epoch values
    # -----------------------------
    device_accs = metrics_test["device_accs_per_epoch"]  # list of lists
    edge_accs = metrics_test["edge_accs_per_epoch"]      # list of lists
    global_accs = metrics_test["global_accs"]            # list of scalars
    device_vs_global = metrics_test["device_vs_global"]  # list of lists
    edge_vs_global = metrics_test["edge_vs_global"]      # list of lists

    num_epochs = len(global_accs)
    colors = {"Device": "skyblue", "Edge": "orange", "Global": "green"}

    # -----------------------------
    # 1. Per-epoch stacked contributions & comparisons
    # -----------------------------
    for epoch_idx in range(num_epochs):
        mean_device = np.mean(device_accs[epoch_idx]) if device_accs[epoch_idx] else 0
        mean_edge = np.mean(edge_accs[epoch_idx]) if edge_accs[epoch_idx] else 0
        global_acc = global_accs[epoch_idx]

        # Contribution bars
        contributions = [
            mean_device,
            max(mean_edge - mean_device, 0),
            max(global_acc - mean_edge, 0),
        ]
        layers = ["Device", "Edge", "Global"]

        plt.figure(figsize=(6, 4))
        bars = plt.bar(layers, contributions, color=[colors[l] for l in layers])
        plt.ylim(0, 1)
        plt.ylabel("Contribution to Final Accuracy")
        plt.title(f"Epoch {epoch_idx+1} Layer Contributions")
        for bar, val in zip(bars, contributions):
            plt.text(bar.get_x() + bar.get_width()/2, float(val)+0.02, f"{val:.3f}",
                     ha="center", va="bottom")
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"epoch_contribution_{epoch_idx+1}.png"))
        plt.close()
        plt.clf()  # clears the current figure
        plt.cla()  # clears current axes

        # Device vs Global & Edge vs Global
        mean_dev_vs_glob = np.mean(device_vs_global[epoch_idx]) if device_vs_global[epoch_idx] else 0
        mean_edge_vs_glob = np.mean(edge_vs_global[epoch_idx]) if edge_vs_global[epoch_idx] else 0

        plt.figure(figsize=(6, 4))
        plt.bar(["Device vs Global", "Edge vs Global"],
                [mean_dev_vs_glob, mean_edge_vs_glob],
                color=[colors["Device"], colors["Edge"]])
        plt.ylim(0, 1)
        plt.ylabel("Accuracy compared to Global")
        plt.title(f"Epoch {epoch_idx+1} Layer vs Global Accuracy")
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"epoch_vs_global_{epoch_idx+1}.png"))
        plt.close()
        plt.clf()  # clears the current figure
        plt.cla()  # clears current axes
        # -----------------------------
        # 1b. Device-level heatmap
        # -----------------------------
        if device_accs[epoch_idx]:
            plt.figure(figsize=(max(6, len(device_accs[epoch_idx])*0.5), 4))
            sns.heatmap(np.array([device_accs[epoch_idx]]), annot=True, cmap="Blues",
                        cbar=True, vmin=0, vmax=1)
            plt.xlabel("Device Index")
            plt.ylabel("Epoch")
            plt.title(f"Epoch {epoch_idx+1} Device Accuracies")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"epoch_device_heatmap_{epoch_idx+1}.png"))
            plt.close()
            plt.clf()  # clears the current figure
            plt.cla()  # clears current axes
        # -----------------------------
        # 1c. Edge-level heatmap
        # -----------------------------
        if edge_accs[epoch_idx]:
            plt.figure(figsize=(max(6, len(edge_accs[epoch_idx])*0.5), 4))
            sns.heatmap(np.array([edge_accs[epoch_idx]]), annot=True, cmap="Oranges",
                        cbar=True, vmin=0, vmax=1)
            plt.xlabel("Edge Index")
            plt.ylabel("Epoch")
            plt.title(f"Epoch {epoch_idx+1} Edge Accuracies")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"epoch_edge_heatmap_{epoch_idx+1}.png"))
            plt.close()
            plt.clf()  # clears the current figure
            plt.cla()  # clears current axes

    # -----------------------------
    # 2. Accuracy trends across epochs
    # -----------------------------
    mean_device_accs = [np.mean(d) if d else 0 for d in device_accs]
    mean_edge_accs = [np.mean(e) if e else 0 for e in edge_accs]

    plt.figure(figsize=(10, 6))
    plt.plot(mean_device_accs, label="Device Accuracy", marker="o", color=colors["Device"])
    plt.plot(mean_edge_accs, label="Edge Accuracy", marker="s", color=colors["Edge"])
    plt.plot(global_accs, label="Global Accuracy", marker="^", color=colors["Global"])
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
    plt.clf()  # clears the current figure
    plt.cla()  # clears current axes

    # Device/Edge vs Global trends
    mean_device_vs_global = [np.mean(d) if d else 0 for d in device_vs_global]
    mean_edge_vs_global = [np.mean(e) if e else 0 for e in edge_vs_global]

    plt.figure(figsize=(10, 6))
    plt.plot(mean_device_vs_global, label="Device vs Global", marker="o", color=colors["Device"])
    plt.plot(mean_edge_vs_global, label="Edge vs Global", marker="s", color=colors["Edge"])
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
    plt.clf()  # clears the current figure
    plt.cla()  # clears current axes

    # -----------------------------
    # 2b. Hierarchical Contribution Evolution (stacked)
    # -----------------------------
    # Calculate incremental gains for each layer
    edge_gain = np.maximum(np.array(mean_edge_accs) - np.array(mean_device_accs), 0)
    global_gain = np.maximum(np.array(global_accs) - np.array(mean_edge_accs), 0)
    base_device = np.array(mean_device_accs)

    epochs = np.arange(1, num_epochs + 1)

    plt.figure(figsize=(10, 6))
    plt.stackplot(
        epochs,
        base_device,
        edge_gain,
        global_gain,
        labels=["Device Base", "Edge Gain", "Global Gain"],
        colors=[colors["Device"], colors["Edge"], colors["Global"]],
        alpha=0.8,
    )
    plt.title("Hierarchical Contribution Evolution Across Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Cumulative Accuracy")
    plt.xticks(epochs, labels=[f"Epoch {i}" for i in epochs])
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "hierarchical_contribution_evolution.png"))
    plt.show()
    plt.close()
    plt.clf()
    plt.cla()

    # -----------------------------
    # 3. Aggregate heatmaps across all epochs
    # -----------------------------

    # Device accuracy heatmap across epochs
    device_matrix = np.array([d if d else [0] * len(device_accs[0]) for d in device_accs])
    plt.figure(figsize=(max(8, device_matrix.shape[1] * 0.5), max(6, device_matrix.shape[0] * 0.5)))
    sns.heatmap(device_matrix, annot=True, cmap="Blues", cbar=True, vmin=0, vmax=1)
    plt.xlabel("Device Index")
    plt.ylabel("Epoch")
    plt.title("Device Accuracies Across Epochs")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "all_epochs_device_heatmap.png"))
    plt.close()
    plt.clf()  # clears the current figure
    plt.cla()  # clears current axes

    # Edge accuracy heatmap across epochs
    edge_matrix = np.array([e if e else [0] * len(edge_accs[0]) for e in edge_accs])
    plt.figure(figsize=(max(8, edge_matrix.shape[1] * 0.5), max(6, edge_matrix.shape[0] * 0.5)))
    sns.heatmap(edge_matrix, annot=True, cmap="Oranges", cbar=True, vmin=0, vmax=1)
    plt.xlabel("Edge Index")
    plt.ylabel("Epoch")
    plt.title("Edge Accuracies Across Epochs")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "all_epochs_edge_heatmap.png"))
    plt.close()
    plt.clf()  # clears the current figure
    plt.cla()  # clears current axes

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
    """, X_pretrain, y_pretrain, X_finetune, y_finetune, X_test, y_test"""
    X_final, y_final = preprocess_data_safe(
        log_path_str, folder_path, scaler_type='minmax'
    )


    # 3. Partition fine-tune data for devices & edges (non-IID)
    devices_data, edge_groups, edge_finetune_data = dirichlet_partition_for_devices_edges_non_iid(
        X_final, y_final,
        num_devices=config["n_device"],
        n_edges=config["n_edges"],
        alpha=config["alpha"],
        seed=config["random_seed"]
    )

    # 2. Encode labels
    le = LabelEncoder()
    y_all = np.concatenate([y for _, _, y, _, _, _, _, _ in devices_data])
    le.fit(y_all)
    num_classes = len(np.unique(y_final))

    # 4. Train HPFL model with accuracy tracking

    history = hpfl_train_with_accuracy(d_data=devices_data, e_groups=edge_groups, edge_finetune_data = edge_finetune_data,
                                           le=le, n_classes=num_classes, verbose=True)
    plot_hpfl_all(history)


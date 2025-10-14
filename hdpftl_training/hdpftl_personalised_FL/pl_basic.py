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
import gc
import logging
import os
import random
import warnings
from datetime import datetime
from typing import List, Tuple, Union
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymc import logit
from scipy import sparse
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import joblib
from lightgbm import early_stopping, LGBMClassifier, LGBMRegressor, log_evaluation
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time
import psutil
from concurrent.futures import ThreadPoolExecutor
import hdpftl_utility.log as log_util

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
    "n_device": 20,
    "device_per_edge": 4,
    "epoch": 5,
    "num_boost_round": 5,
    "num_iterations": 100,
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
    "learning_rate": 0.01,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "early_stopping_rounds": 20,
    "class_weight": "balanced",
    "lambda_l1": 1.0,
    "lambda_l2": 1.0,
    "device": "cpu",
    "eps_threshold": 1e-4
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
    best_params=None,
    init_model=None,
    prev_best_logloss=None,
    verbose=-1,
    **kwargs
):
    """
    Train a LightGBM classifier with validation, continuation, and logloss control.

    Args:
        X_train, y_train : training data
        X_valid, y_valid : validation data
        best_params : tuned hyperparameters
        init_model : existing model for continued training
        prev_best_logloss : previous best logloss (across edges)
        verbose : verbosity level

    Returns:
        model : trained model
        best_logloss : best validation logloss
    """

    # --- Convert safely ---
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_valid = np.array(X_valid) if X_valid is not None else None
    y_valid = np.array(y_valid) if y_valid is not None else None

    if y_train.ndim > 1 and y_train.shape[1] > 1:
        y_train = np.argmax(y_train, axis=1)
    if y_valid is not None and y_valid.ndim > 1 and y_valid.shape[1] > 1:
        y_valid = np.argmax(y_valid, axis=1)

    # --- Determine objective ---
    num_classes = len(np.unique(y_train))
    objective = "binary" if num_classes == 2 else "multiclass"

    # --- Base static parameters (for random search) ---
    base_params = {
        "boosting_type": config.get("boosting", "gbdt"),
        "objective": objective,
        "num_class": num_classes if num_classes > 2 else None,
        "n_estimators": config.get("n_estimators", 5000),
        "learning_rate": config.get("learning_rate", 0.01),
        "num_iterations": config.get("num_iterations", 100),
        "num_leaves": config.get("num_leaves", 31),
        "max_depth": config.get("max_depth", -1),
        "min_child_samples": config.get("min_child_samples", 20),
        "class_weight": config.get("class_weight", None),
        "lambda_l1": config.get("lambda_l1", 0.0),
        "lambda_l2": config.get("lambda_l2", 0.0),
        "feature_fraction": config.get("feature_fraction", 1.0),
        "random_state": config.get("random_seed", 42),
        "device": config.get("device", "cpu"),
        "max_cores": 2,
        "verbose": verbose,
    }

    # --- Merge tuned parameters if available ---
    if best_params is not None:
        model_params = {**base_params, **best_params}
    else:
        model_params = base_params

    # --- Initialize model ---
    model = LGBMClassifier(**model_params)

    # --- Validation setup ---
    fit_kwargs = {}
    if X_valid is not None and y_valid is not None and config.get("early_stopping_rounds", 0):
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
                early_stopping(config.get("early_stopping_rounds", 100)),
                log_evaluation(10)
            ]
        })

    # --- Continue from previous model ---
    if init_model is not None:
        fit_kwargs["init_model"] = init_model

    # --- Train ---
    model.fit(X_train, y_train, **fit_kwargs)

    # --- Compute validation logloss ---
    if X_valid is not None and y_valid is not None:
        y_pred = model.predict_proba(X_valid)
        current_logloss = log_loss(y_valid, y_pred)
    else:
        current_logloss = None

    # --- Logloss control across edges ---
    if prev_best_logloss is not None and current_logloss is not None and current_logloss > prev_best_logloss:
        # Revert to previous model
        if isinstance(init_model, tuple):
            prev_model = init_model[0]
        else:
            prev_model = init_model
        model_to_return = prev_model
        log_util.safe_log(
            f"⚠️ Logloss worsened ({current_logloss:.4f} > {prev_best_logloss:.4f}), reverting to previous model")
    else:
        model_to_return = model
        if current_logloss is not None and prev_best_logloss is not None:
            log_util.safe_log(f"✅ Improved logloss: {current_logloss:.4f} (prev: {prev_best_logloss:.4f})")

    return model_to_return, (current_logloss if current_logloss is not None else prev_best_logloss)


# ============================================================
# Predict probabilities safely, filling missing classes
# ============================================================

def predict_proba_fixed(model, X, n_classes):
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
    full = np.zeros((pred.shape[0], n_classes))
    model_classes = np.asarray(getattr(model, "classes_", np.arange(pred.shape[1]))).astype(int)

    for i, cls in enumerate(model_classes):
        if 0 <= cls < n_classes:
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

def device_layer_boosting(d_data, d_residuals, d_models, le, n_classes, best_params, use_true_labels=False):

    log_util.safe_log("\n" + "*" * 60)
    log_util.safe_log("*" + " " * 28 + "STARTING DEVICE LAYER " + " " * 28 + "*")
    log_util.safe_log("*" * 60 + "\n")

    device_embeddings = [None] * len(d_data)
    device_val_scores = [None] * len(d_data)

    eps_residual = 1e-6  # stability epsilon

    def process_device(idx, dev_tuple):
        X_train, _, y_train, _, _, _, X_device_finetune, y_device_finetune = dev_tuple
        n_samples = X_train.shape[0]
        prev_y_pseudo = None

        # --- Initialize residuals ---
        if d_residuals[idx] is None:
            residual = np.zeros((n_samples, n_classes), dtype=np.float32)
            y_encoded = le.transform(y_train)
            residual[np.arange(n_samples), y_encoded] = 1.0
        else:
            residual = d_residuals[idx].astype(np.float32).copy()

        models_per_device = d_models[idx] if d_models[idx] else []

        boosting_rounds = best_params.get("num_boost_round", 100)

        # --- Track monotonic logloss per device ---
        prev_best_logloss = None

        # --- Sequential boosting loop (per device) ---
        for t in range(boosting_rounds):
            y_pseudo = le.transform(y_train)
            log_util.safe_log(f"[Boosting round {t+1}/{boosting_rounds}] devices={idx}/{num_devices}")

            if len(np.unique(y_pseudo)) < 2:
                log_util.safe_log(f"Device {idx}, round {t}: only single class left, stopping.")
                break

            if np.sum(np.abs(residual)) < 1e-5:
                log_util.safe_log(f"Device {idx}, round {t}: residuals below threshold, stopping.")
                break

            if prev_y_pseudo is not None:
                changes = np.mean(prev_y_pseudo != y_pseudo)
                if changes < config["eps_threshold"]:
                    log_util.safe_log(f"Device {idx}, round {t}: labels stabilized, stopping.")
                    break
            prev_y_pseudo = y_pseudo.copy()

            init_model = models_per_device[-1] if models_per_device else None

            model, current_logloss = train_lightgbm(
                X_train, y_pseudo,
                X_valid=X_device_finetune[:, :X_train.shape[1]] if y_device_finetune is not None else None,
                y_valid=y_device_finetune,
                best_params=best_params,
                init_model=init_model,
                prev_best_logloss=prev_best_logloss,
                verbose=-1
            )

            # Only keep model if logloss improved
            if prev_best_logloss is None or (current_logloss is not None and current_logloss <= prev_best_logloss):
                prev_best_logloss = current_logloss
                if models_per_device is None:
                    models_per_device = [model]
                else:
                    models_per_device.append(model)

            pred_proba = predict_proba_fixed(model, X_train, n_classes).astype(np.float32)
            residual -= pred_proba
            residual = np.clip(residual, -1 + eps_residual, 1 - eps_residual)
            del pred_proba
            gc.collect()

        # --- True-label correction ---
        if y_train is not None and len(models_per_device) > 0:
            y_true_enc = le.transform(y_train)
            y_onehot = np.zeros((n_samples, n_classes), dtype=np.float32)
            y_onehot[np.arange(n_samples), y_true_enc] = 1.0

            y_pred_proba_total = sum(
                predict_proba_fixed(m, X_train, n_classes).astype(np.float32)
                for m in models_per_device
            )
            residual = np.clip(y_onehot - y_pred_proba_total, -1 + eps_residual, 1 - eps_residual)
            log_util.safe_log(f"Device {idx}: residual norm after true-label correction={np.linalg.norm(residual):.4f}")

            del y_onehot, y_pred_proba_total, y_true_enc
            gc.collect()

        # --- Device embeddings ---
        leaf_indices_list = [get_leaf_indices(mdl, X_train) for mdl in models_per_device]
        if leaf_indices_list:
            leaf_indices_concat = np.hstack(leaf_indices_list)
            leaf_embeddings = np.zeros(
                (n_samples, np.max(leaf_indices_concat) + 1),
                dtype=np.float32
            )
            leaf_embeddings[np.arange(n_samples)[:, None], leaf_indices_concat] = 1.0
            device_embeddings[idx] = leaf_embeddings
            del leaf_indices_list, leaf_indices_concat
        else:
            device_embeddings[idx] = np.zeros((n_samples, 1), dtype=np.float32)
        gc.collect()

        # --- Validation weights ---
        if X_device_finetune is not None and y_device_finetune is not None and models_per_device:
            y_val_encoded = le.transform(np.atleast_1d(y_device_finetune))
            scores_per_model = []
            for mdl in models_per_device:
                X_val = X_device_finetune[:, :X_train.shape[1]]
                y_pred_proba = predict_proba_fixed(mdl, X_val, n_classes).astype(np.float32)
                try:
                    loss = log_loss(y_val_encoded, y_pred_proba)
                except ValueError:
                    loss = 1.0
                scores_per_model.append(1.0 / (loss + 1e-7))
                del y_pred_proba, X_val
                gc.collect()

            scores_per_model = np.array(scores_per_model, dtype=np.float32)
            scores_per_model /= scores_per_model.sum() if scores_per_model.sum() > 0 else 1.0
            device_val_scores[idx] = scores_per_model
            del y_val_encoded, scores_per_model
        else:
            device_val_scores[idx] = (
                np.ones(len(models_per_device), dtype=np.float32) / len(models_per_device)
                if models_per_device else np.array([1.0], dtype=np.float32)
            )

        d_residuals[idx] = residual
        d_models[idx] = models_per_device

        del X_train, y_train, X_device_finetune, y_device_finetune, models_per_device, residual
        gc.collect()

    # --- Run all devices in parallel ---
    num_devices = len(d_data)
    with ThreadPoolExecutor(max_workers=num_devices) as executor:
        executor.map(lambda args: process_device(*args), enumerate(d_data))

    # --- Aggregate weights ---
    device_weights = np.array([
        np.mean(w) if isinstance(w, np.ndarray) else float(w)
        for w in device_val_scores
    ], dtype=np.float32)

    gc.collect()
    return d_residuals, d_models, device_embeddings, device_weights


def edge_layer_boosting(
        e_groups, d_embeddings, d_residuals,
        n_classes, le=None, best_param_edge=None,
        X_ftune=None, y_ftune=None, device_weights=None,
        chunk_size=5000, n_random_trials=5
):

    log_util.safe_log("""
    ************************************************************
    *                  STARTING EDGE LAYER                     *
    *   Devices -> Edge-level boosted ensemble -> Edge output  *
    ************************************************************
    """)

    def process_edge_data(d_embeddings, d_residuals, device_weights, edge_devices, eps_residual, n_classes):
        """Internal helper: gather embeddings + weighted residuals for an edge."""
        embeddings_list = [d_embeddings[i] for i in edge_devices]
        residual_list = [
            (d_residuals[i].astype(np.float32) if d_residuals[i] is not None else
             np.zeros((emb.shape[0], n_classes), dtype=np.float32))
            for emb, i in zip(embeddings_list, edge_devices)
        ]

        # Device weights
        if device_weights is None:
            weights = np.ones(len(edge_devices), dtype=float)
        else:
            weights = np.array([device_weights[i] for i in edge_devices], dtype=float)
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)

        # Sparse/dense padding & stacking
        any_sparse = any(sparse.issparse(e) for e in embeddings_list)
        max_cols = max(int(e.shape[1]) for e in embeddings_list)
        if any_sparse:
            csr_list = [e.tocsr() if sparse.issparse(e) else sparse.csr_matrix(e) for e in embeddings_list]
            padded_embeddings = []
            for e_csr in csr_list:
                n_rows, n_cols = e_csr.shape
                if n_cols < max_cols:
                    pad_block = sparse.csr_matrix((n_rows, max_cols - n_cols), dtype=np.float32)
                    e_fixed = sparse.hstack([e_csr.astype(np.float32), pad_block], format="csr")
                else:
                    e_fixed = e_csr.astype(np.float32)[:, :max_cols].tocsr()
                padded_embeddings.append(e_fixed)
            X_edge = sparse.vstack(padded_embeddings, format="csr")
            del csr_list
            gc.collect()
        else:
            def _safe_pad_dense(a, width):
                a = np.asarray(a, dtype=np.float32)
                if a.shape[1] < width:
                    a = np.pad(a, ((0, 0), (0, width - a.shape[1])), mode="constant")
                elif a.shape[1] > width:
                    a = a[:, :width]
                return a

            padded_embeddings = [_safe_pad_dense(e, max_cols) for e in embeddings_list]
            X_edge = np.vstack(padded_embeddings).astype(np.float32)
            del padded_embeddings
            gc.collect()

        # Align residuals -> weighted stacking
        weighted_rows = []
        for emb, r, w in zip(embeddings_list, residual_list, weights):
            n_rows = int(emb.shape[0])
            r = np.asarray(r, dtype=np.float32)
            if r.shape[0] < n_rows:
                r = np.vstack([r, np.zeros((n_rows - r.shape[0], r.shape[1]), dtype=np.float32)])
            elif r.shape[0] > n_rows:
                r = r[:n_rows]
            if r.shape[1] < n_classes:
                r = np.pad(r, ((0, 0), (0, n_classes - r.shape[1])), mode="constant")
            elif r.shape[1] > n_classes:
                r = r[:, :n_classes]
            weighted_rows.append(r * w)
        residual_edge = np.vstack(weighted_rows).astype(np.float32)
        residual_edge = np.clip(residual_edge, -1 + 1e-6, 1 - 1e-6)
        del embeddings_list, residual_list, weighted_rows
        gc.collect()
        return residual_edge, X_edge

    # -----------------------------
    # Initialize storage
    # -----------------------------
    e_models, e_residuals, edge_embeddings_list = [], [], []
    edge_outputs, global_pred_blocks, edge_sample_slices = [], [], {}
    global_offset = 0
    eps_residual = 1e-6

    # -----------------------------
    # Process each edge
    # -----------------------------
    for edge_idx, edge_devices in enumerate(e_groups):
        prev_y_pseudo_edge = None
        X_valid_edge = X_ftune[edge_idx] if X_ftune is not None else None
        y_valid_edge = y_ftune[edge_idx] if y_ftune is not None else None

        if len(edge_devices) == 0:
            log_util.safe_log(f"Edge {edge_idx}: no devices, skipping.")
            e_models.append([])
            e_residuals.append(None)
            edge_embeddings_list.append(None)
            edge_outputs.append(None)
            continue

        residual_edge, X_edge = process_edge_data(
            d_embeddings, d_residuals, device_weights, edge_devices, eps_residual, n_classes
        )

        models_per_edge = []
        best_logloss_edge = None  # Track best logloss for monotonic control

        # -----------------------------
        # Hyperparameter tuning
        # -----------------------------
        if (
            X_valid_edge is not None and y_valid_edge is not None
            and residual_edge is not None
        ):
            best_param_edge = random_search_boosting_params(
                X_edge,
                np.argmax(residual_edge, axis=1),
                X_valid_edge[:, :X_edge.shape[1]] if not sparse.issparse(X_valid_edge) else X_valid_edge[:, :X_edge.shape[1]],
                y_valid_edge,
                le,
                n_classes,
                n_trials=n_random_trials,
                verbose=False
            )
        else:
            best_param_edge = best_param_edge or {
                "learning_rate": config.get("learning_rate_edge", 0.01),
                "num_boost_round": config.get("edge_boosting_rounds", 5)
            }

        boosting_rounds = best_param_edge.get("num_boost_round", config.get("num_boost_round", 5))
        learning_rate = best_param_edge.get("learning_rate", config.get("learning_rate", 0.01))

        # -----------------------------
        # Sequential boosting loop with monotonic logloss
        # -----------------------------
        for t in range(boosting_rounds):
            log_util.safe_log(f"[Boosting round {t+1}/{boosting_rounds}] edges={edge_idx}/{len(e_groups)} ")

            y_pseudo = np.argmax(residual_edge + 1e-9 * np.random.randn(*residual_edge.shape), axis=1)
            unique_classes = np.unique(y_pseudo)
            if len(unique_classes) < 2 or np.sum(np.abs(residual_edge)) < eps_residual:
                break
            if prev_y_pseudo_edge is not None and np.mean(prev_y_pseudo_edge != y_pseudo) < config.get("eps_threshold", 1e-3):
                break
            prev_y_pseudo_edge = y_pseudo.copy()
            init_model = models_per_edge[-1] if models_per_edge else None
            X_valid_slice = (X_valid_edge[:, :X_edge.shape[1]] if (X_valid_edge is not None and X_valid_edge.ndim == 2)
                             else X_valid_edge)

            # --- Train edge model with logloss check ---
            model, current_logloss = train_lightgbm(
                X_edge, y_pseudo,
                X_valid=X_valid_slice,
                y_valid=y_valid_edge,
                best_params=best_param_edge,
                init_model=init_model,
                prev_best_logloss=best_logloss_edge,
                verbose=-1
            )

            # Only update models_per_edge if logloss improved
            if best_logloss_edge is None or (current_logloss is not None and current_logloss <= best_logloss_edge):
                best_logloss_edge = current_logloss
                if models_per_edge is None:
                    models_per_edge = [model]
                else:
                    models_per_edge.append(model)

            # Chunked predict_proba
            num_rows = X_edge.shape[0]
            if num_rows <= chunk_size:
                pred_proba = predict_proba_fixed(model, X_edge, n_classes).astype(np.float32)
            else:
                parts = []
                for start in range(0, num_rows, chunk_size):
                    end = min(start + chunk_size, num_rows)
                    X_slice = X_edge[start:end] if not sparse.issparse(X_edge) else X_edge[start:end]
                    parts.append(predict_proba_fixed(model, X_slice, n_classes).astype(np.float32))
                pred_proba = np.vstack(parts)
                del parts
                gc.collect()

            residual_edge = residual_edge - learning_rate * pred_proba
            residual_edge = np.clip(residual_edge, -1 + eps_residual, 1 - eps_residual)
            del pred_proba, X_valid_slice
            gc.collect()

        # -----------------------------
        # Store edge results
        # -----------------------------
        e_models.append(models_per_edge)
        e_residuals.append(residual_edge.copy())
        edge_embeddings_list.append(X_edge)

        # Average predictions
        model_preds_sum = np.zeros((X_edge.shape[0], n_classes), dtype=np.float32)
        for m in models_per_edge:
            if X_edge.shape[0] <= chunk_size:
                model_preds_sum += predict_proba_fixed(m, X_edge, n_classes).astype(np.float32)
            else:
                for start in range(0, X_edge.shape[0], chunk_size):
                    end = min(start + chunk_size, X_edge.shape[0])
                    X_slice = X_edge[start:end] if not sparse.issparse(X_edge) else X_edge[start:end]
                    model_preds_sum[start:end] += predict_proba_fixed(m, X_slice, n_classes).astype(np.float32)
        e_pred_avg = (model_preds_sum / max(1, len(models_per_edge))).astype(np.float32)
        edge_outputs.append(e_pred_avg)

        # Global indices per device
        row_start = 0
        for dev_idx in edge_devices:
            n_dev = int(d_embeddings[dev_idx].shape[0])
            edge_sample_slices[dev_idx] = np.arange(global_offset + row_start, global_offset + row_start + n_dev)
            row_start += n_dev
        global_pred_blocks.append(e_pred_avg)
        global_offset += X_edge.shape[0]

        del residual_edge, models_per_edge, e_pred_avg
        gc.collect()

    # -----------------------------
    # Global prediction matrix
    # -----------------------------
    if global_pred_blocks:
        padded_blocks = []
        for block in global_pred_blocks:
            if block.shape[1] < n_classes:
                block = np.hstack([block, np.zeros((block.shape[0], n_classes - block.shape[1]), dtype=np.float32)])
            padded_blocks.append(block.astype(np.float32))
        global_pred_matrix = np.vstack(padded_blocks)
    else:
        global_pred_matrix = np.empty((0, n_classes), dtype=np.float32)

    return edge_outputs, e_models, e_residuals, edge_embeddings_list, global_pred_matrix, edge_sample_slices



def global_layer_bayesian_aggregation(
    e_outputs,
    e_embeddings,
    e_residuals=None,
    y_val=None,
    n_classes=2,
    n_random_samples=30,
    prior_var_range=(0.01, 10.0),
    noise_var_range=(1e-3, 1.0),
    verbose=True,
):
    """
    Global Bayesian Aggregation Layer with memory management.
    Combines edge-level predictions using reliability weighting and Bayesian fusion.
    """

    log_util.safe_log("""
    ************************************************************
    *                  STARTING GLOBAL LAYER                   *
    *  Edges -> Bayesian fusion (weighted) -> Global ensemble  *
    ************************************************************
    """)

    eps = 1e-8

    # --------------------------------------------------------
    # 1. Filter valid edges
    # --------------------------------------------------------
    valid_edges = [
        i for i, (out, emb) in enumerate(zip(e_outputs, e_embeddings))
        if out is not None and emb is not None
    ]
    if not valid_edges:
        raise ValueError("No valid edges found!")

    # --------------------------------------------------------
    # 2. Stack predictions across edges
    # --------------------------------------------------------
    edge_preds = []
    for i in valid_edges:
        out = e_outputs[i]
        if out.shape[1] < n_classes:
            out = np.pad(out, ((0, 0), (0, n_classes - out.shape[1])), mode='constant')
        edge_preds.append(out)

    H_global = np.vstack(edge_preds).astype(np.float32)
    del edge_preds
    gc.collect()

    n_samples_total = H_global.shape[0]

    # --------------------------------------------------------
    # 3. Compute reliability weights
    # --------------------------------------------------------
    if e_residuals is not None:
        weights = []
        for i in valid_edges:
            r = e_residuals[i]
            if r is None:
                weights.append(np.ones(H_global.shape[0]) * eps)
                continue
            if r.shape[1] < n_classes:
                r = np.pad(r, ((0, 0), (0, n_classes - r.shape[1])), mode="constant")
            elif r.shape[1] > n_classes:
                r = r[:, :n_classes]
            per_sample_mse = np.mean(r ** 2, axis=1)
            reliability = np.clip(1.0 / (per_sample_mse + eps), a_min=eps, a_max=1e6)
            weights.append(reliability)
        edge_weights = np.hstack(weights)
        del weights, r, per_sample_mse, reliability
        gc.collect()
    else:
        edge_weights = np.ones(n_samples_total, dtype=float)

    edge_weights /= np.sum(edge_weights)

    # --------------------------------------------------------
    # 4. Generate pseudo-labels (if y_val not provided)
    # --------------------------------------------------------
    y_pseudo_global = np.argmax(H_global, axis=1)
    y_pseudo_onehot = np.zeros((n_samples_total, n_classes), dtype=np.float32)
    y_pseudo_onehot[np.arange(n_samples_total), y_pseudo_global] = 1
    global_residuals = y_pseudo_onehot - H_global
    del y_pseudo_global
    gc.collect()

    # --------------------------------------------------------
    # 5. Create validation subset
    # --------------------------------------------------------
    val_indices = np.random.choice(
        n_samples_total, size=max(10, n_samples_total // 5), replace=False
    )
    H_val = H_global[val_indices]
    y_val_internal = y_val if y_val is not None else y_pseudo_onehot[val_indices]
    del val_indices
    gc.collect()

    # --------------------------------------------------------
    # 6. Random sampling for Bayesian tuning
    # --------------------------------------------------------
    best_score, best_params = -np.inf, None

    for _ in range(n_random_samples):
        prior_var = np.exp(np.random.uniform(np.log(prior_var_range[0]), np.log(prior_var_range[1])))
        noise_var = np.exp(np.random.uniform(np.log(noise_var_range[0]), np.log(noise_var_range[1])))

        precision_prior = 1.0 / prior_var
        precision_noise = 1.0 / noise_var

        alpha = np.sum(H_global * edge_weights[:, None], axis=0)
        theta_global = (precision_noise * alpha) / (precision_prior + precision_noise)

        # Softmax probabilities
        exp_logits = np.exp(theta_global - np.max(theta_global))
        y_pred = exp_logits / np.sum(exp_logits)

        if y_val_internal.ndim == 1:
            log_likelihood = np.mean(np.log(y_pred[y_val_internal % len(y_pred)] + eps))
        else:
            log_likelihood = np.mean(np.sum(y_val_internal * np.log(y_pred + eps), axis=1))

        if log_likelihood > best_score:
            best_score = log_likelihood
            best_params = {"prior_var": prior_var, "noise_var": noise_var}

        del exp_logits, y_pred
        gc.collect()

    if verbose:
        log_util.safe_log(f"\n[Bayesian Tuning] Best params: {best_params} | Validation score: {best_score:.6f}\n")

    # --------------------------------------------------------
    # 7. Final Bayesian fusion using best hyperparameters
    # --------------------------------------------------------
    prior_var, noise_var = best_params["prior_var"], best_params["noise_var"]
    precision_prior, precision_noise = 1.0 / prior_var, 1.0 / noise_var

    alpha = np.sum(H_global * edge_weights[:, None], axis=0)
    beta = np.sum(global_residuals * edge_weights[:, None], axis=0)
    theta_global = (precision_noise * (alpha + beta)) / (precision_prior + precision_noise)

    logits = H_global + beta
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    y_global_pred = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    del logits, exp_logits
    gc.collect()

    theta_out = {
        "alpha": alpha,
        "beta": beta,
        "prior_var": prior_var,
        "noise_var": noise_var,
    }

    if verbose:
        log_util.safe_log(f"Final α: {alpha}")
        log_util.safe_log(f"Final β: {beta}")
        log_util.safe_log(f"Samples: {n_samples_total}")

    return y_global_pred, global_residuals, theta_out

# ============================================================
# Forward pass with Device → Edge → Gossip
# ============================================================

process = psutil.Process()

def get_memory_mb():
    return process.memory_info().rss / 1024**2


def forward_pass(
    devices_data, edge_groups, le, num_classes,
    best_param_device= None,
    best_param_edge = None,
    X_edges_finetune=None, y_edges_finetune=None,
    residuals_devices=None, device_models=None,
    pred_chunk_size=1024,
    track_profile=False
):
    """
    Forward pass for HPFL: device → edge → global aggregation.

    Args:
        devices_data (list): Per-device data tuples.
        edge_groups (list of lists): Device indices per edge.
        le (LabelEncoder): Fitted label encoder.
        num_classes (int): Number of classes.
        best_param_device, best_param_edge (dict, optional): Pre-tuned hyperparameters.
        X_edges_finetune, y_edges_finetune (list, optional): Edge fine-tuning data.
        residuals_devices, device_models (list, optional): Previous residuals/models.
        pred_chunk_size (int, default=1024): Batch size for predictions.
        track_profile (bool, default=False): Record time/memory usage.

    Returns:
        tuple: (
            device_models, edge_models, edge_outputs, theta_global,
            residuals_devices, residuals_edges, y_global_pred,
            device_embeddings, edge_embeddings_list, y_global_true,
            y_true_per_edge, global_residuals, edge_sample_slices, profile
)
    """


    profile = {}  # store time & memory per layer

    # Initialize residuals and device models if None
    if residuals_devices is None:
        residuals_devices = [None] * len(devices_data)
    if device_models is None:
        device_models = [None] * len(devices_data)

    # -----------------------------
    # 1. Device Layer
    # -----------------------------
    t0 = time.time()
    mem0 = get_memory_mb()

    residuals_devices, device_models, device_embeddings, device_weights = device_layer_boosting(
        devices_data, residuals_devices, device_models,
        le, num_classes,best_param_device
    )
    assert device_embeddings is not None, "Device embeddings returned as None!"

    device_embeddings = [emb.astype(np.float32) for emb in device_embeddings]
    residuals_devices = [res.astype(np.float32) if res is not None else None for res in residuals_devices]

    del devices_data
    gc.collect()

    if track_profile:
        profile['Device'] = {
            'time': time.time() - t0,
            'mem_MB': get_memory_mb() - mem0
        }

    # -----------------------------
    # 2. Edge Layer
    # -----------------------------
    t0 = time.time()
    mem0 = get_memory_mb()

    n_samples = sum(e.shape[0] for e in device_embeddings if e is not None)

    edge_outputs, edge_models, residuals_edges, edge_embeddings_list, global_pred_matrix, edge_sample_slices = edge_layer_boosting(
        e_groups=edge_groups,
        d_embeddings=device_embeddings,
        d_residuals=residuals_devices,
        n_classes=num_classes,
        best_param_edge=best_param_edge,
        X_ftune=X_edges_finetune,
        y_ftune=y_edges_finetune,
        device_weights=device_weights
    )

    edge_embeddings_list = [emb.astype(np.float32) if emb is not None else None for emb in edge_embeddings_list]
    residuals_edges = [res.astype(np.float32) if res is not None else None for res in residuals_edges]
    global_pred_matrix = global_pred_matrix.astype(np.float32)

    del X_edges_finetune
    gc.collect()

    if track_profile:
        profile['Edge'] = {
            'time': time.time() - t0,
            'mem_MB': get_memory_mb() - mem0
        }

    # -----------------------------
    # 3. Build Global Ground Truth
    # -----------------------------
    t0 = time.time()
    mem0 = get_memory_mb()

    y_global_true = np.full((n_samples,), -1, dtype=int)
    written_mask = np.zeros(n_samples, dtype=bool)
    y_true_per_edge = []

    for edge_idx, edge_devices in enumerate(edge_groups):
        edge_labels = []
        for dev_idx in edge_devices:
            idxs = edge_sample_slices.get(dev_idx, np.arange(device_embeddings[dev_idx].shape[0]))

            # Safe access to y_edge_finetunes
            if y_edges_finetune is not None and dev_idx < len(y_edges_finetune):
                y_dev = np.array(y_edges_finetune[dev_idx])
                if y_dev.size < idxs.size:
                    y_dev = np.pad(y_dev, (0, idxs.size - y_dev.size), mode="edge")
                elif y_dev.size > idxs.size:
                    y_dev = y_dev[:idxs.size]
            else:
                y_dev = np.full(len(idxs), -1, dtype=int)  # unknown labels

            not_written = ~written_mask[idxs]
            y_global_true[idxs[not_written]] = y_dev[not_written]
            written_mask[idxs[not_written]] = True

            edge_labels.append(y_dev)

        y_true_per_edge.append(np.hstack(edge_labels))

    if track_profile:
        profile['GroundTruth'] = {
            'time': time.time() - t0,
            'mem_MB': get_memory_mb() - mem0
        }

    # -----------------------------
    # 4. Global Layer Aggregation
    # -----------------------------
    t0 = time.time()
    mem0 = get_memory_mb()

    y_global_pred, global_residuals, theta_global = global_layer_bayesian_aggregation(
        e_outputs=edge_outputs,
        e_embeddings=edge_embeddings_list,
        e_residuals=residuals_edges,
        y_val=y_edges_finetune[0] if y_edges_finetune is not None else None,
        n_classes=num_classes,
        n_random_samples=50
    )

    if track_profile:
        profile['Global'] = {
            'time': time.time() - t0,
            'mem_MB': get_memory_mb() - mem0
        }

    # -----------------------------
    # 5. Free temporary arrays
    # -----------------------------
    del y_edges_finetune
    gc.collect()

    common_returns = (
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
    )

    if track_profile:
        return common_returns + (profile,)
    else:
        return common_returns + (None,)


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
    Hierarchical backward feedback with multiclass-safe updates for edge and device models,
    now with explicit memory management.
    """
    if verbose:
        log_util.safe_log("\n" + "*" * 60)
        log_util.safe_log("*" + " " * 20 + "STARTING BACKWARD PASS" + " " * 20 + "*")
        log_util.safe_log("*" * 60 + "\n")

    num_edges = len(edge_models)
    assert len(edge_embeddings) == num_edges, "edge_embeddings and edge_models length mismatch"
    assert len(y_true_per_edge) == num_edges, "y_true_per_edge must align with edge_models"

    updated_edge_preds_list = []
    updated_edge_models = []

    # -----------------------------
    # 1) Update Edge Models
    # -----------------------------
    for ei in range(num_edges):
        model = edge_models[ei]
        X_edge = edge_embeddings[ei]
        y_edge = np.asarray(y_true_per_edge[ei]).ravel()

        if X_edge is None or model is None:
            updated_edge_models.append(None)
            updated_edge_preds_list.append(np.zeros((0, n_classes), dtype=np.float32))
            if verbose:
                log_util.safe_log(f"Edge {ei}: no model or data, skipping.")
            continue

        if le is not None:
            y_edge_labels = encode_labels_safe(le, y_edge, n_classes=n_classes)
        else:
            y_edge_labels = y_edge.astype(int)

        n_est = getattr(model, "n_estimators", 100)
        lr = getattr(model, "learning_rate", 0.1)
        md = getattr(model, "max_depth", None)
        rnd = getattr(model, "random_state", 42)

        if use_classification:
            clf = LGBMClassifier(
                objective='multiclass',
                num_class=n_classes,
                n_estimators=n_est,
                learning_rate=lr,
                max_depth=md,
                random_state=rnd,
                verbosity=-1
            )
            clf.fit(X_edge, y_edge_labels)
            preds = clf.predict_proba(X_edge)
            updated_edge_models.append(clf)
        else:
            y_edge_onehot = np.zeros((X_edge.shape[0], n_classes), dtype=np.float32)
            y_edge_onehot[np.arange(X_edge.shape[0]), y_edge_labels] = 1.0
            if hasattr(model, "predict_proba"):
                preds_init = model.predict_proba(X_edge)
                if preds_init.shape[1] < n_classes:
                    preds_init = np.pad(preds_init, ((0,0),(0,n_classes - preds_init.shape[1])), mode='constant')
                residual_targets = y_edge_onehot - preds_init
            else:
                residual_targets = y_edge_onehot
            reg = LGBMRegressor(n_estimators=n_est, learning_rate=lr, max_depth=md, random_state=rnd)
            reg.fit(X_edge, residual_targets)
            preds = reg.predict(X_edge)
            if preds.ndim == 1:
                preds = preds[:, np.newaxis]
            updated_edge_models.append(reg)
            del y_edge_onehot, residual_targets, preds_init
            gc.collect()

        updated_edge_preds_list.append(preds.astype(np.float32))
        del X_edge, y_edge_labels, y_edge, preds
        gc.collect()

        if verbose:
            log_util.safe_log(f"Edge {ei}: updated model, preds shape {updated_edge_preds_list[-1].shape}")

    # -----------------------------
    # Align and stack edge predictions
    # -----------------------------
    if updated_edge_preds_list:
        max_dim = max(pred.shape[1] for pred in updated_edge_preds_list if pred.size > 0)
        aligned_preds = []
        for pred in updated_edge_preds_list:
            if pred.size == 0:
                aligned_preds.append(np.zeros((0, max_dim), dtype=np.float32))
            elif pred.shape[1] < max_dim:
                pred = np.pad(pred, ((0,0),(0,max_dim - pred.shape[1])), mode='constant')
                aligned_preds.append(pred)
            else:
                aligned_preds.append(pred[:, :max_dim])
        updated_edge_preds_stacked = np.vstack(aligned_preds).astype(np.float32)
        del aligned_preds
        gc.collect()
    else:
        updated_edge_preds_stacked = np.zeros((0, n_classes), dtype=np.float32)

    # -----------------------------
    # 2) Update Device Models
    # -----------------------------
    updated_device_models = []
    try:
        y_global_labels = np.hstack([np.asarray(y).ravel() for y in y_true_per_edge])
    except Exception:
        y_global_labels = None

    for dev_idx, model in enumerate(device_models):
        X_dev = device_embeddings[dev_idx]
        if X_dev is None or model is None:
            updated_device_models.append(None)
            if verbose:
                log_util.safe_log(f"Device {dev_idx}: no model or embeddings, skipping.")
            continue

        if edge_sample_slices is None:
            updated_device_models.append(model)
            continue

        global_idxs = edge_sample_slices.get(dev_idx, None)
        if global_idxs is None or updated_edge_preds_stacked.shape[0] <= global_idxs.max():
            updated_device_models.append(model)
            continue

        preds_for_device = updated_edge_preds_stacked[global_idxs, :]
        if y_global_labels is not None and y_global_labels.size > 0 and global_idxs.max() < y_global_labels.size:
            y_dev_true = y_global_labels[global_idxs]
        else:
            y_dev_true = np.argmax(preds_for_device, axis=1)

        n_est = getattr(model, "n_estimators", 100)
        lr = getattr(model, "learning_rate", 0.1)
        md = getattr(model, "max_depth", None)
        rnd = getattr(model, "random_state", 42)

        if use_classification:
            clf = LGBMClassifier(
                objective='multiclass',
                num_class=n_classes,
                n_estimators=n_est,
                learning_rate=lr,
                max_depth=md,
                random_state=rnd,
                verbosity=-1
            )
            clf.fit(X_dev, y_dev_true)
            updated_device_models.append(clf)
        else:
            y_dev_onehot = np.zeros((preds_for_device.shape[0], n_classes), dtype=np.float32)
            y_dev_onehot[np.arange(preds_for_device.shape[0]), y_dev_true] = 1.0
            residual_targets = y_dev_onehot - preds_for_device
            reg = LGBMRegressor(n_estimators=n_est, learning_rate=lr, max_depth=md, random_state=rnd)
            reg.fit(X_dev, residual_targets)
            updated_device_models.append(reg)
            del y_dev_onehot, residual_targets
            gc.collect()

        del X_dev, preds_for_device, y_dev_true
        gc.collect()

        if verbose:
            log_util.safe_log(f"Device {dev_idx}: updated model.")

    if verbose:
        log_util.safe_log("\nBackward hierarchical feedback completed safely.\n")

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


def random_search_boosting_params(
        X_train,
        y_train,
        X_valid,
        y_valid,
        le,
        n_classes,
        n_trials=10,
        verbose=True
):
    """
    Random search for optimal LightGBM hyperparameters for sequential boosting.
    Incorporates monotonic logloss check to avoid degrading performance.

    Args:
        X_train, y_train : training data
        X_valid, y_valid : validation data for scoring
        le : LabelEncoder for encoding y labels
        n_classes : total number of classes in the problem
        n_trials : number of random trials
        verbose : log_util.safe_log trial info

    Returns:
        best_params : dictionary of best hyperparameters
    """

    best_score = np.inf
    best_params = None

    # All possible class labels (0,1,...,n_classes-1)
    all_classes = np.arange(n_classes)

    y_train_enc = encode_labels_safe(le, y_train, n_classes)
    y_valid_enc = encode_labels_safe(le, y_valid, n_classes)

    prev_best_logloss = np.inf  # monotonic logloss baseline

    for trial in range(n_trials):
        # Sample random hyperparameters
        params = {
            "learning_rate": 10 ** np.random.uniform(-2.0, -0.3),
            "num_leaves": np.random.randint(15, 128),
            "max_depth": np.random.randint(3, 12),
            "min_data_in_leaf": np.random.randint(5, 50),
            "min_gain_to_split": np.random.uniform(0.0, 0.5),
            "feature_fraction": np.random.uniform(0.6, 1.0),
            "bagging_fraction": np.random.uniform(0.6, 1.0),
            "bagging_freq": np.random.randint(1, 10),
            "lambda_l1": np.random.uniform(0.0, 1.0),
            "lambda_l2": np.random.uniform(0.0, 1.0),
            "max_bin": np.random.randint(100, 512),
            "num_boost_round": np.random.randint(50, 200)
        }

        # Train model with sampled hyperparameters
        model, current_logloss = train_lightgbm(
            X_train, y_train_enc,
            X_valid=X_valid, y_valid=y_valid_enc,
            best_params=None,
            **params
        )

        # Evaluate log-loss on validation set
        y_pred = predict_proba_fixed(model, X_valid, n_classes)
        score = log_loss(y_valid_enc, y_pred, labels=all_classes)

        # Only accept trial if it improves or maintains monotonic logloss
        if score <= prev_best_logloss:
            best_score = score
            best_params = params
            prev_best_logloss = score

        # Free memory
        del model, y_pred
        gc.collect()

    if verbose:
        log_util.safe_log(f"Best params selected: {best_params} (val loss={best_score:.4f})")

    return best_params



def encode_labels_safe(le, y, n_classes):
        """Encode labels safely — handles numeric and unseen labels gracefully."""
        y_arr = np.array(y)

        # If numeric (like residual pseudo-labels)
        if np.issubdtype(y_arr.dtype, np.number):
            # Clip or fix invalid labels (-1, n_classes, etc.)
            if np.any(y_arr < 0) or np.any(y_arr >= n_classes):
                y_arr = np.clip(y_arr, 0, n_classes - 1)
            return y_arr.astype(int)

        # Otherwise categorical labels
        if not hasattr(le, "classes_") or len(le.classes_) < n_classes:
            le.fit(np.arange(n_classes))

        try:
            return le.transform(y_arr)
        except ValueError:
            # Handle unseen categories gracefully
            known = set(le.classes_)
            new = set(np.unique(y_arr))
            le.fit(np.array(sorted(known.union(new))))
            return le.transform(y_arr)


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
    log_util.safe_log("DEBUG compute_accuracy:")
    log_util.safe_log(f"  y_true shape: {y_true_np.shape}")
    log_util.safe_log(f"  y_pred shape: {y_pred_np.shape}")
    log_util.safe_log(f"  y_true sample: {y_true_np[:10]}")
    log_util.safe_log(f"  y_pred sample: {y_pred_np[:10]}")

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

def safe_pad(e, max_cols, mode="constant", constant_values=0):
    n_rows, n_cols = e.shape
    if n_cols > max_cols:
        e_fixed = e[:, :max_cols]  # truncate
    elif n_cols < max_cols:
        pad_width = max_cols - n_cols
        e_fixed = np.pad(e, ((0, 0), (0, pad_width)), mode=mode, constant_values=constant_values)
    else:
        e_fixed = e.copy()
    return e_fixed

# ==========================================================
# SAVE FUNCTION
# ==========================================================
def save_hpfl_models(device_models, edge_models, epoch,
                     y_global_pred=None, global_residuals=None, theta_global=None,
                     save_root="trained_models"):
    """
    Save all device and edge models, along with Bayesian global outputs for a given epoch.

    Args:
        device_models (list): List of trained device-level models.
        edge_models (list): List of trained edge-level models.
        epoch (int): Current epoch number (0-indexed).
        y_global_pred (np.ndarray, optional): Global aggregated predictions.
        global_residuals (np.ndarray, optional): Global residual feedback matrix.
        theta_global (np.ndarray, optional): Bayesian parameters or global distribution statistics.
        save_root (str): Base directory for saving model files.
    """
    epoch_dir = os.path.join(save_root, f"epoch_{epoch + 1}")
    os.makedirs(epoch_dir, exist_ok=True)

    # ---- Save Device Models ----
    for idx, model in enumerate(device_models):
        joblib.dump(model, os.path.join(epoch_dir, f"device_model_{idx}.pkl"))

    # ---- Save Edge Models ----
    for idx, model in enumerate(edge_models):
        joblib.dump(model, os.path.join(epoch_dir, f"edge_model_{idx}.pkl"))

    # ---- Save Bayesian Global Outputs (if available) ----
    if y_global_pred is not None:
        np.save(os.path.join(epoch_dir, "y_global_pred.npy"), y_global_pred)
    if global_residuals is not None:
        np.save(os.path.join(epoch_dir, "global_residuals.npy"), global_residuals)
    if theta_global is not None:
        np.save(os.path.join(epoch_dir, "theta_global.npy"), theta_global)

    log_util.safe_log(f"✅ HPFL epoch {epoch + 1} saved → {epoch_dir}")


# ==========================================================
# LOAD FUNCTION
# ==========================================================
def load_hpfl_models(epoch, num_devices, num_edges, save_root="trained_models"):
    """
    Load all device, edge models, and Bayesian global outputs for a given epoch.

    Args:
        epoch (int): Epoch number to load (0-indexed).
        num_devices (int): Number of device models expected.
        num_edges (int): Number of edge models expected.
        save_root (str): Base directory where models were saved.

    Returns:
        tuple:
            (
                loaded_device_models,
                loaded_edge_models,
                y_global_pred,
                global_residuals,
                theta_global
            )
    """
    epoch_dir = os.path.join(save_root, f"epoch_{epoch + 1}")

    if not os.path.exists(epoch_dir):
        raise FileNotFoundError(f"❌ Epoch directory not found: {epoch_dir}")

    # ---- Load Device Models ----
    loaded_device_models = []
    for idx in range(num_devices):
        path = os.path.join(epoch_dir, f"device_model_{idx}.pkl")
        if os.path.exists(path):
            loaded_device_models.append(joblib.load(path))
        else:
            log_util.safe_log(f"⚠️ Missing device model {idx} at epoch {epoch + 1}")

    # ---- Load Edge Models ----
    loaded_edge_models = []
    for idx in range(num_edges):
        path = os.path.join(epoch_dir, f"edge_model_{idx}.pkl")
        if os.path.exists(path):
            loaded_edge_models.append(joblib.load(path))
        else:
            log_util.safe_log(f"⚠️ Missing edge model {idx} at epoch {epoch + 1}")

    # ---- Load Bayesian Global Outputs ----
    def safe_load_npy(file_path):
        return np.load(file_path) if os.path.exists(file_path) else None

    y_global_pred = safe_load_npy(os.path.join(epoch_dir, "y_global_pred.npy"))
    global_residuals = safe_load_npy(os.path.join(epoch_dir, "global_residuals.npy"))
    theta_global = safe_load_npy(os.path.join(epoch_dir, "theta_global.npy"))

    log_util.safe_log(f"✅ HPFL epoch {epoch + 1} loaded ← {epoch_dir}")
    return loaded_device_models, loaded_edge_models, y_global_pred, global_residuals, theta_global


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
        log_util.safe_log(f"Device mean acc: {np.mean(device_accs):.4f}, "
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
    device_weight_mode="weighted",
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
        if isinstance(models_per_device, list):
            preds_list = [predict_proba_fixed(m, X_test, num_classes) for m in models_per_device]
        else:
            preds_list = [predict_proba_fixed(models_per_device, X_test, num_classes)]

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
        log_util.safe_log("Per-device logloss:", log_losses)
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

        # Device weights for edge aggregation
        if device_weight_mode == "samples":
            w_devices = [np.atleast_2d(X_tests[d]).shape[0] for d in edge_devices if d < len(X_tests)]
        else:
            w_devices = [1.0] * len(device_edge_preds)
        w_devices = normalize_weights(w_devices)

        # Safely combine predictions with varying number of samples
        weighted_preds = [wi * dp for wi, dp in zip(w_devices, device_edge_preds)]
        device_mix_pred = np.vstack(weighted_preds)

        if edge_idx < len(edge_models):
            mdl = edge_models[edge_idx]
            edge_pred_accum = predict_proba_fixed(mdl, X_edge, num_classes)
            if calibrate:
                edge_pred_accum = calibrate_probs(y_edge_true, edge_pred_accum)
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
        "device_means": device_accs,
        "device_stds": np.std(device_accs),
        "edge_means": edge_accs,
        "edge_stds": np.std(edge_accs),
        "global_accs": global_acc,
        "device_vs_global": device_accs,
        "edge_vs_global": edge_accs,
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
    HPFL Training Loop with Multi-Level Accuracy Tracking
    """

    # -------------------------
    # 1️⃣ Device-level splits
    # -------------------------
    X_trains, X_tests, y_trains, y_tests = [], [], [], []
    X_device_finetunes, y_device_finetunes = [], []

    for dev_tuple in d_data:
        (
            X_train, X_test, y_train, y_test,
            X_edge_finetune, y_edge_finetune,
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

    # --- Hyperparameter tuning via random search ---
    if X_device_finetunes[0] is not None and y_device_finetunes[0] is not None:
        best_param_device = random_search_boosting_params(
            X_trains[0], y_trains[0],
            X_device_finetunes[0][:, :X_trains[0].shape[1]],
            y_device_finetunes[0],
            le, n_classes,
            n_trials=5,
            verbose=True
        )
    else:
        best_param_device = {"learning_rate": config["learning_rate"], "num_boost_round": config["num_boost_round"]}

    for epoch in range(num_epochs):
        log_util.safe_log(f"""
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
            profile = forward_pass(
            d_data, e_groups, le, n_classes,
            best_param_device=best_param_device,
            X_edges_finetune=X_edges_finetune,
            y_edges_finetune=y_edges_finetune,
            residuals_devices=residuals_devices,
            device_models=device_models,
            track_profile=True
        )
        log_util.safe_log("Profile:", profile)

        # -----------------------------
        # 2. Backward Pass
        # -----------------------------
        updated_edge_models, updated_device_models, updated_edge_preds_stacked = backward_pass(
            edge_models=edge_models,
            device_models=device_models,
            edge_embeddings=edge_embeddings,
            device_embeddings=device_embeddings,
            y_true_per_edge=y_true_per_edge,
            edge_sample_slices=edge_sample_slices,
            global_pred_matrix=y_global_pred,
            n_classes=n_classes,
            use_classification=True,
            verbose=True,
            le=le
        )

        # -----------------------------
        # 3. Compute Multi-Level Accuracy
        # -----------------------------
        metrics_test = evaluate_multilevel_performance(
            device_models=updated_device_models,
            edge_models=updated_edge_models,
            edge_groups=e_groups,
            X_tests=X_tests,
            y_tests=y_tests,
            le=le,
            num_classes=n_classes,
            combine_mode="weighted",
            top_k=3,
        )

        # -----------------------------
        # 4. Save Models
        # -----------------------------
        save_hpfl_models(
            device_models, edge_models, epoch,
            y_global_pred=y_global_pred,
            global_residuals=global_residuals,
            theta_global=theta_global
        )

        # -----------------------------
        # 5. Update history
        # -----------------------------
        metrics = metrics_test["metrics"]

        history["device_accs_per_epoch"].append(metrics["device"]["acc"])
        history["edge_accs_per_epoch"].append(metrics["edge"]["acc"])
        history["global_accs"].append(metrics["global"]["acc"])

        history["device_means"].append(np.mean(metrics["device"]["acc"]))
        history["device_stds"].append(np.std(metrics["device"]["acc"]))
        history["edge_means"].append(np.mean(metrics["edge"]["acc"]))
        history["edge_stds"].append(np.std(metrics["edge"]["acc"]))

        global_acc = metrics["global"]["acc"]
        history["device_vs_global"].append([acc - global_acc for acc in metrics["device"]["acc"]])
        history["edge_vs_global"].append([acc - global_acc for acc in metrics["edge"]["acc"]])

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

    log_util.safe_log("\n================ Device Metrics ================\n")

    for idx, models_per_device in enumerate(d_models):

        if not models_per_device:
            log_util.safe_log(f"Device {idx}: ❌ No trained models found.")
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

        log_util.safe_log(f"✅ Device {idx}: Accuracy={acc:.4f}, LogLoss={ll:.4f}, "
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

    log_util.safe_log(f"\nOverall Mean Accuracy: {mean_acc:.4f}")
    log_util.safe_log(f"Overall Mean Log Loss: {np.nanmean(log_losses):.4f}")
    log_util.safe_log(f"Overall Mean Brier Score: {np.nanmean(brier_scores):.4f}")
    log_util.safe_log(f"Overall Mean Top-{top_k} Accuracy: {np.nanmean(topk_accuracies):.4f}\n")

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
    log_util.safe_log("Device accs:", device_accs)
    log_util.safe_log("Mean device accs:", mean_device_accs)

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

    log_util.safe_log(f"All plots saved to folder: {save_dir}")

# ============================================================
#                     Main
# ============================================================

if __name__ == "__main__":
    folder_path = "CIC_IoT_DIAD_2024"
    today_str = datetime.now().strftime("%Y-%m-%d")
    log_path_str = os.path.join("logs", f"{folder_path}_{today_str}")
    os.makedirs(log_path_str, exist_ok=True)
    log_util.setup_logging(log_path_str)
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
    # Collect all labels across devices and edges
    # Combine all y from devices and edge fine-tuning data safely
    y_all = np.concatenate([
        *(y for _, _, y, _, _, _, _, _ in devices_data if y is not None and len(y) > 0),
        *(np.ravel(y) for _, y in edge_finetune_data if y is not None and len(y) > 0)
    ])

    # Fit LabelEncoder only if there are labels
    if len(y_all) > 0:
        le.fit(y_all)
    else:
        log_util.safe_log("⚠️ Warning: No true labels found to fit LabelEncoder.")
    num_classes = len(np.unique(y_final))

    # 4. Train HPFL model with accuracy tracking

    history = hpfl_train_with_accuracy(d_data=devices_data, e_groups=edge_groups, edge_finetune_data = edge_finetune_data,
                                           le=le, n_classes=num_classes, verbose=True)
    plot_hpfl_all(history)


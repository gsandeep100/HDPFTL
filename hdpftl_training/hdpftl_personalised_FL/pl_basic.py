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
    "eps_threshold": 1e-4,
    "max_no_improve_edge": 5,
    "max_no_improve_device": 5
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

    # --- Base static parameters ---
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

    # --- Merge tuned parameters ---
    model_params = {**base_params, **best_params} if best_params is not None else base_params

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

    # --- Align validation features safely ---
    if X_valid is not None and y_valid is not None:
        X_valid = np.atleast_2d(np.asarray(X_valid, dtype=np.float32))
        n_features_model = X_train.shape[1]
        n_features_valid = X_valid.shape[1]
        if n_features_valid < n_features_model:
            X_valid = np.pad(X_valid, ((0, 0), (0, n_features_model - n_features_valid)), mode="constant")
        elif n_features_valid > n_features_model:
            X_valid = X_valid[:, :n_features_model]

        y_pred = model.predict_proba(X_valid)

        # --- Safe logloss computation even if y_valid misses some classes ---
        try:
            current_logloss = log_loss(y_valid, y_pred, labels=np.arange(num_classes))
        except ValueError as e:
            log_util.safe_log(f"⚠️ Logloss computation failed due to class mismatch: {e}")
            log_util.safe_log("Falling back to filtered label set for logloss computation.")
            valid_labels = np.unique(y_valid)
            valid_mask = np.isin(np.arange(num_classes), valid_labels)
            y_pred_filtered = y_pred[:, valid_mask] if y_pred.shape[1] > len(valid_labels) else y_pred
            try:
                current_logloss = log_loss(y_valid, y_pred_filtered, labels=valid_labels)
            except Exception as inner_e:
                log_util.safe_log(f"⚠️ Logloss fallback also failed: {inner_e}")
                current_logloss = np.nan
    else:
        current_logloss = None

    # --- Logloss control ---
    if prev_best_logloss is not None and current_logloss is not None and current_logloss > prev_best_logloss:
        prev_model = init_model[0] if isinstance(init_model, tuple) else init_model
        model_to_return = prev_model
        log_util.safe_log(
            f"⚠️ Logloss worsened ({current_logloss:.4f} > {prev_best_logloss:.4f}), reverting to previous model")
    else:
        model_to_return = model
        if current_logloss is not None and prev_best_logloss is not None:
            log_util.safe_log(f"✅ Improved logloss: {current_logloss:.4f} (prev: {prev_best_logloss:.4f})")

    return model_to_return, current_logloss if current_logloss is not None else prev_best_logloss



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

    # --- Convert to float32 safely ---
    if not isinstance(X, np.ndarray):
        # Prefer zero-copy if possible
        if hasattr(X, "to_numpy"):
            X_np = X.to_numpy(dtype=np.float32, copy=False)
        else:
            X_np = np.asarray(X, dtype=np.float32)
    else:
        X_np = X.astype(np.float32, copy=False)

    n_features_model = getattr(model, "n_features_in_", X_np.shape[1])

    # --- Align input features with model ---
    if X_np.shape[1] > n_features_model:
        X_np = X_np[:, :n_features_model]
    elif X_np.shape[1] < n_features_model:
        X_np = np.pad(
            X_np,
            ((0, 0), (0, n_features_model - X_np.shape[1])),
            mode="constant"
        )

    # --- Feature names ---
    columns = getattr(model, "feature_name_", [f"f{i}" for i in range(X_np.shape[1])])

    # Avoid DataFrame duplication when not needed
    if hasattr(model, "predict_proba") or not isinstance(model, lgb.Booster):
        X_df = pd.DataFrame(X_np, columns=columns)
    else:
        X_df = None

    # --- Predict probabilities ---
    if isinstance(model, lgb.Booster):
        pred = model.predict(X_np, raw_score=False)
    elif hasattr(model, "predict_proba"):
        pred = model.predict_proba(X_df)
    else:
        pred = model.predict(X_df, raw_score=False)

    pred = np.atleast_2d(pred)

    # --- Flatten binary predictions ---
    if pred.shape[1] == 1:
        pred = np.hstack([1 - pred, pred])

    # --- Clip to avoid extreme values ---
    eps = 1e-6
    pred = np.clip(pred, eps, 1 - eps)

    # --- Align with total number of classes ---
    full = np.zeros((pred.shape[0], n_classes), dtype=np.float32)
    model_classes = np.asarray(
        getattr(model, "classes_", np.arange(pred.shape[1]))
    ).astype(int)

    for i, cls in enumerate(model_classes):
        if 0 <= cls < n_classes:
            full[:, cls] = pred[:, i]

    # --- Normalize rows ---
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
    # --- Safe and memory-efficient conversion ---
    if not isinstance(X, np.ndarray):
        if hasattr(X, "to_numpy"):
            X_np = X.to_numpy(dtype=np.float32, copy=False)
        else:
            X_np = np.asarray(X, dtype=np.float32)
    else:
        X_np = X.astype(np.float32, copy=False)

    # --- Align features to model ---
    n_features_model = getattr(model, "n_features_in_", X_np.shape[1])
    if X_np.shape[1] > n_features_model:
        X_np = X_np[:, :n_features_model]
    elif X_np.shape[1] < n_features_model:
        X_np = np.pad(
            X_np,
            ((0, 0), (0, n_features_model - X_np.shape[1])),
            mode="constant"
        )

    columns = getattr(model, "feature_name_", [f"f{i}" for i in range(X_np.shape[1])])

    # Create DataFrame only if needed
    if isinstance(model, lgb.Booster):
        X_df = None
    else:
        X_df = pd.DataFrame(X_np, columns=columns)

    # --- Predict leaf indices ---
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

def device_layer_boosting(
        d_data,            # list of device tuples: (X_train, X_test, y_train, ..., X_device_finetune, y_device_finetune)
        d_residuals,       # list of residual arrays or None
        d_models,          # list of model lists or None
        le,                # label encoder with transform/ inverse_transform
        n_classes,
        best_params,       # dict with 'num_boost_round', 'learning_rate', ...
        chunk_size=5000
):
    """
    Device-level gradient boosting: residual-driven pseudo-labels, gradient updates, leaf embeddings.
    Returns updated (d_residuals, d_models, device_embeddings, device_weights).
    """

    num_devices = len(d_data)
    device_embeddings = [None] * num_devices
    device_val_scores = [None] * num_devices
    eps_residual = 1e-6

    def process_device(idx, dev_tuple):
        # Unpack device tuple (expected shape — keep consistent with your existing pipeline)
        X_train, _, y_train, _, _, _, X_device_finetune, y_device_finetune = dev_tuple
        n_samples = X_train.shape[0]

        # init residual
        if d_residuals[idx] is None:
            residual = np.zeros((n_samples, n_classes), dtype=np.float32)
            if y_train is not None:
                y_enc = le.transform(y_train)
                residual[np.arange(n_samples), y_enc] = 1.0
                del y_enc
        else:
            residual = d_residuals[idx].astype(np.float32, copy=False)

        models_per_device = list(d_models[idx]) if d_models[idx] else []
        boosting_rounds = int(best_params.get("num_boost_round", 50))
        learning_rate = float(best_params.get("learning_rate", 0.01))

        prev_res_norm = np.inf

        for t in range(boosting_rounds):
            res_norm = np.sum(np.abs(residual))
            if res_norm < eps_residual:
                log_util.safe_log(f"Device {idx}: residuals small -> stop (round {t}).")
                break
            if abs(prev_res_norm - res_norm) < eps_residual * n_samples:
                log_util.safe_log(f"Device {idx}: residuals stabilized -> stop (round {t}).")
                break
            prev_res_norm = res_norm

            # pseudo-labels from residual (argmax)
            y_pseudo = np.argmax(residual + 1e-9 * np.random.randn(*residual.shape), axis=1)

            init_model = models_per_device[-1] if models_per_device else None

            model, _ = train_lightgbm(
                X_train.astype(np.float32, copy=False), y_pseudo,
                X_valid=(X_device_finetune[:, :X_train.shape[1]].astype(np.float32, copy=False)
                         if X_device_finetune is not None else None),
                y_valid=y_device_finetune,
                best_params=best_params,
                init_model=init_model,
                prev_best_logloss=None,
                verbose=-1
            )
            models_per_device.append(model)

            # update residuals (gradient step)
            if X_train.shape[0] <= chunk_size:
                pred_proba = predict_proba_fixed(model, X_train, n_classes).astype(np.float32)
            else:
                # chunked prediction for memory safety
                preds = []
                for s in range(0, X_train.shape[0], chunk_size):
                    e = min(s + chunk_size, X_train.shape[0])
                    preds.append(predict_proba_fixed(model, X_train[s:e], n_classes).astype(np.float32))
                pred_proba = np.vstack(preds)
                del preds
            residual -= learning_rate * pred_proba
            residual = np.clip(residual, -1 + eps_residual, 1 - eps_residual)

            del pred_proba, y_pseudo, init_model
            gc.collect()

        # final true-label correction (if available)
        if y_train is not None and len(models_per_device) > 0:
            y_true_enc = le.transform(y_train)
            y_onehot = np.zeros((n_samples, n_classes), dtype=np.float32)
            y_onehot[np.arange(n_samples), y_true_enc] = 1.0

            y_pred_total = np.zeros((n_samples, n_classes), dtype=np.float32)
            for mdl in models_per_device:
                # chunked predictions on train for memory safety
                if n_samples <= chunk_size:
                    y_pred_total += predict_proba_fixed(mdl, X_train, n_classes).astype(np.float32)
                else:
                    for s in range(0, n_samples, chunk_size):
                        e = min(s + chunk_size, n_samples)
                        y_pred_total[s:e] += predict_proba_fixed(mdl, X_train[s:e], n_classes).astype(np.float32)

            residual = np.clip(y_onehot - y_pred_total, -1 + eps_residual, 1 - eps_residual)
            del y_onehot, y_pred_total, y_true_enc
            gc.collect()

        # device leaf embeddings
        leaf_indices_list = [get_leaf_indices(mdl, X_train) for mdl in models_per_device]
        if leaf_indices_list:
            leaf_concat = np.hstack(leaf_indices_list)
            # sparse one-hot construction might be better in memory, but using dense for compatibility
            leaf_emb = np.zeros((n_samples, np.max(leaf_concat) + 1), dtype=np.float32)
            leaf_emb[np.arange(n_samples)[:, None], leaf_concat] = 1.0
            device_embeddings[idx] = leaf_emb
            del leaf_indices_list, leaf_concat, leaf_emb
        else:
            device_embeddings[idx] = np.zeros((n_samples, 1), dtype=np.float32)
        gc.collect()

        # simple device score (can be improved)
        device_val_scores[idx] = (np.ones(len(models_per_device), dtype=np.float32) / len(models_per_device)
                                  if models_per_device else np.array([1.0], dtype=np.float32))

        # writebacks
        d_residuals[idx] = residual
        d_models[idx] = models_per_device

        # cleanup locals
        del residual, models_per_device, X_train, y_train, X_device_finetune, y_device_finetune
        gc.collect()

    # run devices in parallel
    with ThreadPoolExecutor(max_workers=num_devices) as ex:
        futures = [ex.submit(process_device, idx, dev_tuple) for idx, dev_tuple in enumerate(d_data)]
        for f in futures:
            f.result()

    # device_weights from device_val_scores (average)
    device_weights = np.array([np.mean(w) if isinstance(w, np.ndarray) else float(w)
                               for w in device_val_scores], dtype=np.float32)
    device_weights /= device_weights.sum() if device_weights.sum() > 0 else 1.0
    gc.collect()
    return d_residuals, d_models, device_embeddings, device_weights


def edge_layer_boosting(
        e_groups, d_embeddings, d_residuals,
        n_classes, le=None, best_param_edge=None,
        X_ftune=None, y_ftune=None, device_weights=None,
        chunk_size=5000, n_random_trials=5
):
    """
    Edge-level gradient boosting: consumes device embeddings/residuals, runs boosting per edge,
    stacks per-round predictions as edge features; updates e_residuals for global use.
    """

    log_util.safe_log("""
    ************************************************************
    *                  STARTING EDGE LAYER                     *
    *   Devices -> Edge-level boosted ensemble -> Edge output  *
    *           Gradient-loss based early stopping             *
    ************************************************************
    """)
    e_models, e_residuals, edge_embeddings_list = [], [], []
    edge_outputs, global_pred_blocks, edge_sample_slices = [], [], {}
    global_offset = 0
    eps_residual = 1e-6

    def process_edge_data(edge_devices):
        embeddings_list = [d_embeddings[i] for i in edge_devices]
        residuals_list = [
            (d_residuals[i].astype(np.float32, copy=False) if d_residuals[i] is not None else
             np.zeros((emb.shape[0], n_classes), dtype=np.float32))
            for emb, i in zip(embeddings_list, edge_devices)
        ]
        # device weights for devices under this edge
        if device_weights is None:
            weights = np.ones(len(edge_devices), dtype=np.float32)
        else:
            weights = np.array([device_weights[i] for i in edge_devices], dtype=np.float32)
        weights = weights / (weights.sum() if weights.sum() > 0 else 1.0)

        # pad embeddings to same width (dense path)
        max_cols = max(int(e.shape[1]) for e in embeddings_list)
        X_edge = np.vstack([
            np.pad(e.astype(np.float32), ((0, 0), (0, max_cols - e.shape[1])), mode='constant')
            if e.shape[1] < max_cols else e.astype(np.float32)[:, :max_cols]
            for e in embeddings_list
        ])

        # weighted residual stacking across devices under edge
        weighted_rows = []
        for emb, r, w in zip(embeddings_list, residuals_list, weights):
            n_rows = emb.shape[0]
            r = np.asarray(r, dtype=np.float32)
            if r.shape[0] < n_rows:
                r = np.vstack([r, np.zeros((n_rows - r.shape[0], r.shape[1]), dtype=np.float32)])
            else:
                r = r[:n_rows]
            if r.shape[1] < n_classes:
                r = np.pad(r, ((0, 0), (0, n_classes - r.shape[1])), mode='constant')
            else:
                r = r[:, :n_classes]
            weighted_rows.append(r * w)
        residual_edge = np.vstack(weighted_rows)
        residual_edge = np.clip(residual_edge, -1 + eps_residual, 1 - eps_residual)
        del embeddings_list, residuals_list, weighted_rows
        gc.collect()
        return residual_edge, X_edge

    for edge_idx, edge_devices in enumerate(e_groups):
        if len(edge_devices) == 0:
            e_models.append([])
            e_residuals.append(None)
            edge_embeddings_list.append(None)
            edge_outputs.append(None)
            continue

        residual_edge, X_edge = process_edge_data(edge_devices)
        models_per_edge = []
        boosting_round_outputs = []
        boosting_rounds = int((best_param_edge or {}).get("num_boost_round", config.get("edge_boosting_rounds", 5)))
        learning_rate = float((best_param_edge or {}).get("learning_rate", config.get("learning_rate", 0.01)))

        prev_res_norm = np.inf

        for t in range(boosting_rounds):
            res_norm = np.sum(np.abs(residual_edge))
            if res_norm < eps_residual:
                log_util.safe_log(f"Edge {edge_idx}: residuals small -> stop (round {t}).")
                break
            if abs(prev_res_norm - res_norm) < eps_residual * X_edge.shape[0]:
                log_util.safe_log(f"Edge {edge_idx}: residuals stabilized -> stop (round {t}).")
                break
            prev_res_norm = res_norm

            y_pseudo = np.argmax(residual_edge + 1e-9 * np.random.randn(*residual_edge.shape), axis=1)
            init_model = models_per_edge[-1] if models_per_edge else None

            model, _ = train_lightgbm(
                X_edge.astype(np.float32, copy=False), y_pseudo,
                X_valid=(X_ftune[edge_idx].astype(np.float32, copy=False)[:,:X_edge.shape[1]]
                         if X_ftune is not None else None),
                y_valid=(y_ftune[edge_idx] if y_ftune is not None else None),
                best_params=best_param_edge or {},
                init_model=init_model,
                prev_best_logloss=None,
                verbose=-1
            )

            models_per_edge.append(model)

            # predict (chunked) and store round outputs
            num_rows = X_edge.shape[0]
            if num_rows <= chunk_size:
                pred_proba = predict_proba_fixed(model, X_edge, n_classes).astype(np.float32)
            else:
                preds = []
                for s in range(0, num_rows, chunk_size):
                    e = min(s + chunk_size, num_rows)
                    preds.append(predict_proba_fixed(model, X_edge[s:e], n_classes).astype(np.float32))
                pred_proba = np.vstack(preds)
                del preds
            boosting_round_outputs.append(pred_proba)

            # gradient update
            residual_edge -= learning_rate * pred_proba
            residual_edge = np.clip(residual_edge, -1 + eps_residual, 1 - eps_residual)

            del pred_proba, y_pseudo, init_model
            gc.collect()

        # stack round outputs horizontally -> features per sample for this edge
        if boosting_round_outputs:
            e_pred_features = np.hstack(boosting_round_outputs).astype(np.float32)
        else:
            e_pred_features = np.zeros((X_edge.shape[0], n_classes), dtype=np.float32)

        e_models.append(models_per_edge)
        e_residuals.append(residual_edge.copy())
        edge_embeddings_list.append(X_edge)
        edge_outputs.append(e_pred_features)

        # book-keeping sample slices (map device -> global row indices)
        row_start = 0
        for dev_idx in edge_devices:
            n_dev = int(d_embeddings[dev_idx].shape[0])
            edge_sample_slices[dev_idx] = np.arange(global_offset + row_start, global_offset + row_start + n_dev)
            row_start += n_dev
        global_pred_blocks.append(e_pred_features)
        global_offset += X_edge.shape[0]

        del residual_edge, models_per_edge, e_pred_features, X_edge, boosting_round_outputs
        gc.collect()

    # stack global_pred_blocks with padding to same width
    if global_pred_blocks:
        max_cols = max(block.shape[1] for block in global_pred_blocks)
        padded = []
        for block in global_pred_blocks:
            if block.shape[1] < max_cols:
                block = np.hstack([block, np.zeros((block.shape[0], max_cols - block.shape[1]), dtype=np.float32)])
            padded.append(block.astype(np.float32))
        global_pred_matrix = np.vstack(padded)
        del padded
    else:
        global_pred_matrix = np.empty((0, n_classes), dtype=np.float32)

    gc.collect()
    return edge_outputs, e_models, e_residuals, edge_embeddings_list, global_pred_matrix, edge_sample_slices


def global_layer_bayesian_aggregation(
    e_outputs,
    e_residuals=None,
    e_embeddings=None,
    verbose=True,
    eps=1e-12,
    chunk_size=2048
):
    """
    Global Bayesian feature-level aggregation with reliability weighting.

    Args:
        e_outputs (list[np.ndarray]): Edge outputs (feature matrices per edge).
        e_residuals (list[np.ndarray], optional): Edge residuals (same shape as outputs or sample-wise).
        e_embeddings (list[np.ndarray], optional): Edge feature embeddings for diagnostics (unused in aggregation).
        verbose (bool): Print summary.
        eps (float): Numerical stability constant.
        chunk_size (int): Chunk size for memory-safe normalization.

    Returns:
        tuple: (
            y_global_pred_featurespace: np.ndarray (n_samples, n_features),
            None,  # placeholder for global residuals
            theta_global: dict of priors/statistics
        )
    """

    # -----------------------------
    # 1. Validate Inputs
    # -----------------------------
    valid_edges = [i for i, out in enumerate(e_outputs) if out is not None and out.size > 0]
    if not valid_edges:
        raise ValueError("No valid edges found for global aggregation!")

    H_list, weight_list = [], []
    for i in valid_edges:
        out = np.nan_to_num(e_outputs[i], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        H_list.append(out)

        if e_residuals and e_residuals[i] is not None:
            res = np.nan_to_num(e_residuals[i], nan=0.0).astype(np.float32, copy=False)
            per_sample_mse = np.mean(res**2, axis=1) if res.ndim > 1 else res**2
            # reliability = exp(-MSE)
            reliability = np.exp(-per_sample_mse / (np.mean(per_sample_mse) + eps))
        else:
            reliability = np.ones(out.shape[0], dtype=np.float32)

        # normalize per edge
        reliability /= np.mean(reliability) + eps
        weight_list.append(reliability)

        del res, per_sample_mse
        gc.collect()

    # -----------------------------
    # 2. Stack features & weights
    # -----------------------------
    H_global = np.hstack(H_list).astype(np.float32, copy=False)
    sample_weights = np.hstack(weight_list).astype(np.float32, copy=False)
    sample_weights /= np.sum(sample_weights) + eps

    n_samples, n_features = H_global.shape
    if verbose:
        print(f"[GlobalBayes] Aggregating {len(valid_edges)} edges | shape=({n_samples},{n_features})")

    # -----------------------------
    # 3. Normalize features (chunked)
    # -----------------------------
    H_norm = np.empty_like(H_global, dtype=np.float32)
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        block = H_global[start:end]
        f_min = np.min(block, axis=0, keepdims=True)
        f_max = np.max(block, axis=0, keepdims=True)
        f_range = np.clip(f_max - f_min, eps, None)
        H_norm[start:end] = np.clip((block - f_min) / f_range, eps, 1.0)
        del block, f_min, f_max, f_range
        gc.collect()

    del H_global
    gc.collect()

    # -----------------------------
    # 4. Feature Priors (mean & variance)
    # -----------------------------
    feat_mean = np.average(H_norm, axis=0, weights=sample_weights)
    feat_var = np.average((H_norm - feat_mean) ** 2, axis=0, weights=sample_weights)
    feat_precision = 1.0 / (feat_var + eps)

    theta_global = {
        "feat_mean": feat_mean.astype(np.float32),
        "feat_precision": feat_precision.astype(np.float32),
        "sample_weights": sample_weights,
        "n_features": n_features,
    }

    # -----------------------------
    # 5. Weighted Log-Likelihood
    # -----------------------------
    log_likelihood = np.log1p(H_norm - 1 + eps) * sample_weights[:, None]

    # Softmax across features (feature-wise contribution to global score)
    log_likelihood -= np.max(log_likelihood, axis=1, keepdims=True)
    exp_likelihood = np.exp(log_likelihood, dtype=np.float32)
    y_global_pred_featurespace = exp_likelihood / (np.sum(exp_likelihood, axis=1, keepdims=True) + eps)

    del H_norm, log_likelihood, exp_likelihood
    gc.collect()

    if verbose:
        print(f"[GlobalBayes] Feature-level posterior aggregated. {n_features} features combined safely.")

    return y_global_pred_featurespace, None, theta_global




# ============================================================
# Forward pass with Device → Edge → Gossip
# ============================================================

process = psutil.Process()

def get_memory_mb():
    return process.memory_info().rss / 1024**2


def forward_pass(
    devices_data, edge_groups, le, num_classes,
    best_param_device=None,
    best_param_edge=None,
    X_edges_finetune=None, y_edges_finetune=None,
    residuals_devices=None, device_models=None,
    pred_chunk_size=1024,
    track_profile=False
):
    """
    Forward pass for HPFL: device → edge → global (feature-level residual aggregation).

    This version omits class-level Bayesian aggregation and uses
    fully feature-based residual fusion for the global layer.

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
            device_models, edge_models, edge_outputs, fusion_meta,
            residuals_devices, residuals_edges, y_global_pred,
            device_embeddings, edge_embeddings_list, y_global_true,
            y_true_per_edge, global_residuals, edge_sample_slices, profile
        )
    """

    profile = {}

    # ------------------------------------------------------------
    # 1. Device Layer
    # ------------------------------------------------------------
    t0, mem0 = time.time(), get_memory_mb()
    residuals_devices = residuals_devices or [None] * len(devices_data)
    device_models = device_models or [None] * len(devices_data)

    residuals_devices, device_models, device_embeddings, device_weights = device_layer_boosting(
        devices_data, residuals_devices, device_models, le, num_classes, best_param_device
    )

    assert device_embeddings is not None, "Device embeddings returned as None!"

    # Type normalization and cleanup
    device_embeddings = [emb.astype(np.float32, copy=False) for emb in device_embeddings]
    residuals_devices = [
        res.astype(np.float32, copy=False) if res is not None else None for res in residuals_devices
    ]
    del devices_data
    gc.collect()

    if track_profile:
        profile['Device'] = {'time': time.time() - t0, 'mem_MB': get_memory_mb() - mem0}

    # ------------------------------------------------------------
    # 2. Edge Layer
    # ------------------------------------------------------------
    t0, mem0 = time.time(), get_memory_mb()
    n_samples = sum(e.shape[0] for e in device_embeddings if e is not None)

    edge_outputs, edge_models, residuals_edges, edge_embeddings_list, _, edge_sample_slices = edge_layer_boosting(
        e_groups=edge_groups,
        d_embeddings=device_embeddings,
        d_residuals=residuals_devices,
        n_classes=num_classes,
        best_param_edge=best_param_edge,
        X_ftune=X_edges_finetune,
        y_ftune=y_edges_finetune,
        device_weights=device_weights
    )

    # Normalize types and cleanup
    edge_embeddings_list = [emb.astype(np.float32, copy=False) if emb is not None else None for emb in edge_embeddings_list]
    residuals_edges = [res.astype(np.float32, copy=False) if res is not None else None for res in residuals_edges]
    del X_edges_finetune
    gc.collect()

    if track_profile:
        profile['Edge'] = {'time': time.time() - t0, 'mem_MB': get_memory_mb() - mem0}

    # ------------------------------------------------------------
    # 3. Build Global Ground Truth
    # ------------------------------------------------------------
    t0, mem0 = time.time(), get_memory_mb()
    y_global_true = np.full((n_samples,), -1, dtype=int)
    written_mask = np.zeros(n_samples, dtype=bool)
    y_true_per_edge = []

    for edge_idx, edge_devices in enumerate(edge_groups):
        edge_labels = []
        for dev_idx in edge_devices:
            idxs = edge_sample_slices.get(dev_idx, np.arange(device_embeddings[dev_idx].shape[0]))

            if y_edges_finetune is not None and dev_idx < len(y_edges_finetune):
                y_dev = np.array(y_edges_finetune[dev_idx])
                if y_dev.size < idxs.size:
                    y_dev = np.pad(y_dev, (0, idxs.size - y_dev.size), mode="edge")
                elif y_dev.size > idxs.size:
                    y_dev = y_dev[:idxs.size]
            else:
                y_dev = np.full(len(idxs), -1, dtype=int)

            not_written = ~written_mask[idxs]
            y_global_true[idxs[not_written]] = y_dev[not_written]
            written_mask[idxs[not_written]] = True
            edge_labels.append(y_dev)

        y_true_per_edge.append(np.hstack(edge_labels))

    if track_profile:
        profile['GroundTruth'] = {'time': time.time() - t0, 'mem_MB': get_memory_mb() - mem0}

    # ------------------------------------------------------------
    # 4. Global Layer Aggregation (Feature-level residual fusion)
    # ------------------------------------------------------------
    t0, mem0 = time.time(), get_memory_mb()

    y_global_pred, global_residuals, fusion_meta = global_layer_bayesian_aggregation(
        e_outputs=edge_embeddings_list,
        e_residuals=residuals_edges,
        e_embeddings=None,
        verbose=True
    )

    if track_profile:
        profile['Global'] = {'time': time.time() - t0, 'mem_MB': get_memory_mb() - mem0}

    # ------------------------------------------------------------
    # 5. Cleanup and Return
    # ------------------------------------------------------------
    del y_edges_finetune
    gc.collect()

    common_returns = (
        device_models,
        edge_models,
        edge_outputs,
        fusion_meta,
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

    return common_returns + ((profile,) if track_profile else (None,))


def backward_pass(edge_models, device_models,
                  edge_embeddings, device_embeddings,
                  y_true_per_edge,
                  edge_sample_slices=None,
                  global_pred_matrix=None,
                  n_classes=2,
                  use_classification=True,
                  verbose=True,
                  le=None,
                  feature_mode=True):
    """
    Hierarchical backward feedback with multiclass-safe updates for edge and device models,
    memory-managed and aligned with feature-level HPFL aggregation.
    """
    if verbose:
        log_util.safe_log("\n" + "*" * 60)
        log_util.safe_log("*" + " " * 20 + "STARTING BACKWARD PASS" + " " * 20 + "*")
        log_util.safe_log("*" * 60 + "\n")

    num_edges = len(edge_models)
    updated_edge_preds_list, updated_edge_models = [], []

    # -----------------------------
    # 1) Update Edge Models
    # -----------------------------
    for ei in range(num_edges):
        model = edge_models[ei]
        X_edge = edge_embeddings[ei]
        y_edge = np.asarray(y_true_per_edge[ei]).ravel()

        if X_edge is None or model is None or y_edge.size == 0:
            updated_edge_models.append(None)
            updated_edge_preds_list.append(np.zeros((0, n_classes), dtype=np.float32))
            if verbose:
                log_util.safe_log(f"Edge {ei}: no model or data, skipping.")
            continue

        # feature-level fallback (mean pooling)
        if feature_mode and X_edge.ndim > 1 and X_edge.shape[0] > 1:
            X_edge = np.mean(X_edge, axis=0, keepdims=True)

        y_edge_labels = encode_labels_safe(le, y_edge, n_classes=n_classes) if le else np.clip(y_edge.astype(int), 0, n_classes - 1)

        params = dict(
            n_estimators=getattr(model, "n_estimators", 100),
            learning_rate=getattr(model, "learning_rate", 0.1),
            max_depth=getattr(model, "max_depth", -1),
            random_state=getattr(model, "random_state", 42)
        )

        if use_classification:
            clf = LGBMClassifier(objective='multiclass', num_class=n_classes, **params)
            clf.fit(X_edge, y_edge_labels)
            preds = clf.predict_proba(X_edge)
            updated_edge_models.append(clf)
        else:
            reg = LGBMRegressor(**params)
            y_onehot = np.zeros((X_edge.shape[0], n_classes), dtype=np.float32)
            y_onehot[np.arange(X_edge.shape[0]), y_edge_labels] = 1.0
            residual_targets = y_onehot - model.predict(X_edge)
            reg.fit(X_edge, residual_targets)
            preds = reg.predict(X_edge)
            updated_edge_models.append(reg)
            del y_onehot, residual_targets
            gc.collect()

        updated_edge_preds_list.append(preds.astype(np.float32, copy=False))
        del X_edge, y_edge_labels, y_edge, preds
        gc.collect()

        if verbose:
            log_util.safe_log(f"Edge {ei}: updated model with feature_mode={feature_mode}")

    # -----------------------------
    # 2) Stack Updated Edge Predictions
    # -----------------------------
    if updated_edge_preds_list:
        max_dim = max(pred.shape[1] for pred in updated_edge_preds_list if pred.size > 0)
        aligned_preds = [
            np.pad(pred, ((0, 0), (0, max_dim - pred.shape[1])), mode='constant')
            if pred.shape[1] < max_dim else pred[:, :max_dim]
            for pred in updated_edge_preds_list
        ]
        updated_edge_preds_stacked = np.vstack(aligned_preds).astype(np.float32, copy=False)
        del aligned_preds
        gc.collect()
    else:
        updated_edge_preds_stacked = np.zeros((0, n_classes), dtype=np.float32)

    # -----------------------------
    # 3) Update Device Models
    # -----------------------------
    updated_device_models = []
    y_global_labels = np.hstack([np.asarray(y).ravel() for y in y_true_per_edge]) if y_true_per_edge else None

    for di, model in enumerate(device_models):
        X_dev = device_embeddings[di]
        if X_dev is None or model is None:
            updated_device_models.append(None)
            continue

        if feature_mode and X_dev.ndim > 1 and X_dev.shape[0] > 1:
            X_dev = np.mean(X_dev, axis=0, keepdims=True)

        if edge_sample_slices and di in edge_sample_slices:
            global_idxs = edge_sample_slices[di]
            preds_for_device = updated_edge_preds_stacked[global_idxs, :] if global_idxs.max() < updated_edge_preds_stacked.shape[0] else None
        else:
            preds_for_device = None

        if preds_for_device is None:
            updated_device_models.append(model)
            continue

        y_dev_true = (
            y_global_labels[global_idxs] if y_global_labels is not None else np.argmax(preds_for_device, axis=1)
        )

        params = dict(
            n_estimators=getattr(model, "n_estimators", 100),
            learning_rate=getattr(model, "learning_rate", 0.1),
            max_depth=getattr(model, "max_depth", -1),
            random_state=getattr(model, "random_state", 42)
        )

        if use_classification:
            clf = LGBMClassifier(objective='multiclass', num_class=n_classes, **params)
            clf.fit(X_dev, y_dev_true)
            updated_device_models.append(clf)
        else:
            reg = LGBMRegressor(**params)
            y_onehot = np.zeros((X_dev.shape[0], n_classes), dtype=np.float32)
            y_onehot[np.arange(X_dev.shape[0]), y_dev_true] = 1.0
            residual_targets = y_onehot - preds_for_device
            reg.fit(X_dev, residual_targets)
            updated_device_models.append(reg)
            del y_onehot, residual_targets
            gc.collect()

        del X_dev, preds_for_device, y_dev_true
        gc.collect()

        if verbose:
            log_util.safe_log(f"Device {di}: updated model (feature_mode={feature_mode}).")

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

    # Encode labels once
    y_train_enc = encode_labels_safe(le, y_train, n_classes)
    y_valid_enc = encode_labels_safe(le, y_valid, n_classes)

    prev_best_logloss = np.inf  # monotonic logloss baseline
    all_classes = np.arange(n_classes)

    for trial in range(n_trials):
        # -----------------------------
        # Sample random hyperparameters
        # -----------------------------
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

        # -----------------------------
        # Train LightGBM model
        # -----------------------------
        model, _ = train_lightgbm(
            X_train, y_train_enc,
            X_valid=X_valid, y_valid=y_valid_enc,
            best_params=None,
            **params
        )

        # -----------------------------
        # Evaluate validation log-loss
        # -----------------------------
        y_pred = predict_proba_fixed(model, X_valid, n_classes)
        score = log_loss(y_valid_enc, y_pred, labels=all_classes)

        # -----------------------------
        # Accept trial only if logloss improves monotonic baseline
        # -----------------------------
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
    """
    Encode labels safely — handles numeric and unseen labels gracefully.

    Args:
        le : LabelEncoder instance
        y  : labels to encode (array-like)
        n_classes : total number of classes

    Returns:
        np.ndarray of encoded integer labels
    """
    y_arr = np.array(y)

    # -----------------------------
    # 1) Handle numeric labels
    # -----------------------------
    if np.issubdtype(y_arr.dtype, np.number):
        # Clip any invalid labels (<0 or >= n_classes)
        y_arr = np.clip(y_arr, 0, n_classes - 1)
        return y_arr.astype(int)

    # -----------------------------
    # 2) Ensure LabelEncoder is fitted
    # -----------------------------
    if not hasattr(le, "classes_") or len(le.classes_) < n_classes:
        le.fit(np.arange(n_classes))

    # -----------------------------
    # 3) Transform labels safely
    # -----------------------------
    try:
        return le.transform(y_arr)
    except ValueError:
        # Handle unseen categories
        known_classes = set(le.classes_)
        new_classes = set(np.unique(y_arr))
        combined_classes = np.array(sorted(known_classes.union(new_classes)))
        le.fit(combined_classes)
        return le.transform(y_arr)


def pad_predictions(pred_list, num_samples=None, num_classes=None):
    """
    Pad a list of predictions to a uniform shape for stacking.
    Any None predictions are replaced with zeros.
    """
    # -----------------------------
    # 1) Determine target dimensions
    # -----------------------------
    if num_samples is None:
        num_samples = max(pred.shape[0] for pred in pred_list if pred is not None)
    if num_classes is None:
        num_classes = max(pred.shape[1] for pred in pred_list if pred is not None)

    padded_preds = []

    # -----------------------------
    # 2) Pad each prediction
    # -----------------------------
    for pred in pred_list:
        if pred is None:
            padded = np.zeros((num_samples, num_classes), dtype=float)
        else:
            pred = np.atleast_2d(pred)

            # Pad rows if smaller
            if pred.shape[0] < num_samples:
                pad_rows = num_samples - pred.shape[0]
                pred = np.pad(pred, ((0, pad_rows), (0, 0)), mode='constant')

            # Pad columns if smaller
            if pred.shape[1] < num_classes:
                pad_cols = num_classes - pred.shape[1]
                pred = np.pad(pred, ((0, 0), (0, pad_cols)), mode='constant')

            # Truncate if larger
            pred = pred[:num_samples, :num_classes]

            padded = pred

        padded_preds.append(padded)

    # -----------------------------
    # 3) Stack all padded predictions
    # -----------------------------
    return np.stack(padded_preds, axis=0)


def average_predictions(pred_list, num_samples=None, num_classes=None):
    """
    Safely average predictions across models, with padding.
    """
    # -----------------------------
    # 1) Handle empty input
    # -----------------------------
    if not pred_list:
        return None

    # -----------------------------
    # 2) Convert all non-None predictions to numpy arrays
    # -----------------------------
    pred_list_clean = [np.array(p) for p in pred_list if p is not None]

    if not pred_list_clean:
        return None

    # -----------------------------
    # 3) Pad predictions to uniform shape
    # -----------------------------
    padded_preds = pad_predictions(pred_list_clean, num_samples, num_classes)

    # -----------------------------
    # 4) Average along model axis
    # -----------------------------
    avg_preds = np.mean(padded_preds, axis=0)  # shape: (num_samples, num_classes)

    return avg_preds


def safe_array(X):
    """Convert input to numeric NumPy array for LightGBM."""
    # -----------------------------
    # 1) Handle None input
    # -----------------------------
    if X is None:
        return None

    # -----------------------------
    # 2) Convert pandas objects to numpy
    # -----------------------------
    if isinstance(X, (pd.DataFrame, pd.Series)):
        arr = X.to_numpy(dtype=np.float32)
    else:
        arr = np.asarray(X, dtype=np.float32)

    # -----------------------------
    # 3) Return numeric array
    # -----------------------------
    return arr


def safe_labels(y):
    """Convert labels to integer NumPy array for LightGBM."""
    # -----------------------------
    # 1) Handle None input
    # -----------------------------
    if y is None:
        return None

    # -----------------------------
    # 2) Convert pandas objects to numpy
    # -----------------------------
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = y.to_numpy()

    # -----------------------------
    # 3) Ensure array
    # -----------------------------
    y = np.asarray(y)

    # -----------------------------
    # 4) Convert one-hot labels to class indices if needed
    # -----------------------------
    if y.ndim > 1 and y.shape[1] > 1:
        y = np.argmax(y, axis=1)

    # -----------------------------
    # 5) Convert to integer type
    # -----------------------------
    return y.astype(np.int32)


def safe_fit(X, y, *, model=None, **fit_kwargs):
    """
    Safely fit a LightGBM classifier with numeric arrays and correct label handling.
    """
    # -----------------------------
    # 1) Convert inputs to safe arrays
    # -----------------------------
    X_safe = safe_array(X)
    y_safe = safe_labels(y)

    # -----------------------------
    # 2) Determine number of classes and objective
    # -----------------------------
    n_classes = len(np.unique(y_safe))
    objective = 'binary' if n_classes == 2 else 'multiclass'

    # -----------------------------
    # 3) Extract verbose if provided
    # -----------------------------
    verbose = fit_kwargs.pop("verbose", None)

    # -----------------------------
    # 4) Initialize or use provided model
    # -----------------------------
    if model is None:
        model = LGBMClassifier(objective=objective)
        if verbose is not None:
            model.set_params(verbose=verbose)

    # -----------------------------
    # 5) Set num_class for multiclass problems
    # -----------------------------
    if n_classes > 2:
        model.set_params(num_class=n_classes)

    # -----------------------------
    # 6) Fit model with provided keyword arguments
    # -----------------------------
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
    # -----------------------------
    # 1) Convert X and y to numpy arrays
    # -----------------------------
    X_np = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
    y_np = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)

    rng = np.random.default_rng(seed)
    unique_classes = np.unique(y_np)

    # -----------------------------
    # 2) Dirichlet non-IID split for devices
    # -----------------------------
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

    # -----------------------------
    # 3) Device-level train / finetune splits
    # -----------------------------
    devices_data = []
    for dev_id, idxs in enumerate(device_indices):
        X_dev, y_dev = X_np[idxs], y_np[idxs]

        # Step 3a: Device train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_dev, y_dev, test_size=0.1, random_state=seed
        )

        # Step 3b: Split training into device finetune + edge finetune
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

    # -----------------------------
    # 4) Assign devices to edges
    # -----------------------------
    def make_edge_groups(num_devices, n_edges, random_state=None):
        rng_local = np.random.default_rng(random_state)
        devices = np.arange(num_devices)
        rng_local.shuffle(devices)
        return np.array_split(devices, n_edges)

    edge_groups = make_edge_groups(num_devices, n_edges, random_state=seed)

    # -----------------------------
    # 5) Aggregate edge finetune data
    # -----------------------------
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
        preds_list = [predict_proba_fixed(m, X_test, num_classes)
                      for m in (models_per_device if isinstance(models_per_device, list) else [models_per_device])]

        # Compute weights
        if combine_mode == "weighted":
            w = np.array(device_val_scores[idx]) if device_val_scores and idx < len(device_val_scores) else np.ones(len(preds_list))
            w = normalize_weights(w)
        else:
            w = None

        # Combine predictions
        if combine_mode == "last":
            y_pred_probs = preds_list[-1]
        elif combine_mode == "average":
            y_pred_probs = np.mean(preds_list, axis=0)
        elif combine_mode == "weighted":
            y_pred_probs = sum(wi * pi for pi, wi in zip(preds_list, w))
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
            edge_pred_accum = predict_proba_fixed(mdl, np.vstack([X_tests[d] for d in edge_devices]), num_classes)
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
    y_global_true = np.concatenate([le.transform(y_tests[d]) for d in range(len(X_tests))])
    global_acc = accuracy_score(y_global_true, global_pred.argmax(axis=1))

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

    # Initialize placeholders
    residuals_devices = [None] * len(d_data)
    device_models = [None] * len(d_data)
    residuals_edges = [None] * len(e_groups)
    edge_models = [None] * len(e_groups)
    device_embeddings = [None] * len(d_data)
    edge_embeddings = [None] * len(e_groups)

    num_epochs = config["epoch"]

    # Initialize history dictionary
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
        best_param_device = {
            "learning_rate": config["learning_rate"],
            "num_boost_round": config["num_boost_round"]
        }

    for epoch in range(num_epochs):
        log_util.safe_log(f"""
        ************************************************************
        *                                                          *
        *                  === Epoch {epoch + 1}/{num_epochs} ===  *
        *                                                          *
        ************************************************************
        """)

        # -----------------------------
        # 1️⃣ Forward Pass
        # -----------------------------
        (
            device_models, edge_models, edge_outputs, theta_global,
            residuals_devices, residuals_edges, y_global_pred,
            device_embeddings, edge_embeddings, y_global_true,
            y_true_per_edge, global_residuals, edge_sample_slices,
            profile
        ) = forward_pass(
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
        # 2️⃣ Backward Pass
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
        # 3️⃣ Compute Multi-Level Accuracy
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
        # 4️⃣ Save Models
        # -----------------------------
        save_hpfl_models(
            device_models, edge_models, epoch,
            y_global_pred=y_global_pred,
            global_residuals=global_residuals,
            theta_global=theta_global
        )

        # -----------------------------
        # 5️⃣ Update history
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

    Args:
        y_true: array-like of shape (n_samples,), true class indices
        y_prob: array-like of shape (n_samples, n_classes), predicted probabilities

    Returns:
        float32: mean multi-class Brier score
    """
    n_samples, n_classes = y_prob.shape
    y_one_hot = np.zeros_like(y_prob, dtype=np.float32)
    y_one_hot[np.arange(n_samples), y_true] = 1
    squared_diff = (y_prob - y_one_hot) ** 2
    return np.mean(np.sum(squared_diff, axis=1))


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
    folder_path = "CIC_IoT_IDAD_Dataset_2024"
    log_util.setup_logging(f"Memory available: {psutil.virtual_memory().available / 1024 ** 3:.2f} GB")

    # Main log folder
    log_path_str = os.path.join("logs", f"{folder_path}")

    # Setup logging in the new subfolder
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


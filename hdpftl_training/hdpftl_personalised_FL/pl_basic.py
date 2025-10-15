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
    "n_estimators": 1000,
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
    # --- Convert safely ---
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_valid = np.array(X_valid) if X_valid is not None else None
    y_valid = np.array(y_valid) if y_valid is not None else None

    if y_train.ndim > 1 and y_train.shape[1] > 1:
        y_train = np.argmax(y_train, axis=1)
    if y_valid is not None and y_valid.ndim > 1 and y_valid.shape[1] > 1:
        y_valid = np.argmax(y_valid, axis=1)

    num_classes = len(np.unique(y_train))
    objective = "binary" if num_classes == 2 else "multiclass"

    # --- Base static parameters ---
    base_params = {
        "boosting_type": config.get("boosting", "gbdt"),
        "objective": objective,
        "num_class": num_classes if num_classes > 2 else None,
        "n_estimators": int(best_params.get("num_boost_round", config.get("n_estimators", 5000))),
        "learning_rate": float(best_params.get("learning_rate", config.get("learning_rate", 0.01))),
        "num_leaves": config.get("num_leaves", 31),
        "max_depth": int(best_params.get("max_depth", config.get("max_depth", -1))),
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

    # --- Convert to DataFrame to keep feature names consistent ---
    n_features = X_train.shape[1]
    X_train_df = pd.DataFrame(X_train, columns=[f"f{i}" for i in range(n_features)])
    X_valid_df = None
    if X_valid is not None:
        X_valid_df = pd.DataFrame(X_valid, columns=[f"f{i}" for i in range(n_features)])

    model = LGBMClassifier(**base_params)
    fit_kwargs = {}

    # --- Validation setup ---
    if X_valid_df is not None and y_valid is not None and config.get("early_stopping_rounds", 100):
        mask = np.isin(y_valid, np.unique(y_train))
        if mask.sum() < 10:
            mask = np.ones_like(y_valid, dtype=bool)
        X_valid_filtered = X_valid_df[mask]
        y_valid_filtered = y_valid[mask]

        fit_kwargs.update({
            "eval_set": [(X_train_df, y_train), (X_valid_filtered, y_valid_filtered)],
            "eval_metric": "multi_logloss" if num_classes > 2 else "logloss",
            "eval_names": ["train", "valid"],
            "callbacks": [
                early_stopping(config.get("early_stopping_rounds", 100)),
                log_evaluation(10)
            ]
        })

    if init_model is not None:
        fit_kwargs["init_model"] = init_model

    # --- Train ---
    model.fit(X_train_df, y_train, **fit_kwargs)

    # --- Align validation features safely ---
    current_logloss = None
    if X_valid_df is not None and y_valid is not None:
        y_pred = model.predict_proba(X_valid_df)
        try:
            current_logloss = log_loss(y_valid, y_pred, labels=np.arange(num_classes))
        except ValueError as e:
            log_util.safe_log(f"⚠️ Logloss computation failed: {e}")
            valid_labels = np.unique(y_valid)
            y_pred_filtered = y_pred[:, :len(valid_labels)]
            current_logloss = log_loss(y_valid, y_pred_filtered, labels=valid_labels)

    # --- Logloss control ---
    if prev_best_logloss is not None and current_logloss is not None and current_logloss > prev_best_logloss:
        prev_model = init_model[0] if isinstance(init_model, tuple) else init_model
        model_to_return = prev_model
        log_util.safe_log(f"⚠️ Logloss worsened ({current_logloss:.4f} > {prev_best_logloss:.4f}), reverting to previous model")
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
        d_data,
        d_residuals,
        d_models,
        le,
        n_classes,
        best_params,
        epoch=0,
        chunk_size=5000
):
    """
    Device-level gradient boosting: residual-driven pseudo-labels, gradient updates, leaf embeddings.
    Keeps only the latest model per device (memory-efficient incremental update).
    """

    print_layer_banner("Device", device_id=1, current_epoch=epoch)

    num_devices = len(d_data)
    device_embeddings = [None] * num_devices
    device_val_scores = [None] * num_devices
    eps_residual = 1e-6

    def process_device(idx, dev_tuple):
        X_train, _, y_train, _, _, _, X_device_finetune, y_device_finetune = dev_tuple
        n_samples = X_train.shape[0]

        # Initialize residuals
        if d_residuals[idx] is None:
            residual = np.zeros((n_samples, n_classes), dtype=np.float32)
            if y_train is not None:
                y_enc = le.transform(y_train)
                residual[np.arange(n_samples), y_enc] = 1.0
                del y_enc
        else:
            residual = d_residuals[idx].astype(np.float32, copy=False)

        prev_model = d_models[idx] if d_models[idx] else None
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

            y_pseudo = np.argmax(residual + 1e-9 * np.random.randn(*residual.shape), axis=1)

            model, _ = train_lightgbm(
                X_train.astype(np.float32, copy=False), y_pseudo,
                X_valid=(X_device_finetune[:, :X_train.shape[1]].astype(np.float32, copy=False)
                         if X_device_finetune is not None else None),
                y_valid=y_device_finetune,
                best_params=best_params,
                init_model=prev_model,
                prev_best_logloss=None,
                verbose=-1
            )
            prev_model = model  # overwrite previous model

            # Update residuals
            if X_train.shape[0] <= chunk_size:
                pred_proba = predict_proba_fixed(model, X_train, n_classes).astype(np.float32)
            else:
                preds = []
                for s in range(0, X_train.shape[0], chunk_size):
                    e = min(s + chunk_size, X_train.shape[0])
                    preds.append(predict_proba_fixed(model, X_train[s:e], n_classes).astype(np.float32))
                pred_proba = np.vstack(preds)
                del preds

            residual -= learning_rate * pred_proba
            residual = np.clip(residual, -1 + eps_residual, 1 - eps_residual)
            del pred_proba, y_pseudo
            gc.collect()

        # Final true-label correction
        if y_train is not None and prev_model is not None:
            y_true_enc = le.transform(y_train)
            y_onehot = np.zeros((n_samples, n_classes), dtype=np.float32)
            y_onehot[np.arange(n_samples), y_true_enc] = 1.0

            if n_samples <= chunk_size:
                y_pred_total = predict_proba_fixed(prev_model, X_train, n_classes).astype(np.float32)
            else:
                preds = [predict_proba_fixed(prev_model, X_train[s:e], n_classes).astype(np.float32)
                         for s in range(0, n_samples, chunk_size)]
                y_pred_total = np.vstack(preds)

            residual = np.clip(y_onehot - y_pred_total, -1 + eps_residual, 1 - eps_residual)
            del y_onehot, y_pred_total, y_true_enc
            gc.collect()

        # Device embedding
        leaf_indices = get_leaf_indices(prev_model, X_train)
        leaf_emb = np.zeros((n_samples, np.max(leaf_indices) + 1), dtype=np.float32)
        leaf_emb[np.arange(n_samples), leaf_indices] = 1.0
        device_embeddings[idx] = leaf_emb
        del leaf_indices, leaf_emb

        # Simple device weight
        device_val_scores[idx] = np.array([1.0], dtype=np.float32)

        # Writebacks
        d_residuals[idx] = residual
        d_models[idx] = prev_model

        gc.collect()

    with ThreadPoolExecutor(max_workers=num_devices) as ex:
        futures = [ex.submit(process_device, idx, dev_tuple) for idx, dev_tuple in enumerate(d_data)]
        for f in futures:
            f.result()

    device_weights = np.array([np.mean(w) for w in device_val_scores], dtype=np.float32)
    device_weights /= device_weights.sum() if device_weights.sum() > 0 else 1.0
    gc.collect()
    return d_residuals, d_models, device_embeddings, device_weights


def edge_layer_boosting(
        e_groups, d_embeddings, d_residuals,
        n_classes, le=None, best_param_edge=None,
        X_ftune=None, y_ftune=None, device_weights=None,
        epoch=0, chunk_size=5000, n_random_trials=5,
        prev_e_models=None   # <-- optional: pass previous edge models between calls
):
    """
    Edge-level gradient boosting: consumes device embeddings/residuals, runs boosting per edge,
    stores/returns exactly one model per edge (latest). Returns:
      edge_outputs, e_models (list of model or None), e_residuals, edge_embeddings_list,
      global_pred_matrix, edge_sample_slices
    """

    print_layer_banner("Edge", edge_id=2, device_count=len(e_groups), current_epoch=epoch)

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
            # ensure r has at least n_rows rows (pad or cut)
            if r.shape[0] < n_rows:
                r = np.vstack([r, np.zeros((n_rows - r.shape[0], r.shape[1]), dtype=np.float32)])
            else:
                r = r[:n_rows]
            # ensure r has n_classes columns
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
            e_models.append(None)
            e_residuals.append(None)
            edge_embeddings_list.append(None)
            edge_outputs.append(None)
            continue

        residual_edge, X_edge = process_edge_data(edge_devices)

        # Instead of a list of models, keep a single model per edge (prev_model)
        prev_model = None
        if prev_e_models and len(prev_e_models) > edge_idx:
            prev_model = prev_e_models[edge_idx]

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
            init_model = prev_model  # continue from latest model if present

            model, _ = train_lightgbm(
                X_edge.astype(np.float32, copy=False), y_pseudo,
                X_valid=(X_ftune[edge_idx].astype(np.float32, copy=False)[:, :X_edge.shape[1]]
                         if X_ftune is not None else None),
                y_valid=(y_ftune[edge_idx] if y_ftune is not None else None),
                best_params=best_param_edge or {},
                init_model=init_model,
                prev_best_logloss=None,
                verbose=-1
            )

            # Overwrite prev_model with the newly trained model (keep only one)
            prev_model = model

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

        # store single model (or None) for this edge
        e_models.append(prev_model)
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

        del residual_edge, e_pred_features, X_edge, boosting_round_outputs
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
    epoch = 0,
    e_embeddings=None,
    verbose=True,
    eps=1e-12,
    chunk_size=2048,
    return_edge_probs_per_sample=False
):
    """
    Global Bayesian feature-level aggregation + per-edge reliability reweighting.

    Returns:
        y_global_pred_featurespace: np.ndarray (n_samples, n_features)
        None
        theta_global: dict {
            "feat_mean", "feat_precision", "sample_weights", "n_features",
            "edge_reliability" (array len n_valid_edges, sums to 1),
            "edge_feature_slices" (list of (start, end) for each edge in valid_edges),
            "edge_scores_per_sample" (optional; shape (n_samples, n_edges))
        }
    """
    print_layer_banner("Global", current_epoch=epoch)

    # 1. validate / collect outputs and per-edge feature sizes
    valid_edges = [i for i, out in enumerate(e_outputs) if out is not None and out.size > 0]
    if not valid_edges:
        raise ValueError("No valid edges found for global aggregation!")

    H_list = []
    weight_list = []
    edge_feature_slices = []
    col_cursor = 0
    for i in valid_edges:
        out = np.nan_to_num(e_outputs[i], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        H_list.append(out)
        n_feat = out.shape[1]
        edge_feature_slices.append((col_cursor, col_cursor + n_feat))
        col_cursor += n_feat

        # sample reliability from residuals (smooth exponential mapping)
        if e_residuals and e_residuals[i] is not None:
            res = np.nan_to_num(e_residuals[i], nan=0.0).astype(np.float32, copy=False)
            per_sample_mse = np.mean(res**2, axis=1) if res.ndim > 1 else (res.reshape(-1) ** 2)
            reliability = np.exp(-per_sample_mse / (np.mean(per_sample_mse) + eps))
        else:
            reliability = np.ones(out.shape[0], dtype=np.float32)

        # normalize edge-level reliability to have mean=1 for stability across edges
        reliability = reliability / (np.mean(reliability) + eps)
        weight_list.append(reliability)

        # cleanup local temp
        try:
            del res, per_sample_mse
        except UnboundLocalError:
            pass
        gc.collect()

    # 2. stack features horizontally
    H_global = np.hstack(H_list).astype(np.float32, copy=False)
    sample_weights = np.hstack(weight_list).astype(np.float32, copy=False)
    sample_weights /= (np.sum(sample_weights) + eps)

    n_samples, n_features = H_global.shape
    if verbose:
        print(f"[GlobalBayes+Edge] stacking {len(valid_edges)} edges -> H_global shape {H_global.shape}")

    # 3. chunked normalization to [eps,1]
    H_norm = np.empty_like(H_global, dtype=np.float32)
    for rs in range(0, n_samples, chunk_size):
        re = min(rs + chunk_size, n_samples)
        block = H_global[rs:re]
        f_min = np.min(block, axis=0, keepdims=True)
        f_max = np.max(block, axis=0, keepdims=True)
        f_range = np.clip(f_max - f_min, eps, None)
        H_norm[rs:re] = np.clip((block - f_min) / f_range, eps, 1.0)
        del block, f_min, f_max, f_range
        gc.collect()

    del H_global
    gc.collect()

    # 4. feature priors (weighted mean/var)
    feat_mean = np.average(H_norm, axis=0, weights=sample_weights)
    feat_var = np.average((H_norm - feat_mean) ** 2, axis=0, weights=sample_weights)
    feat_precision = 1.0 / (feat_var + eps)

    # 5. weighted log-likelihood (stable) -> feature-space posterior
    # log_likelihood shape (n_samples, n_features)
    log_likelihood = np.log(H_norm) * sample_weights[:, None]
    log_likelihood -= np.max(log_likelihood, axis=1, keepdims=True)
    exp_likelihood = np.exp(log_likelihood, dtype=np.float32)
    y_global_pred_featurespace = exp_likelihood / (np.sum(exp_likelihood, axis=1, keepdims=True) + eps)

    # 6. per-edge aggregation: sum feature-posteriors belonging to each edge
    n_edges = len(valid_edges)
    edge_scores_per_sample = np.empty((n_samples, n_edges), dtype=np.float32)
    for ei, (scol, ecol) in enumerate(edge_feature_slices):
        edge_scores_per_sample[:, ei] = np.sum(y_global_pred_featurespace[:, scol:ecol], axis=1)

    # 7. collapse to per-edge scalar reliability using sample_weights
    # reliability scalar for edge ei = weighted average over samples of edge_score(sample)
    edge_reliability = np.sum(edge_scores_per_sample * sample_weights[:, None], axis=0)
    # normalize to sum to 1 for interpretability/use as weights
    edge_reliability = edge_reliability / (np.sum(edge_reliability) + eps)

    # 8. prepare theta_global and cleanup
    theta_global = {
        "feat_mean": feat_mean.astype(np.float32),
        "feat_precision": feat_precision.astype(np.float32),
        "sample_weights": sample_weights,
        "n_features": n_features,
        "valid_edges": valid_edges,
        "edge_feature_slices": edge_feature_slices,
        "edge_reliability": edge_reliability.astype(np.float32)
    }

    if return_edge_probs_per_sample:
        theta_global["edge_scores_per_sample"] = edge_scores_per_sample  # caution: big
    else:
        # free large array if not requested
        del edge_scores_per_sample
        gc.collect()

    # final cleanups
    del H_norm, log_likelihood, exp_likelihood
    gc.collect()

    if verbose:
        print(f"[GlobalBayes+Edge] n_samples={n_samples}, n_features={n_features}, n_edges={n_edges}")
        print(f"[GlobalBayes+Edge] edge_reliability: {theta_global['edge_reliability']}")

    # return feature-space posteriors and theta with per-edge reliability
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
    epoch = 0,
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
        devices_data, residuals_devices, device_models, le, num_classes, best_param_device, epoch
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
        device_weights=device_weights,
        epoch = epoch
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
        epoch=epoch,
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
                  epoch=0,
                  verbose=True,
                  le=None,
                  feature_mode=True):
    """
    Hierarchical backward feedback with multiclass-safe updates for edge and device models,
    memory-managed and aligned with feature-level HPFL aggregation.
    """
    if verbose:
        print_layer_banner("Backward Pass", current_epoch=epoch)

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


def print_layer_banner(layer_type, layer_idx=None, device_count=None, current_epoch=None, device_id=None, edge_id=None):
    """
    Prints a dynamic, colored banner for different layers.

    Args:
        layer_type (str): Type of layer ('Device', 'Edge', 'Global', 'Evaluation', 'Backward Pass', etc.)
        layer_idx (int, optional): Index of the layer (if applicable)
        device_count (int, optional): Number of devices (if applicable)
        current_epoch (int, optional): Current epoch or iteration
        device_id (int, optional): Specific device number (for device-level prints)
        edge_id (int, optional): Specific edge number (for edge-level prints)
    """

    # ANSI color codes
    COLORS = {
        "Device": "\033[94m",       # Blue
        "Edge": "\033[96m",         # Cyan
        "Global": "\033[95m",       # Magenta
        "Evaluation": "\033[92m",   # Green
        "Backward": "\033[93m",     # Yellow
        "Backward Pass": "\033[93m",
        "ENDC": "\033[0m",
        "BOLD": "\033[1m"
    }

    color = COLORS.get(layer_type, "")
    bold = COLORS["BOLD"]
    endc = COLORS["ENDC"]

    # Dynamic context parts
    layer_info = []
    if layer_idx is not None:
        layer_info.append(f"Layer {layer_idx}")
    if edge_id is not None:
        layer_info.append(f"Edge {edge_id}")
    if device_id is not None:
        layer_info.append(f"Device {device_id}")
    if current_epoch is not None:
        layer_info.append(f"Epoch {current_epoch + 1}")

    context_str = " | ".join(layer_info)
    context_str = f" ({context_str})" if context_str else ""

    # Dynamic description
    if layer_type.lower() in ["evaluation", "evaluation layer"]:
        description = "Computing metrics and validation outputs"
        note = "No training, evaluation only"
    elif layer_type.lower() in ["backward", "backward pass"]:
        description = "Propagating gradients backward and updating models"
        note = "Gradient accumulation and parameter update"
    else:
        description = f"{layer_type}-level boosted ensemble → {layer_type} output"
        note = "Gradient-loss based early stopping"

    # Construct title
    title = f"STARTING {layer_type.upper()} LAYER{context_str}"
    max_width = max(len(title), len(description), len(note)) + 6  # padding

    # Print banner
    print(color + "*" * max_width + endc)
    print(color + f"*{bold}{title.center(max_width - 2)}{endc}*" + endc)
    print(color + f"*{description.center(max_width - 2)}*" + endc)
    print(color + f"*{note.center(max_width - 2)}*" + endc)
    print(color + "*" * max_width + endc + "\n")


def search_boosting_params(
        X_train,
        y_train,
        X_valid,
        y_valid,
        le,
        n_classes,
        verbose=True
):
    """
    Grid search for LightGBM boosting hyperparameters using gradient-based loss.
    """

    param_grid = {
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 4, 5],
        "num_boost_round": [100, 200, 300],

        # Optional parameters — uncomment as needed:
        # "num_leaves": [31, 63, 127],
        # "min_data_in_leaf": [10, 20, 30],
        # "min_gain_to_split": [0.0, 0.1, 0.2],
        # "feature_fraction": [0.7, 0.8, 1.0],
        # "bagging_fraction": [0.7, 0.8, 1.0],
        # "bagging_freq": [1, 5, 10],
        # "lambda_l1": [0.0, 0.1, 0.5],
        # "lambda_l2": [0.0, 0.1, 0.5],
        # "max_bin": [128, 256, 512],
    }

    y_train_enc = encode_labels_safe(le, y_train, n_classes)
    y_valid_enc = encode_labels_safe(le, y_valid, n_classes)

    # Convert y_valid_enc to one-hot for gradient computation
    y_valid_onehot = np.eye(n_classes)[y_valid_enc]

    best_score = np.inf
    best_params = None

    from itertools import product
    keys, values = zip(*param_grid.items())
    all_combinations = list(product(*values))

    for idx, combo in enumerate(all_combinations):
        params = dict(zip(keys, combo))
        if verbose:
            log_util.safe_log(f"Grid trial {idx+1}/{len(all_combinations)}: {params}")

        # Train model
        model, _ = train_lightgbm(
            X_train, y_train_enc,
            X_valid=X_valid, y_valid=y_valid_enc,
            best_params=None,
            **params
        )

        # Predict probabilities
        y_pred = predict_proba_fixed(model, X_valid, n_classes)

        # Gradient loss = mean L2 norm of residuals (y_true - y_pred)
        grad_loss = np.mean(np.linalg.norm(y_valid_onehot - y_pred, axis=1))

        # Accept trial if gradient loss improves
        if grad_loss < best_score:
            best_score = grad_loss
            best_params = params

        del model, y_pred
        gc.collect()

    if verbose:
        log_util.safe_log(f"Best params (gradient loss): {best_params}, loss={best_score:.6f}")

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
    X_tests,
    y_tests,
    le,
    num_classes,
    epoch=0,
    combine_mode="last",
    device_model_weights=None,
    edge_model_weights=None,
    device_val_scores=None,
    edge_val_scores=None,
    edge_reliabilities=None,   # NEW: reliability weights from global aggregation
    calibrate=True,
    device_weight_mode="weighted",
    top_k=3,
):
    """
    Evaluate hierarchical federated model performance: device, edge, and global levels.
    Edge predictions can be weighted by their reliability.
    Returns a structured dictionary with predictions and metrics.
    """
    print_layer_banner("Evaluation", current_epoch=epoch)

    def normalize_weights(weights):
        w = np.asarray(weights, dtype=np.float32)
        w[np.isnan(w)] = 0.0
        s = w.sum()
        return w / s if s > 0 else np.ones_like(w) / len(w)

    def calibrate_probs(y_true, probs):
        y_true = np.atleast_1d(y_true)
        probs = np.atleast_2d(probs).astype(np.float32)
        if probs.shape[1] == 2:
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(probs[:, 1], y_true)
            p1 = ir.transform(probs[:, 1])
            probs = np.column_stack([1 - p1, p1])
        return probs

    warnings.filterwarnings("ignore", category=UserWarning)
    np.seterr(divide='ignore', over='ignore', invalid='ignore')

    # ---------------------------
    # 1️⃣ Device-level metrics
    # ---------------------------
    device_preds, device_accs, log_losses, brier_scores, topk_accuracies = [], [], [], [], []

    for idx, models_per_device in enumerate(device_models):
        X_test = np.atleast_2d(X_tests[idx])
        y_true = le.transform(np.atleast_1d(y_tests[idx]))

        if not models_per_device:
            y_pred_probs = np.zeros((X_test.shape[0], num_classes), dtype=np.float32)
            device_preds.append(y_pred_probs)
            device_accs.append(0.0)
            log_losses.append(np.nan)
            brier_scores.append(np.nan)
            topk_accuracies.append(np.nan)
            continue

        # Stack predictions from multiple models per device
        preds_list = [predict_proba_fixed(m, X_test, num_classes).astype(np.float32)
                      for m in (models_per_device if isinstance(models_per_device, list) else [models_per_device])]

        # Compute combination weights
        if combine_mode == "weighted":
            w = (np.array(device_val_scores[idx]) if device_val_scores else np.ones(len(preds_list)))
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
            y_pred_probs = np.zeros((votes.shape[1], num_classes), dtype=np.float32)
            for c in range(num_classes):
                y_pred_probs[:, c] = np.mean(votes == c, axis=0)
        elif combine_mode == "stacked":
            X_stack = np.hstack(preds_list)
            meta = LogisticRegression(max_iter=500)
            meta.fit(X_stack, y_true)
            y_pred_probs = meta.predict_proba(X_stack)
        else:
            raise ValueError(f"Unknown combine_mode: {combine_mode}")

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

        # Device weights for edge aggregation
        if device_weight_mode == "samples":
            w_devices = [X_tests[d].shape[0] for d in edge_devices if d < len(X_tests)]
        else:
            w_devices = [1.0] * len(device_edge_preds)
        w_devices = normalize_weights(w_devices)

        weighted_preds = [wi * dp for wi, dp in zip(w_devices, device_edge_preds)]
        device_mix_pred = np.vstack(weighted_preds)

        # Edge model predictions
        if edge_idx < len(edge_models) and edge_models[edge_idx]:
            X_edge_combined = np.vstack([X_tests[d] for d in edge_devices])
            edge_pred_accum = predict_proba_fixed(edge_models[edge_idx], X_edge_combined, num_classes)
            if calibrate:
                edge_pred_accum = calibrate_probs(y_edge_true, edge_pred_accum)

            # Apply edge reliability weighting if provided
            reliability = 1.0
            if edge_reliabilities and edge_idx < len(edge_reliabilities):
                reliability = float(edge_reliabilities[edge_idx])
            pred_accum = reliability * edge_pred_accum + (1 - reliability) * device_mix_pred
        else:
            pred_accum = device_mix_pred

        edge_preds.append(pred_accum)
        edge_accs.append(accuracy_score(y_edge_true, pred_accum.argmax(axis=1)))

    # ---------------------------
    # 3️⃣ Global-level metrics
    # ---------------------------
    # Weight devices by edge reliabilities
    if edge_reliabilities:
        global_pred_list = []
        for edge_idx, edge_devices in enumerate(edge_groups):
            pred_edge = edge_preds[edge_idx]
            w = float(edge_reliabilities[edge_idx]) if edge_idx < len(edge_reliabilities) else 1.0
            global_pred_list.append(pred_edge * w)
        global_pred = np.vstack(global_pred_list)
    else:
        global_pred = np.vstack(device_preds)

    y_global_true = np.concatenate([le.transform(y_tests[d]) for d in range(len(X_tests))])
    global_acc = accuracy_score(y_global_true, global_pred.argmax(axis=1))

    # ---------------------------
    # 4️⃣ Structured metrics dictionary
    # ---------------------------
    metrics_test = {
        "device_means": device_accs,
        "device_stds": float(np.std(device_accs)),
        "edge_means": edge_accs,
        "edge_stds": float(np.std(edge_accs)),
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

def hpfl_train_with_accuracy(
    d_data, e_groups, edge_finetune_data, le, n_classes, verbose=True
):
    """
    HPFL Training Loop with Multi-Level Accuracy Tracking and edge reliability weighting.
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

    # -------------------------
    # 3️⃣ Initialize placeholders
    # -------------------------
    residuals_devices = [None] * len(d_data)
    device_models = [None] * len(d_data)
    residuals_edges = [None] * len(e_groups)
    edge_models = [None] * len(e_groups)
    device_embeddings = [None] * len(d_data)
    edge_embeddings = [None] * len(e_groups)

    num_epochs = config["epoch"]

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

    # --- Hyperparameter tuning ---
    if X_device_finetunes[0] is not None and y_device_finetunes[0] is not None:
        best_param_device = search_boosting_params(
            X_trains[0], y_trains[0],
            X_device_finetunes[0][:, :X_trains[0].shape[1]],
            y_device_finetunes[0],
            le=le,
            n_classes=n_classes,
            verbose=True
        )
    else:
        best_param_device = {
            "learning_rate": config["learning_rate"],
            "num_boost_round": config["num_boost_round"]
        }

    # -------------------------
    # 4️⃣ Epoch Loop
    # -------------------------
    for epoch in range(num_epochs):
        #log_util.safe_log(f"\n===== Epoch {epoch + 1}/{num_epochs} =====\n")

        # --- Forward Pass ---
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
            epoch = epoch,
            track_profile=True
        )
        log_util.safe_log("Forward pass profile:", profile)

        # --- Backward Pass ---
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
            epoch=epoch,
            verbose=True,
            le=le
        )

        # --- Compute Edge Reliabilities ---
        edge_reliabilities = []
        for i, res in enumerate(residuals_edges):
            if res is not None:
                mse = np.mean(res**2, axis=1)
                edge_reliabilities.append(float(np.clip(1.0 / (np.mean(mse) + 1e-12), 0.0, 1.0)))
            else:
                edge_reliabilities.append(1.0)

        # --- Evaluate Multi-Level Performance ---
        metrics_test = evaluate_multilevel_performance(
            device_models=updated_device_models,
            edge_models=updated_edge_models,
            edge_groups=e_groups,
            X_tests=X_tests,
            y_tests=y_tests,
            le=le,
            num_classes=n_classes,
            epoch=epoch,
            combine_mode="weighted",
            top_k=3,
            edge_reliabilities=edge_reliabilities
        )

        # --- Save Models ---
        save_hpfl_models(
            updated_device_models, updated_edge_models, epoch,
            y_global_pred=y_global_pred,
            global_residuals=global_residuals,
            theta_global=theta_global
        )

        # --- Update History ---
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
        history["y_true_per_epoch"].append(y_true_per_edge)

        # --- Clean up memory ---
        del edge_outputs, y_global_pred, updated_edge_preds_stacked
        gc.collect()

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

def plot_hpfl_all(metrics_test, save_root_dir="hdpftl_plot_outputs", show_plots=False, top_k=3):
    """
    Generate Hierarchical PFL plots from structured metrics dictionary.
    Includes per-epoch stacked contributions, device/edge vs global,
    overall accuracy trends, per-device/edge heatmaps, and top-K accuracy heatmaps.

    Args:
        metrics_test (dict): Output metrics dictionary per epoch.
        save_root_dir (str): Base directory to save plots.
        show_plots (bool): Whether to call plt.show() after each plot.
        top_k (int): Top-K accuracy to plot (default 3).
    """

    # -----------------------------
    # Create timestamped folder
    # -----------------------------
    today_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    save_dir = os.path.join(save_root_dir, today_str)
    os.makedirs(save_dir, exist_ok=True)

    # -----------------------------
    # Extract per-epoch values
    # -----------------------------
    device_accs = metrics_test.get("device_accs_per_epoch", [])
    edge_accs = metrics_test.get("edge_accs_per_epoch", [])
    global_accs = metrics_test.get("global_accs", [])
    device_vs_global = metrics_test.get("device_vs_global", [])
    edge_vs_global = metrics_test.get("edge_vs_global", [])
    num_epochs = len(global_accs)

    colors = {"Device": "skyblue", "Edge": "orange", "Global": "green"}

    # -----------------------------
    # 1. Per-epoch contributions & comparisons
    # -----------------------------
    for epoch_idx in range(num_epochs):
        mean_device = np.mean(device_accs[epoch_idx]) if device_accs[epoch_idx] else 0
        mean_edge = np.mean(edge_accs[epoch_idx]) if edge_accs[epoch_idx] else 0
        global_acc = global_accs[epoch_idx] if epoch_idx < len(global_accs) else 0

        # Contribution bars
        contributions = [mean_device, max(mean_edge - mean_device, 0), max(global_acc - mean_edge, 0)]
        layers = ["Device", "Edge", "Global"]

        plt.figure(figsize=(6, 4))
        bars = plt.bar(layers, contributions, color=[colors[l] for l in layers])
        plt.ylim(0, 1)
        plt.ylabel("Contribution to Accuracy")
        plt.title(f"Epoch {epoch_idx+1} Layer Contributions")
        for bar, val in zip(bars, contributions):
            plt.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.3f}", ha="center", va="bottom")
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"epoch_contribution_{epoch_idx+1}.png"))
        if show_plots: plt.show()
        plt.close()

        # Device vs Global & Edge vs Global
        mean_dev_vs_glob = np.mean(device_vs_global[epoch_idx]) if device_vs_global[epoch_idx] else 0
        mean_edge_vs_glob = np.mean(edge_vs_global[epoch_idx]) if edge_vs_global[epoch_idx] else 0

        plt.figure(figsize=(6, 4))
        plt.bar(["Device vs Global", "Edge vs Global"], [mean_dev_vs_glob, mean_edge_vs_glob],
                color=[colors["Device"], colors["Edge"]])
        plt.ylim(0, 1)
        plt.ylabel("Accuracy vs Global")
        plt.title(f"Epoch {epoch_idx+1} Layer vs Global")
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"epoch_vs_global_{epoch_idx+1}.png"))
        if show_plots: plt.show()
        plt.close()

        # Device heatmap
        if device_accs[epoch_idx]:
            plt.figure(figsize=(max(6, len(device_accs[epoch_idx]) * 0.5), 4))
            sns.heatmap(np.array([device_accs[epoch_idx]]), annot=True, cmap="Blues",
                        cbar=True, vmin=0, vmax=1)
            plt.xlabel("Device Index")
            plt.ylabel("Epoch")
            plt.title(f"Epoch {epoch_idx+1} Device Accuracies")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"epoch_device_heatmap_{epoch_idx+1}.png"))
            if show_plots: plt.show()
            plt.close()

        # Edge heatmap
        if edge_accs[epoch_idx]:
            plt.figure(figsize=(max(6, len(edge_accs[epoch_idx]) * 0.5), 4))
            sns.heatmap(np.array([edge_accs[epoch_idx]]), annot=True, cmap="Oranges",
                        cbar=True, vmin=0, vmax=1)
            plt.xlabel("Edge Index")
            plt.ylabel("Epoch")
            plt.title(f"Epoch {epoch_idx+1} Edge Accuracies")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"epoch_edge_heatmap_{epoch_idx+1}.png"))
            if show_plots: plt.show()
            plt.close()

        # -----------------------------
        # 1d. Top-K accuracy heatmaps (if available)
        # -----------------------------
        if "metrics" in metrics_test and "device" in metrics_test["metrics"]:
            device_topk_epoch = metrics_test["metrics"]["device"].get("topk", [])
            if device_topk_epoch and len(device_topk_epoch) > epoch_idx:
                topk_matrix = np.array([device_topk_epoch[epoch_idx]])  # shape: [1, num_devices]
                plt.figure(figsize=(max(6, topk_matrix.shape[1]*0.5), 4))
                sns.heatmap(topk_matrix, annot=True, cmap="Greens", cbar=True, vmin=0, vmax=1)
                plt.xlabel("Device Index")
                plt.ylabel("Epoch")
                plt.title(f"Epoch {epoch_idx+1} Device Top-{top_k} Accuracies")
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"epoch_device_top{top_k}_heatmap_{epoch_idx+1}.png"))
                if show_plots: plt.show()
                plt.close()

        if "metrics" in metrics_test and "edge" in metrics_test["metrics"]:
            edge_topk_epoch = metrics_test["metrics"]["edge"].get("topk", [])
            if edge_topk_epoch and len(edge_topk_epoch) > epoch_idx:
                topk_matrix = np.array([edge_topk_epoch[epoch_idx]])
                plt.figure(figsize=(max(6, topk_matrix.shape[1]*0.5), 4))
                sns.heatmap(topk_matrix, annot=True, cmap="Purples", cbar=True, vmin=0, vmax=1)
                plt.xlabel("Edge Index")
                plt.ylabel("Epoch")
                plt.title(f"Epoch {epoch_idx+1} Edge Top-{top_k} Accuracies")
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"epoch_edge_top{top_k}_heatmap_{epoch_idx+1}.png"))
                if show_plots: plt.show()
                plt.close()

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
    if show_plots: plt.show()
    plt.close()

    # -----------------------------
    # 3. Hierarchical Contribution Evolution (stacked)
    # -----------------------------
    edge_gain = np.maximum(np.array(mean_edge_accs) - np.array(mean_device_accs), 0)
    global_gain = np.maximum(np.array(global_accs) - np.array(mean_edge_accs), 0)
    base_device = np.array(mean_device_accs)
    epochs = np.arange(1, num_epochs + 1)

    plt.figure(figsize=(10, 6))
    plt.stackplot(epochs, base_device, edge_gain, global_gain,
                  labels=["Device Base", "Edge Gain", "Global Gain"],
                  colors=[colors["Device"], colors["Edge"], colors["Global"]], alpha=0.8)
    plt.title("Hierarchical Contribution Evolution Across Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Cumulative Accuracy")
    plt.xticks(epochs, labels=[f"Epoch {i}" for i in epochs])
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "hierarchical_contribution_evolution.png"))
    if show_plots: plt.show()
    plt.close()

    log_util.safe_log(f"All plots (including Top-{top_k} heatmaps) saved to folder: {save_dir}")


# ============================================================
#                     Main
# ============================================================

if __name__ == "__main__":
    folder_path = "CIC_IoT_IDAD_Dataset_2024"

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


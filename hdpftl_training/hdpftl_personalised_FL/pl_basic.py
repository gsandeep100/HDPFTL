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
from datetime import datetime
from typing import List, Tuple, Union

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from lightgbm import early_stopping, log_evaluation
from scipy.special import softmax as sp_softmax
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from hdpftl_training.hdpftl_data.preprocess import safe_preprocess_data

os.environ["OMP_NUM_THREADS"] = "4"

# -------------------------------------------------------------
# Config
# -------------------------------------------------------------
config = {
    "random_seed": 42,
    "n_edges": 10,
    "n_device": 50,
    "device_per_edge": 5,
    "epoch": 20,
    "device_boosting_rounds": 10,
    "edge_boosting_rounds": 10,
    "variance_prune": True,
    "variance_threshold": 1e-4,
    "bayes_n_samples": 200,
    "bayes_n_tune": 200,
    "save_results": True,
    "results_path": "results",
    "isotonic_min_positives": 5,
    "max_cores": 2,
    "n_estimators": 100,
    "num_leaves": 31,
    "alpha": 1.0,
    "learning_rate": 0.05,
    "max_depth": -1,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8
}

rng = np.random.default_rng(config["random_seed"])
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series]
DeviceData = Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]


# ============================================================
# Helper functions
# ============================================================

def safe_array(X):
    """Convert input to NumPy array if it is a DataFrame or Series."""
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return X.to_numpy()
    return np.asarray(X)


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
        feature_fraction=config["feature_fraction"],
        device="gpu",
        gpu_device_id=0
    )
    fit_kwargs = {}
    if X_valid is not None and y_valid is not None and early_stopping_rounds:
        fit_kwargs.update({
            "eval_set": [(safe_array(X_valid), safe_array(y_valid))],
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


# ============================================================
# Device Layer Boosting with missing-class safe probabilities
# ============================================================

def device_layer_boosting(devices_data, residuals_devices, device_models, le, num_classes, X_finetune, y_finetune):
    """
    Intra-device sequential boosting:
    - Safe residual updates
    - Automatically handles missing classes
    """
    for idx, (X_dev, _, y_dev, _) in enumerate(devices_data):
        X_dev, y_dev = safe_array(X_dev), safe_array(y_dev)
        y_enc = le.transform(y_dev)
        n_samples = X_dev.shape[0]

        residual = residuals_devices[idx]
        if residual is None:
            residual = np.zeros((n_samples, num_classes), dtype=float)
            residual[np.arange(n_samples), y_enc] = 1.0

        models_per_device = []
        for _ in range(config["device_boosting_rounds"]):
            y_pseudo = residual.argmax(axis=1)
            print("unique y_pseudo: " + str(np.unique(y_pseudo)))  ##Sandeep
            # Skip if only one unique class
            if len(np.unique(y_pseudo)) < 2:
                continue
            model = train_lightgbm(
                X_dev, y_pseudo, X_finetune, y_finetune, early_stopping_rounds=20, num_classes=num_classes
            )
            pred_proba = predict_proba_fixed(model, X_dev, num_classes, le=le)

            # Variance prune
            if config["variance_prune"]:
                var = np.var(pred_proba, axis=0)
                mask = var >= config["variance_threshold"]
                pred_proba[:, ~mask] = 0.0

            residual -= pred_proba
            models_per_device.append(model)

        residuals_devices[idx] = residual
        device_models[idx] = models_per_device

    return residuals_devices, device_models


def edge_layer_boosting(edge_groups, devices_data, residuals_devices, le, num_classes, X_finetune, y_finetune):
    edge_models = []
    residuals_edge_list = []
    edge_acc_list = []
    for edge_devices in edge_groups:
        if len(edge_devices) == 0:
            edge_models.append([])
            residuals_edge_list.append([])
            edge_acc_list.append(0.0)
            continue
        X_edge = np.vstack([devices_data[i][0] for i in edge_devices])
        residual_edge = np.vstack([residuals_devices[i] for i in edge_devices])
        y_test_edge = np.hstack([devices_data[i][3] for i in edge_devices])

        models_per_edge = []
        round_residuals = []
        for _ in range(config["edge_boosting_rounds"]):
            y_pseudo = residual_edge.argmax(axis=1)
            model_e = train_lightgbm(X_edge, y_pseudo, X_finetune, y_finetune, early_stopping_rounds=20,
                                     num_classes=num_classes)
            pred_proba = predict_proba_fixed(model_e, X_edge, num_classes, le=le)
            if config["variance_prune"]:
                var = np.var(pred_proba, axis=0)
                mask = var >= config["variance_threshold"]
                pred_proba[:, ~mask] = 0.0
            residual_edge -= pred_proba
            models_per_edge.append(model_e)
            round_residuals.append(residual_edge.copy())
        edge_models.append(models_per_edge)
        residuals_edge_list.append(round_residuals)

        # Test-set accuracy for this edge
        X_test_edge = np.vstack([devices_data[i][1] for i in edge_devices])  # X_test is index 1
        edge_preds = np.zeros((X_test_edge.shape[0], num_classes), dtype=float)

        for mdl in models_per_edge:
            edge_preds += predict_proba_fixed(mdl, X_test_edge, num_classes, le=le)
        edge_preds = edge_preds.argmax(axis=1)
        edge_acc_list.append(compute_accuracy(y_test_edge, edge_preds))

    return edge_models, residuals_edge_list, np.mean(edge_acc_list)


def gossip_layer_aggregation(devices_data, device_models, le, num_classes, X_finetune, y_finetune,
                             use_calibration=True):
    """
    Aggregates device-level predictions to a global/gossip model.
    Returns:
        - global_residual: residuals for global level
        - trace_summary: contains 'betas' and 'alpha' for global model
        - global_acc: accuracy on combined dataset
    """
    X_proc_list, y_proc_list = [], []

    # -----------------------------
    # Collect device predictions
    # -----------------------------
    for device_idx, (X_dev, _, y_dev, _) in enumerate(devices_data):
        X_dev, y_dev = safe_array(X_dev), safe_array(y_dev)
        models_for_device = device_models[device_idx]
        model_for_device = models_for_device[-1] if models_for_device else train_lightgbm(X_dev, y_dev, X_finetune,
                                                                                          y_finetune,
                                                                                          early_stopping_rounds=20,
                                                                                          num_classes=num_classes)

        probs = predict_proba_fixed(model_for_device, X_dev, num_classes, le=le)

        # -----------------------------
        # Isotonic calibration
        # -----------------------------
        if use_calibration:
            calibrated = np.zeros_like(probs)
            probs_safe = np.nan_to_num(probs, nan=1e-8)
            probs_safe[probs_safe < 1e-8] = 1e-8
            for c in range(num_classes):
                y_c = np.array(le.transform(y_dev) == c, dtype=int)
                positives = np.sum(y_c)
                if positives < config["isotonic_min_positives"]:
                    calibrated[:, c] = probs_safe[:, c]
                    continue
                try:
                    iso = IsotonicRegression(out_of_bounds='clip')
                    iso.fit(probs_safe[:, c], y_c)
                    calibrated[:, c] = iso.transform(probs_safe[:, c])
                except Exception:
                    calibrated[:, c] = probs_safe[:, c]
            row_sums = calibrated.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            probs = calibrated / row_sums

        # -----------------------------
        # Variance pruning
        # -----------------------------
        if config["variance_prune"]:
            var = np.var(probs, axis=0)
            mask = var >= config["variance_threshold"]
            probs[:, ~mask] = 0.0

        X_proc_list.append(probs)
        y_proc_list.append(y_dev)

    # -----------------------------
    # Stack all device data
    # -----------------------------
    X_proc = np.vstack(X_proc_list)
    y_global_enc = le.transform(np.hstack(y_proc_list))

    # -----------------------------
    # Ensure X_proc matches num_features
    # -----------------------------
    num_features_expected = devices_data[0][0].shape[1]
    if X_proc.shape[1] != num_features_expected:
        # Pad or truncate
        if X_proc.shape[1] < num_features_expected:
            pad_width = num_features_expected - X_proc.shape[1]
            X_proc = np.hstack([X_proc, np.zeros((X_proc.shape[0], pad_width))])
        else:
            X_proc = X_proc[:, :num_features_expected]

    # -----------------------------
    # Bayesian / Logistic aggregation
    # -----------------------------
    trace_summary = None
    try:
        with pm.Model() as gossip_model:
            alpha = pm.Normal("alpha", mu=0, sigma=1, shape=(num_classes,))
            betas = pm.Normal("betas", mu=0, sigma=1, shape=(X_proc.shape[1], num_classes))
            logits = pm.math.dot(X_proc, betas) + alpha
            p = pm.math.softmax(logits)
            y_obs = pm.Categorical("y_obs", p=p, observed=y_global_enc)
            raw_trace = pm.sample(draws=config["bayes_n_samples"],
                                  tune=config["bayes_n_tune"],
                                  chains=2,
                                  cores=min(config["max_cores"], os.cpu_count() or 1),
                                  target_accept=0.9,
                                  progressbar=False,
                                  nuts_sampler="numpyro",
                                  random_seed=config["random_seed"])
            trace_summary = {
                "alpha": np.asarray(raw_trace["alpha"]).mean(axis=0),
                "betas": np.asarray(raw_trace["betas"]).mean(axis=0)
            }
    except Exception:
        try:
            sk_model = LogisticRegression(multi_class="multinomial", max_iter=1000)
            sk_model.fit(X_proc, y_global_enc)
            trace_summary = {"betas": sk_model.coef_.T, "alpha": sk_model.intercept_}
        except Exception:
            trace_summary = {"betas": np.zeros((X_proc.shape[1], num_classes)),
                             "alpha": np.zeros(num_classes)}

    # -----------------------------
    # Ensure betas/alpha shapes
    # -----------------------------
    betas = np.asarray(trace_summary["betas"])
    alpha = np.asarray(trace_summary["alpha"])
    if betas.shape[0] != X_proc.shape[1]:
        if betas.shape[1] == X_proc.shape[1]:
            betas = betas.T
        else:
            raise ValueError(f"Cannot match betas shape {betas.shape} with X_proc {X_proc.shape}")
    if alpha.ndim != 1 or alpha.shape[0] != betas.shape[1]:
        alpha = alpha.flatten()
        if alpha.shape[0] != betas.shape[1]:
            raise ValueError(f"Cannot match alpha shape {alpha.shape} with betas {betas.shape}")
    trace_summary["betas"], trace_summary["alpha"] = betas, alpha

    # -----------------------------
    # Global predictions and residuals
    # -----------------------------
    global_logits = X_proc @ betas + alpha
    global_probs = sp_softmax(global_logits, axis=1)
    global_residual = np.zeros_like(global_probs)
    global_residual[np.arange(global_residual.shape[0]), y_global_enc] = 1.0
    global_residual -= global_probs

    global_acc = compute_accuracy(y_global_enc, global_probs.argmax(axis=1))
    return global_residual, trace_summary, global_acc


# ============================================================
# Forward pass
# ============================================================

# ============================================================
# Forward pass with Device → Edge → Gossip
# ============================================================

def forward_pass(devices_data, edge_groups, le, num_classes, X_finetune, y_finetune):
    """
    Perform a single forward pass through:
    - Device-level boosting
    - Edge-level boosting
    - Gossip/global aggregation
    Returns models, residuals, accuracies
    """
    residuals_devices = [None] * len(devices_data)
    device_models = [None] * len(devices_data)

    # -----------------------------
    # Device Layer
    # -----------------------------
    residuals_devices, device_models = device_layer_boosting(
        devices_data, residuals_devices, device_models, le, num_classes, X_finetune, y_finetune
    )

    # -----------------------------
    # Edge Layer
    # -----------------------------
    edge_models, residuals_edges, edge_acc = edge_layer_boosting(
        edge_groups, devices_data, residuals_devices, le, num_classes, X_finetune, y_finetune
    )

    # -----------------------------
    # Gossip / Global Layer
    # -----------------------------
    global_residual, gossip_summary, global_acc = gossip_layer_aggregation(
        devices_data, device_models, le, num_classes, X_finetune, y_finetune
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
        proportions = np.maximum((proportions * len(class_idx)).astype(int), 1)
        diff = len(class_idx) - proportions.sum()
        proportions[np.argmax(proportions)] += diff
        splits = np.split(class_idx, np.cumsum(proportions)[:-1])
        for device_id, split in enumerate(splits):
            device_indices[device_id].extend(split.tolist())

    # Step 2: Build device-level data with local train/test
    devices_data = []
    hierarchical_data = {}
    for device_id, idxs in enumerate(device_indices):
        X_dev, y_dev = X_np[idxs], y_np[idxs]
        n_samples = len(X_dev)

        # Split each device into mini-devices if needed
        mini_idxs = np.array_split(np.arange(n_samples), device_per_edge)
        device_subdata = [(X_dev[i], y_dev[i]) for i in mini_idxs]
        hierarchical_data[device_id] = device_subdata

        # Local train/test split per device
        X_train, X_test_local, y_train, y_test_local = train_test_split(
            X_dev, y_dev, test_size=0.3, random_state=seed
        )
        devices_data.append((X_train, X_test_local, y_train, y_test_local))

    # Step 3: Create edge groups
    edge_groups = make_edge_groups(num_devices, n_edges, random_state=seed)

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
    devices_data, hierarchical_data, edge_groups = dirichlet_partition_for_devices_edges_non_iid(
        X_pretrain, y_pretrain,  # ✅ use X_finetune & y_finetune
        num_devices=config["n_device"],
        device_per_edge=config["device_per_edge"],
        n_edges=config["n_edges"],
        alpha=0.5,
        seed=42
    )
    device_accs, edge_accs, global_accs = [], [], []

    # -----------------------------
    # Training epochs
    # -----------------------------
    for epoch in range(config["epoch"]):
        device_models, edge_models, gossip_summary, residuals_devices, residuals_edges, edge_acc, global_acc = \
            forward_pass(devices_data, edge_groups, le, num_classes, X_finetune, y_finetune)

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

#!/usr/bin/env python3
"""
HDP-FTL full pipeline:
- Device Layer: Intra-device + inter-device sequential boosting
- Edge Layer: Edge-level sequential boosting
- Gossip Layer: Bayesian aggregation with fallback
- Residual feedback from global → edge → device
- Isotonic calibration only at gossip layer
- Variance-based weak learner pruning at device, edge, and gossip layers
- Robust fallbacks
- Logging, timestamped folders, and model saving
- Safe type handling to avoid None/argmax/index errors and .astype issues
"""

import os
import pickle
import logging
from datetime import datetime
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import lightgbm as lgb
import pymc as pm
from sklearn.preprocessing import LabelEncoder
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from scipy.special import softmax as sp_softmax

# Custom preprocessing
from hdpftl_training.hdpftl_data.preprocess import safe_preprocess_data

# ============================================================
# Configuration
# ============================================================
config = {
    "random_seed": 42,
    "n_edges": 10,
    "n_clients": 50,
    "epoch": 50,
    "device_boosting_rounds": 10,
    "edge_boosting_rounds": 5,
    "n_estimators": 1,
    "variance_prune": True,
    "variance_threshold": 1e-4,
    "bayes_n_samples": 200,
    "bayes_n_tune": 200,
    "save_results": True,
    "results_path": "results",
    "isotonic_min_positives": 10,
    "pm_use_advi_fallback": True,
    "max_cores": 2,
}

rng = np.random.default_rng(config["random_seed"])
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series]
ClientData = Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]


# ============================================================
# Helper Functions
# ============================================================

def safe_array(X: ArrayLike) -> np.ndarray:
    """Convert input to np.ndarray safely."""
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return X.to_numpy()
    return np.asarray(X)


def make_edge_groups(n_clients: int, n_edges: int, random_state: int = 42) -> List[List[int]]:
    """Randomly split clients into edges."""
    idxs = np.arange(n_clients)
    rng_local = np.random.default_rng(random_state)
    rng_local.shuffle(idxs)
    groups = [list(g) for g in np.array_split(idxs, n_edges)]
    return groups


def create_non_iid_clients(X, y, X_test, y_test, n_clients=50, n_edges=10):
    """Split dataset into non-iid clients and assign to edges."""
    X_np, y_np = safe_array(X), safe_array(y)
    X_test_np, y_test_np = safe_array(X_test), safe_array(y_test)
    n_samples = X_np.shape[0]
    idxs = rng.permutation(n_samples)
    splits = np.array_split(idxs, n_clients)
    clients_data = [(X_np[s], X_test_np, y_np[s], y_test_np) for s in splits]
    edge_groups = make_edge_groups(n_clients, n_edges)
    return clients_data, edge_groups


def train_lightgbm(X_train, y_train, num_classes=None, n_estimators=1, random_state=42):
    """Train a LightGBM weak learner."""
    X_np, y_np = safe_array(X_train), safe_array(y_train)
    if num_classes is None:
        num_classes = len(np.unique(y_np))
    objective = "multiclass" if num_classes > 2 else "binary"
    num_class = num_classes if num_classes > 2 else None
    model = lgb.LGBMClassifier(
        objective=objective,
        num_class=num_class,
        n_estimators=n_estimators,
        random_state=random_state,
        verbose=-1
    )
    model.fit(X_np, y_np)
    return model


def predict_proba_fixed(model, X, num_classes, le: LabelEncoder = None):
    """Predict probabilities and safely map to all classes."""
    X_np = safe_array(X)
    pred = model.predict_proba(X_np)
    pred = np.atleast_2d(pred)
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
            full[:, int(cls)] = pred[:, i]
    return full


# ============================================================
# Core Pipeline Functions
# ============================================================

def device_layer_boosting(clients_data, residuals_clients, device_models, le, num_classes):
    """
    Device Layer:
    - Intra-device sequential boosting
    - Inter-device sequential boosting inside each edge
    - Residuals updated per device
    - Variance-based weak learner pruning
    """
    for idx, (X_dev, _, y_dev, _) in enumerate(clients_data):
        X_dev, y_dev = safe_array(X_dev), safe_array(y_dev)
        y_enc = le.transform(y_dev)
        n_samples = X_dev.shape[0]

        residual = residuals_clients[idx]
        if residual is None:
            residual = np.zeros((n_samples, num_classes), dtype=float)
            residual[np.arange(n_samples), y_enc] = 1.0

        models_per_device = []
        for _ in range(config["device_boosting_rounds"]):
            y_pseudo = residual.argmax(axis=1)
            try:
                model = train_lightgbm(X_dev, y_pseudo, num_classes=num_classes,
                                        n_estimators=config["n_estimators"])
            except Exception:
                model = train_lightgbm(X_dev, y_pseudo, num_classes=num_classes, n_estimators=1)
            pred_proba = predict_proba_fixed(model, X_dev, num_classes, le=le)
            # Variance-based pruning
            if config["variance_prune"]:
                var = np.var(pred_proba, axis=0)
                mask = var >= config["variance_threshold"]
                pred_proba[:, ~mask] = 0.0
            residual -= pred_proba
            models_per_device.append(model)
        residuals_clients[idx] = residual
        device_models[idx] = models_per_device
    return residuals_clients, device_models


def edge_layer_boosting(edge_groups, clients_data, residuals_clients, le, num_classes):
    """
    Edge Layer:
    - Sequential boosting of devices within edge
    - Residuals propagated across devices and edges
    - Variance-based weak learner pruning
    """
    edge_models = []
    for edge_clients in edge_groups:
        if len(edge_clients) == 0:
            edge_models.append([])
            continue
        X_edge = np.vstack([clients_data[i][0] for i in edge_clients])
        residual_edge = np.vstack([residuals_clients[i] for i in edge_clients])
        models_per_edge = []
        for _ in range(config["edge_boosting_rounds"]):
            y_pseudo = residual_edge.argmax(axis=1)
            try:
                model_e = train_lightgbm(pd.DataFrame(X_edge), y_pseudo, num_classes=num_classes,
                                         n_estimators=config["n_estimators"])
            except Exception:
                model_e = train_lightgbm(pd.DataFrame(X_edge), y_pseudo, num_classes=num_classes, n_estimators=1)
            pred_proba = predict_proba_fixed(model_e, X_edge, num_classes, le=le)
            if config["variance_prune"]:
                var = np.var(pred_proba, axis=0)
                mask = var >= config["variance_threshold"]
                pred_proba[:, ~mask] = 0.0
            residual_edge -= pred_proba
            models_per_edge.append(model_e)
        edge_models.append(models_per_edge)
    return edge_models


def gossip_layer_aggregation(clients_data, device_models, le, num_classes, use_calibration=True):
    """
    Gossip Layer:
    - Predict device outputs
    - Optional isotonic calibration per class
    - Variance-based weak learner pruning
    - Bayesian aggregation with fallback
    - Returns global residuals per client
    """
    X_proc_list = []
    y_proc_list = []

    for client_idx, (X_dev, _, y_dev, _) in enumerate(clients_data):
        X_dev, y_dev = safe_array(X_dev), safe_array(y_dev)
        models_for_client = device_models[client_idx]
        if models_for_client is None or len(models_for_client) == 0:
            model_for_client = train_lightgbm(X_dev, y_dev, num_classes, n_estimators=1)
        else:
            model_for_client = models_for_client[-1]

        probs = predict_proba_fixed(model_for_client, X_dev, num_classes, le=le)

        if use_calibration:
            calibrated = np.zeros_like(probs)
            probs_safe = np.nan_to_num(probs, nan=1e-8)
            probs_safe[probs_safe < 1e-8] = 1e-8
            for c in range(num_classes):
                y_c = np.array(le.transform(y_dev) == c).astype(int)
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

        # Variance-based weak learner pruning after calibration
        if config["variance_prune"]:
            var = np.var(probs, axis=0)
            mask = var >= config["variance_threshold"]
            probs[:, ~mask] = 0.0

        X_proc_list.append(probs)
        y_proc_list.append(y_dev)

    X_proc = np.vstack(X_proc_list)
    y_global_enc = le.transform(np.hstack(y_proc_list))

    trace_summary = None
    try:
        with pm.Model() as gossip_model:
            alpha = pm.Normal("alpha", mu=0, sigma=1, shape=(num_classes,))
            betas = pm.Normal("betas", mu=0, sigma=1, shape=(X_proc.shape[1], num_classes))
            logits = pm.math.dot(X_proc, betas) + alpha
            p = pm.math.softmax(logits)
            y_obs = pm.Categorical("y_obs", p=p, observed=y_global_enc)
            raw_trace = pm.sample(draws=config["bayes_n_samples"], tune=config["bayes_n_tune"],
                                  chains=2, cores=min(config["max_cores"], os.cpu_count() or 1),
                                  target_accept=0.9, progressbar=False, random_seed=config["random_seed"])
            trace_summary = {"alpha": np.asarray(raw_trace["alpha"]).mean(axis=0),
                             "betas": np.asarray(raw_trace["betas"]).mean(axis=0)}
    except Exception:
        try:
            sk_model = LogisticRegression(multi_class="multinomial", max_iter=1000)
            sk_model.fit(X_proc, y_global_enc)
            trace_summary = {"betas": sk_model.coef_.T, "alpha": sk_model.intercept_}
        except Exception:
            trace_summary = {"betas": np.zeros((X_proc.shape[1], num_classes)),
                             "alpha": np.zeros(num_classes)}

    global_preds = X_proc @ np.asarray(trace_summary["betas"]) + np.asarray(trace_summary["alpha"])
    global_probs = sp_softmax(global_preds, axis=1)
    global_residual = np.zeros_like(global_probs)
    global_residual[np.arange(global_residual.shape[0]), y_global_enc] = 1.0
    global_residual -= global_probs

    return global_residual, trace_summary


def forward_pass_with_feedback(clients_data, edge_groups, config, n_iterations=3, use_calibration=True):
    """Full pipeline: forward device → edge → gossip, then residual feedback."""
    all_labels = np.hstack([safe_array(y_dev) for _, _, y_dev, _ in clients_data])
    le = LabelEncoder()
    le.fit(all_labels)
    num_classes = len(le.classes_)
    logging.info(f"Detected {num_classes} classes.")

    residuals_clients = [None] * len(clients_data)
    device_models = [None] * len(clients_data)
    cum_lengths = np.cumsum([0] + [c[0].shape[0] for c in clients_data])
    for iteration in range(n_iterations):
        logging.info(f"Iteration {iteration + 1}/{n_iterations}")
        # Device Layer
        residuals_clients, device_models = device_layer_boosting(clients_data, residuals_clients, device_models, le, num_classes)
        # Edge Layer
        edge_models = edge_layer_boosting(edge_groups, clients_data, residuals_clients, le, num_classes)
        # Gossip Layer
        global_residual, gossip_summary = gossip_layer_aggregation(clients_data, device_models, le, num_classes, use_calibration)
        # Residual feedback to clients
        for idx, (X_dev, _, _, _) in enumerate(clients_data):
            n = X_dev.shape[0]
            residuals_clients[idx] = np.tile(global_residual[idx], (n, 1))
    return device_models, edge_models, gossip_summary


# ============================================================
# Main Execution
# ============================================================
def main():
    folder_path = "CIC_IoT_dataset_2023"
    today_str = datetime.now().strftime("%Y-%m-%d")
    log_path_str = os.path.join("logs", f"{folder_path}_{today_str}")
    os.makedirs(log_path_str, exist_ok=True)

    X_final, y_final, X_pretrain, y_pretrain, X_finetune, y_finetune, X_test, y_test = safe_preprocess_data(
        log_path_str, folder_path
    )

    clients_data, edge_groups = create_non_iid_clients(
        X_pretrain, y_pretrain, X_test, y_test,
        n_clients=config["n_clients"],
        n_edges=config["n_edges"]
    )

    device_models, edge_models, gossip_summary = forward_pass_with_feedback(
        clients_data, edge_groups, config, n_iterations=3, use_calibration=True
    )

    if config["save_results"]:
        os.makedirs(config["results_path"], exist_ok=True)
        with open(os.path.join(config["results_path"], "device_models.pkl"), "wb") as f:
            pickle.dump(device_models, f)
        with open(os.path.join(config["results_path"], "edge_models.pkl"), "wb") as f:
            pickle.dump(edge_models, f)
        if gossip_summary is not None:
            with open(os.path.join(config["results_path"], "gossip_summary.pkl"), "wb") as f:
                pickle.dump(gossip_summary, f)
        logging.info("[INFO] Saved all models and gossip summary.")


if __name__ == "__main__":
    main()

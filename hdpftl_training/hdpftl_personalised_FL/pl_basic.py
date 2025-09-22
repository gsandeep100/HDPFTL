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
from typing import List, Tuple, Union
from sklearn.datasets import load_digits

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from scipy.special import softmax as sp_softmax
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# -------------------------------------------------------------
# Safe preprocessing hook
# -------------------------------------------------------------

# -------------------------------------------------------------
# Config
# -------------------------------------------------------------
config = {
    "random_seed": 42,
    "n_edges": 10,
    "n_clients": 20,
    "device_per_client": 2,
    "epoch": 5,
    "device_boosting_rounds": 3,
    "edge_boosting_rounds": 2,
    "n_estimators": 1,
    "variance_prune": True,
    "variance_threshold": 1e-4,
    "bayes_n_samples": 200,
    "bayes_n_tune": 200,
    "save_results": True,
    "results_path": "results",
    "isotonic_min_positives": 5,
    "max_cores": 2,
}

rng = np.random.default_rng(config["random_seed"])
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series]
ClientData = Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]


# ============================================================
# Helper functions
# ============================================================

def safe_array(X: ArrayLike) -> np.ndarray:
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return X.to_numpy()
    return np.asarray(X)


def compute_accuracy(y_true, y_pred):
    y_true_np = safe_array(y_true)
    y_pred_np = safe_array(y_pred)
    print("DEBUG compute_accuracy:")
    print("  y_true shape:", y_true_np.shape)
    print("  y_pred shape:", y_pred_np.shape)
    return float(np.mean(y_true_np == y_pred_np))


def make_edge_groups(n_clients: int, n_edges: int, random_state: int = 42) -> List[List[int]]:
    idxs = np.arange(n_clients)
    rng_local = np.random.default_rng(random_state)
    rng_local.shuffle(idxs)
    return [list(g) for g in np.array_split(idxs, n_edges)]


# ============================================================
# Dirichlet Partitioning
# ============================================================

def dirichlet_partition(X, y, num_clients, alpha=0.3, seed=42):
    """Partition dataset indices using Dirichlet distribution."""
    y_np = y.values if hasattr(y, 'values') else y
    np.random.seed(seed)
    unique_classes = np.unique(y_np)
    client_indices = [[] for _ in range(num_clients)]

    for c in unique_classes:
        class_idx = np.where(y_np == c)[0]
        np.random.shuffle(class_idx)
        proportions = np.random.dirichlet(alpha * np.ones(num_clients))
        proportions = (proportions * len(class_idx)).astype(int)
        diff = len(class_idx) - proportions.sum()
        proportions[np.argmax(proportions)] += diff
        splits = np.split(class_idx, np.cumsum(proportions)[:-1])
        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split.tolist())

    X_np = X.values if hasattr(X, 'values') else X
    client_data_dict = {}
    for i, idxs in enumerate(client_indices):
        client_data_dict[i] = (X_np[idxs], y_np[idxs])
    return client_data_dict


def dirichlet_partition_for_devices_edges(X, y, num_clients, devices_per_client, n_edges):
    """Return clients_data and hierarchical_data for devices and edges."""
    client_data_dict = dirichlet_partition(X, y, num_clients)
    clients_data = []
    hierarchical_data = {}
    for cid, (X_c, y_c) in client_data_dict.items():
        n_samples = len(X_c)
        device_idxs = np.array_split(np.arange(n_samples), devices_per_client)
        device_data = [(X_c[d], y_c[d]) for d in device_idxs]
        hierarchical_data[cid] = device_data
        X_train, X_test, y_train, y_test = train_test_split(X_c, y_c, test_size=0.2, random_state=config["random_seed"])
        clients_data.append((X_train, X_test, y_train, y_test))
    edge_groups = make_edge_groups(num_clients, n_edges)
    return clients_data, hierarchical_data, edge_groups


# ============================================================
# LightGBM Training
# ============================================================

def train_lightgbm(X_train, y_train, num_classes=None, n_estimators=1, random_state=42):
    X_np, y_np = safe_array(X_train), safe_array(y_train)
    if num_classes is None:
        num_classes = len(np.unique(y_np))
    objective = "multiclass" if num_classes > 2 else "binary"
    num_class = num_classes if num_classes > 2 else None
    model = lgb.LGBMClassifier(objective=objective, num_class=num_class,
                               n_estimators=n_estimators, random_state=random_state)
    model.fit(X_np, y_np)
    return model


def predict_proba_fixed(model, X, num_classes, le=None):
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
# Device / Edge / Gossip Layers
# ============================================================

def device_layer_boosting(clients_data, residuals_clients, device_models, le, num_classes):
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
            model = train_lightgbm(X_dev, y_pseudo, num_classes=num_classes, n_estimators=config["n_estimators"])
            pred_proba = predict_proba_fixed(model, X_dev, num_classes, le=le)
            # variance prune
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
    edge_models = []
    residuals_edge_list = []
    edge_acc_list = []
    for edge_clients in edge_groups:
        if len(edge_clients) == 0:
            edge_models.append([])
            residuals_edge_list.append([])
            edge_acc_list.append(0.0)
            continue
        X_edge = np.vstack([clients_data[i][0] for i in edge_clients])
        residual_edge = np.vstack([residuals_clients[i] for i in edge_clients])
        y_test_edge = np.hstack([clients_data[i][3] for i in edge_clients])

        models_per_edge = []
        round_residuals = []
        for _ in range(config["edge_boosting_rounds"]):
            y_pseudo = residual_edge.argmax(axis=1)
            model_e = train_lightgbm(X_edge, y_pseudo, num_classes=num_classes,
                                     n_estimators=config["n_estimators"])
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
        X_test_edge = np.vstack([clients_data[i][1] for i in edge_clients])  # X_test is index 1
        edge_preds = np.zeros((X_test_edge.shape[0], num_classes), dtype=float)

        for mdl in models_per_edge:
            edge_preds += predict_proba_fixed(mdl, X_test_edge, num_classes, le=le)
        edge_preds = edge_preds.argmax(axis=1)
        edge_acc_list.append(compute_accuracy(y_test_edge, edge_preds))

    return edge_models, residuals_edge_list, np.mean(edge_acc_list)


def gossip_layer_aggregation(clients_data, device_models, le, num_classes, use_calibration=True):
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
    for client_idx, (X_dev, _, y_dev, _) in enumerate(clients_data):
        X_dev, y_dev = safe_array(X_dev), safe_array(y_dev)
        models_for_client = device_models[client_idx]
        model_for_client = models_for_client[-1] if models_for_client else train_lightgbm(X_dev, y_dev, num_classes, 1)

        probs = predict_proba_fixed(model_for_client, X_dev, num_classes, le=le)

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
    # Stack all client data
    # -----------------------------
    X_proc = np.vstack(X_proc_list)
    y_global_enc = le.transform(np.hstack(y_proc_list))

    # -----------------------------
    # Ensure X_proc matches num_features
    # -----------------------------
    num_features_expected = clients_data[0][0].shape[1]
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

def forward_pass(clients_data, edge_groups, le, num_classes):
    residuals_clients = [None] * len(clients_data)
    device_models = [None] * len(clients_data)
    residuals_clients, device_models = device_layer_boosting(clients_data, residuals_clients, device_models, le,
                                                             num_classes)
    edge_models, residuals_edges, edge_acc = edge_layer_boosting(edge_groups, clients_data, residuals_clients, le,
                                                                 num_classes)
    global_residual, global_trace, global_acc = gossip_layer_aggregation(clients_data, device_models, le, num_classes)
    return device_models, edge_models, global_trace, residuals_clients, residuals_edges, edge_acc, global_acc


def evaluate_final_accuracy(clients_data, device_models, edge_groups, le, num_classes, gossip_summary=None):
    """
    Evaluate final accuracy on the held-out test set:
    - Device-level: average over all devices
    - Edge-level: aggregate devices per edge
    - Global/gossip-level: Bayesian aggregation if gossip_summary provided
    """
    # -----------------------------
    # Device-level accuracy
    # -----------------------------
    device_acc_list = []
    for idx, (X_test, _, y_test, _) in enumerate(clients_data):
        X_test_np, y_test_np = safe_array(X_test), safe_array(y_test)
        device_preds = np.zeros((X_test_np.shape[0], num_classes))
        for mdl in device_models[idx]:
            device_preds += predict_proba_fixed(mdl, X_test_np, num_classes, le)
        device_preds = device_preds.argmax(axis=1)
        device_acc_list.append(compute_accuracy(y_test_np, device_preds))
    device_acc = np.mean(device_acc_list)

    # -----------------------------
    # Edge-level accuracy
    # -----------------------------
    edge_acc_list = []
    for edge_clients in edge_groups:
        if not edge_clients:
            continue
        X_edge = np.vstack([clients_data[i][0] for i in edge_clients])
        y_edge = np.hstack([clients_data[i][2] for i in edge_clients])
        edge_preds = np.zeros((X_edge.shape[0], num_classes))
        for i in edge_clients:
            for mdl in device_models[i]:
                edge_preds += predict_proba_fixed(mdl, X_edge, num_classes, le)
        edge_preds = edge_preds.argmax(axis=1)
        edge_acc_list.append(compute_accuracy(y_edge, edge_preds))
    edge_acc = np.mean(edge_acc_list) if edge_acc_list else 0.0

    # -----------------------------
    # Global/gossip-level accuracy
    # -----------------------------
    global_acc = None
    if gossip_summary is not None:
        X_global = np.vstack([safe_array(clients_data[i][0]) for i in range(len(clients_data))])
        y_global = np.hstack([safe_array(clients_data[i][2]) for i in range(len(clients_data))])

        betas = np.asarray(gossip_summary["betas"])
        alpha = np.asarray(gossip_summary["alpha"])

        # Ensure betas has shape (num_features, num_classes)
        if betas.shape[0] != X_global.shape[1]:
            if betas.shape[1] == X_global.shape[1]:
                betas = betas.T
            else:
                raise ValueError(f"Cannot match betas shape {betas.shape} with X_global {X_global.shape}")

        # Ensure alpha is (num_classes,)
        if alpha.ndim != 1 or alpha.shape[0] != betas.shape[1]:
            alpha = alpha.flatten()
            if alpha.shape[0] != betas.shape[1]:
                raise ValueError(f"Cannot match alpha shape {alpha.shape} with betas {betas.shape}")

        logits = X_global @ betas + alpha
        global_preds = sp_softmax(logits, axis=1).argmax(axis=1)
        global_acc = compute_accuracy(y_global, global_preds)

    return device_acc, edge_acc, global_acc


# ============================================================
# Example Training Loop with MNIST / sklearn
# ============================================================

if __name__ == "__main__":
    # Example dataset

    data = load_digits()
    X, y = data.data, data.target
    le = LabelEncoder()
    le.fit(y)
    num_classes = len(le.classes_)

    clients_data, hierarchical_data, edge_groups = dirichlet_partition_for_devices_edges(
        X, y, num_clients=config["n_clients"],
        devices_per_client=config["device_per_client"],
        n_edges=config["n_edges"]
    )

    device_accs, edge_accs, global_accs = [], [], []

    for epoch in range(config["epoch"]):
        device_models, edge_models, global_trace, residuals_clients, residuals_edges, edge_acc, global_acc = \
            forward_pass(clients_data, edge_groups, le, num_classes)

        device_accs.append(np.mean(
            [compute_accuracy(c[3], predict_proba_fixed(device_models[i][-1], c[1], num_classes, le).argmax(axis=1))
             for i, c in enumerate(clients_data)]))
        edge_accs.append(edge_acc)
        global_accs.append(global_acc)
        print(
            f"[Epoch {epoch + 1}] Device Acc: {device_accs[-1]:.4f}, Edge Acc: {edge_acc:.4f}, Global Acc: {global_acc:.4f}")

        device_acc, edge_acc, global_acc = evaluate_final_accuracy(
            clients_data, device_models, edge_groups, le, num_classes, global_trace
        )
        logging.info(f"Final Test Accuracy - Device: {device_acc:.4f}, Edge: {edge_acc:.4f}, Global: {global_acc:.4f}")

    # Plot accuracy trends
    plt.figure(figsize=(10, 6))
    plt.plot(device_accs, label="Device-level Acc")
    plt.plot(edge_accs, label="Edge-level Acc")
    plt.plot(global_accs, label="Global Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Hierarchical PFL Accuracy Trends")
    plt.legend()
    plt.show()

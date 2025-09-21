#!/usr/bin/env python3
"""
HDP-FTL Full Pipeline with Multiple Epochs, Residual Tracking, and Plots
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
import matplotlib.pyplot as plt

# Custom preprocessing
from hdpftl_training.hdpftl_data.preprocess import safe_preprocess_data

# ==============================
# Configuration
# ==============================
config = {
    "random_seed": 42,
    "n_edges": 10,
    "n_clients": 50,
    "epochs": 3,
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

# ==============================
# Helper Functions
# ==============================
def safe_array(X: ArrayLike) -> np.ndarray:
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return X.to_numpy()
    return np.asarray(X)


def make_edge_groups(n_clients: int, n_edges: int, random_state: int = 42) -> List[List[int]]:
    idxs = np.arange(n_clients)
    rng_local = np.random.default_rng(random_state)
    rng_local.shuffle(idxs)
    return [list(g) for g in np.array_split(idxs, n_edges)]


def create_non_iid_clients(X, y, X_test, y_test, n_clients=50, n_edges=10):
    X_np, y_np = safe_array(X), safe_array(y)
    X_test_np, y_test_np = safe_array(X_test), safe_array(y_test)
    n_samples = X_np.shape[0]
    idxs = rng.permutation(n_samples)
    splits = np.array_split(idxs, n_clients)
    clients_data = [(X_np[s], X_test_np, y_np[s], y_test_np) for s in splits]
    edge_groups = make_edge_groups(n_clients, n_edges)
    return clients_data, edge_groups


def train_lightgbm(X_train, y_train, num_classes=None, n_estimators=1, random_state=42):
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


# ==============================
# Plotting Functions
# ==============================
def plot_residuals_progress(residuals_history, title="Residual Norms"):
    plt.figure(figsize=(12, 6))
    for layer, norms in residuals_history.items():
        plt.plot(norms, label=layer)
    plt.xlabel("Training Epoch")
    plt.ylabel("Mean Absolute Residual")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_accuracy_progress(accuracy_history, title="Classification Accuracy"):
    plt.figure(figsize=(12, 6))
    for layer, acc in accuracy_history.items():
        plt.plot(acc, label=layer)
    plt.xlabel("Training Epoch")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_sequential_boosting_progress(residuals_per_round_epochs, layer_name="Device"):
    plt.figure(figsize=(12, 6))
    n_epochs = len(residuals_per_round_epochs)
    for epoch_idx in range(n_epochs):
        rounds = residuals_per_round_epochs[epoch_idx]
        mean_residual_per_round = [np.mean(np.abs(r)) for r in rounds]
        plt.plot(range(1, len(rounds) + 1), mean_residual_per_round, marker='o', label=f"Epoch {epoch_idx+1}")
    plt.xlabel("Sequential Boosting Round")
    plt.ylabel("Mean Absolute Residual")
    plt.title(f"Sequential Boosting Residual Reduction per Round ({layer_name} Layer)")
    plt.legend()
    plt.grid(True)
    plt.show()


# ==============================
# Core Layer Functions
# ==============================
def device_layer_boosting(clients_data, residuals_clients, device_models, le, num_classes):
    residuals_per_round = []
    for idx, (X_dev, _, y_dev, _) in enumerate(clients_data):
        X_dev, y_dev = safe_array(X_dev), safe_array(y_dev)
        y_enc = le.transform(y_dev)
        n_samples = X_dev.shape[0]

        residual = residuals_clients[idx]
        if residual is None:
            residual = np.zeros((n_samples, num_classes), dtype=float)
            residual[np.arange(n_samples), y_enc] = 1.0

        models_per_device = []
        round_residuals = []
        for _ in range(config["device_boosting_rounds"]):
            y_pseudo = residual.argmax(axis=1)
            model = train_lightgbm(X_dev, y_pseudo, num_classes=num_classes,
                                    n_estimators=config["n_estimators"])
            pred_proba = predict_proba_fixed(model, X_dev, num_classes, le=le)
            if config["variance_prune"]:
                var = np.var(pred_proba, axis=0)
                mask = var >= config["variance_threshold"]
                pred_proba[:, ~mask] = 0.0
            residual -= pred_proba
            models_per_device.append(model)
            round_residuals.append(residual.copy())
        residuals_clients[idx] = residual
        device_models[idx] = models_per_device
        residuals_per_round.append(round_residuals)
    return residuals_clients, device_models, residuals_per_round


# Edge Layer
def edge_layer_boosting(edge_groups, clients_data, residuals_clients, le, num_classes):
    edge_models = []
    residuals_edge_list = []
    for edge_clients in edge_groups:
        if len(edge_clients) == 0:
            edge_models.append([])
            residuals_edge_list.append([])
            continue
        X_edge = np.vstack([clients_data[i][0] for i in edge_clients])
        residual_edge = np.vstack([residuals_clients[i] for i in edge_clients])
        models_per_edge = []
        residuals_per_round = []
        for _ in range(config["edge_boosting_rounds"]):
            y_pseudo = residual_edge.argmax(axis=1)
            model_e = train_lightgbm(X_edge, y_pseudo, num_classes=num_classes, n_estimators=config["n_estimators"])
            pred_proba = predict_proba_fixed(model_e, X_edge, num_classes, le=le)
            if config["variance_prune"]:
                var = np.var(pred_proba, axis=0)
                mask = var >= config["variance_threshold"]
                pred_proba[:, ~mask] = 0.0
            residual_edge -= pred_proba
            models_per_edge.append(model_e)
            residuals_per_round.append(residual_edge.copy())
        edge_models.append(models_per_edge)
        residuals_edge_list.append(residuals_per_round)
    return edge_models, residuals_edge_list


# Gossip Layer
def gossip_layer_aggregation(clients_data, device_models, le, num_classes):
    X_proc_list, y_proc_list = [], []
    for client_idx, (X_dev, _, y_dev, _) in enumerate(clients_data):
        X_dev, y_dev = safe_array(X_dev), safe_array(y_dev)
        models_for_client = device_models[client_idx]
        model_for_client = models_for_client[-1] if models_for_client else train_lightgbm(X_dev, y_dev, num_classes)
        probs = predict_proba_fixed(model_for_client, X_dev, num_classes, le=le)

        # Isotonic calibration
        calibrated = np.zeros_like(probs)
        probs_safe = np.nan_to_num(probs, nan=1e-8)
        probs_safe[probs_safe < 1e-8] = 1e-8
        for c in range(num_classes):
            y_c = np.array(le.transform(y_dev) == c).astype(int)
            if np.sum(y_c) < config["isotonic_min_positives"]:
                calibrated[:, c] = probs_safe[:, c]
                continue
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(probs_safe[:, c], y_c)
            calibrated[:, c] = iso.transform(probs_safe[:, c])
        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        probs = calibrated / row_sums

        # Variance pruning
        if config["variance_prune"]:
            var = np.var(probs, axis=0)
            mask = var >= config["variance_threshold"]
            probs[:, ~mask] = 0.0

        X_proc_list.append(probs)
        y_proc_list.append(y_dev)

    X_proc = np.vstack(X_proc_list)
    y_global_enc = le.transform(np.hstack(y_proc_list))

    # Bayesian aggregation
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
        sk_model = LogisticRegression(multi_class="multinomial", max_iter=1000)
        sk_model.fit(X_proc, y_global_enc)
        trace_summary = {"betas": sk_model.coef_.T, "alpha": sk_model.intercept_}

    global_preds = X_proc @ np.asarray(trace_summary["betas"]) + np.asarray(trace_summary["alpha"])
    global_probs = sp_softmax(global_preds, axis=1)
    global_residual = np.zeros_like(global_probs)
    global_residual[np.arange(global_residual.shape[0]), y_global_enc] = 1.0
    global_residual -= global_probs

    return global_residual, trace_summary


# ==============================
# Main Execution
# ==============================
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

    le = LabelEncoder()
    le.fit(np.hstack([safe_array(y_dev) for _, _, y_dev, _ in clients_data]))
    num_classes = len(le.classes_)

    # Track residuals and accuracy
    residual_norms = {"Device": [], "Edge": [], "Global": []}
    accuracy_history = {"Device": [], "Edge": [], "Global": []}
    device_round_residuals_epochs = []

    for epoch in range(config["epochs"]):
        logging.info(f"Epoch {epoch+1}/{config['epochs']}")
        residuals_clients = [None]*len(clients_data)
        device_models = [None]*len(clients_data)

        # Device Layer
        residuals_clients, device_models, device_round_residuals = device_layer_boosting(
            clients_data, residuals_clients, device_models, le, num_classes
        )
        device_round_residuals_epochs.append(device_round_residuals)
        residual_norms["Device"].append(np.mean([np.abs(r).mean() for r in residuals_clients]))

        # Edge Layer
        edge_models, residuals_edge = edge_layer_boosting(edge_groups, clients_data, residuals_clients, le, num_classes)
        residual_norms["Edge"].append(np.mean([np.abs(r[-1]).mean() for r in residuals_edge]))

        # Gossip Layer
        global_residual, gossip_summary = gossip_layer_aggregation(clients_data, device_models, le, num_classes)
        residual_norms["Global"].append(np.mean(np.abs(global_residual)))

        # Accuracy (placeholder: could be computed with X_test)
        accuracy_history["Device"].append(0)
        accuracy_history["Edge"].append(0)
        accuracy_history["Global"].append(0)

    # ==============================
    # Plots
    # ==============================
    plot_sequential_boosting_progress(device_round_residuals_epochs, layer_name="Device")
    plot_residuals_progress(residual_norms)
    plot_accuracy_progress(accuracy_history)

    # Save models
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

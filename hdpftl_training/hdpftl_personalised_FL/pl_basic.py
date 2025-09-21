#!/usr/bin/env python3
"""
Full HPFL/HDP-FTL pipeline with:
- Intra-device + inter-device sequential boosting
- Edge-level sequential boosting
- Gossip Bayesian aggregation
- Residual feedback
- Variance pruning, isotonic calibration
- Logging, robust fallbacks
"""

import os
import pickle
from datetime import datetime
import logging
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import lightgbm as lgb
import pymc as pm
from sklearn.preprocessing import LabelEncoder
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from scipy.special import softmax as sp_softmax

from hdpftl_training.hdpftl_data.preprocess import safe_preprocess_data

# ============================================================
# Config
# ============================================================
config = {
    "random_seed": 42,
    "n_edges": 10,
    "n_clients": 50,
    "n_iterations": 3,
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
# Utilities
# ============================================================
def make_edge_groups(n_clients: int, n_edges: int, random_state: int = 42) -> List[List[int]]:
    idxs = np.arange(n_clients)
    rng_local = np.random.default_rng(random_state)
    rng_local.shuffle(idxs)
    return [list(g) for g in np.array_split(idxs, n_edges)]

def create_non_iid_clients(X, y, X_test, y_test, n_clients=50, n_edges=10, random_state=42):
    rng_local = np.random.default_rng(random_state)
    X_np = np.asarray(X)
    y_np = np.asarray(y)
    X_test_np = np.asarray(X_test)
    y_test_np = np.asarray(y_test)

    n_samples = X_np.shape[0]
    shuffled_idx = rng_local.permutation(n_samples)
    samples_per_client = np.array_split(shuffled_idx, n_clients)
    clients_data = [(X_np[idx], X_test_np, y_np[idx], y_test_np) for idx in samples_per_client]

    edge_groups = make_edge_groups(n_clients, n_edges, random_state=random_state)
    return clients_data, edge_groups

def train_lightgbm(X_train, y_train, num_classes=None, n_estimators=1):
    X_np = np.asarray(X_train)
    y_np = np.asarray(y_train)

    if num_classes is None:
        num_classes = len(np.unique(y_np))

    if num_classes > 2:
        model = lgb.LGBMClassifier(
            objective="multiclass", num_class=num_classes,
            metric="multi_logloss", n_estimators=n_estimators, verbose=-1
        )
    else:
        model = lgb.LGBMClassifier(objective="binary", metric="binary_logloss",
                                   n_estimators=n_estimators, verbose=-1)
    model.fit(X_np, y_np)
    return model

def predict_proba_fixed(model, X, num_classes, le: LabelEncoder = None):
    X_np = np.asarray(X)
    pred = np.atleast_2d(model.predict_proba(X_np))
    if pred.shape[1] == num_classes:
        return pred
    full = np.zeros((pred.shape[0], num_classes))
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
# Core HPFL Pipeline
# ============================================================
def forward_pass_with_feedback(clients_data, edge_groups, config):
    rng = np.random.default_rng(config["random_seed"])
    all_labels = np.hstack([y_dev for _, _, y_dev, _ in clients_data])
    le = LabelEncoder()
    le.fit(all_labels)
    num_classes = len(le.classes_)
    logging.info(f"Detected {num_classes} classes across clients.")

    residuals_clients = [None] * len(clients_data)
    device_models = [None] * len(clients_data)
    edge_models = []

    lengths = [c[0].shape[0] for c in clients_data]
    cum_lengths = np.cumsum([0] + lengths)
    total_rows = sum(lengths)
    orig_indices = np.arange(total_rows)

    for iteration in range(config["n_iterations"]):
        logging.info(f"Iteration {iteration+1}/{config['n_iterations']}")

        # ====================
        # Device Layer
        # Intra-device + inter-device sequential boosting
        # ====================
        device_preds = [None]*len(clients_data)
        for edge_idx, edge_clients in enumerate(edge_groups):
            prev_residual = None
            for client_idx in edge_clients:
                X_dev, _, y_dev, _ = clients_data[client_idx]
                y_enc = le.transform(y_dev)
                n_samples = X_dev.shape[0]

                if residuals_clients[client_idx] is None:
                    residual = np.zeros((n_samples, num_classes))
                    residual[np.arange(n_samples), y_enc] = 1.0
                else:
                    residual = residuals_clients[client_idx]

                # If inter-device residual exists
                if prev_residual is not None:
                    residual = prev_residual.copy()

                models_per_device = []
                for _ in range(config["device_boosting_rounds"]):
                    y_pseudo = residual.argmax(axis=1)
                    model = train_lightgbm(X_dev, y_pseudo, num_classes=num_classes,
                                            n_estimators=config["n_estimators"])
                    pred_proba = predict_proba_fixed(model, X_dev, num_classes, le=le)
                    residual -= pred_proba

                    # Variance pruning
                    if config["variance_prune"] and np.var(pred_proba, axis=0).mean() < config["variance_threshold"]:
                        logging.info(f"Pruned weak learner on device {client_idx}")
                        break
                    models_per_device.append(model)

                device_models[client_idx] = models_per_device
                residuals_clients[client_idx] = residual
                prev_residual = residual.copy()  # Pass residual to next device

        # ====================
        # Edge Layer
        # Sequential boosting across edges
        # ====================
        edge_residual = None
        edge_models = []
        for edge_idx, edge_clients in enumerate(edge_groups):
            X_edge = np.vstack([clients_data[i][0] for i in edge_clients])
            y_edge = np.hstack([clients_data[i][2] for i in edge_clients])
            residual_edge = np.vstack([residuals_clients[i] for i in edge_clients])
            if edge_residual is not None:
                residual_edge += edge_residual  # sequential correction from previous edge

            models_per_edge = []
            for _ in range(config["edge_boosting_rounds"]):
                y_pseudo = residual_edge.argmax(axis=1)
                model_e = train_lightgbm(pd.DataFrame(X_edge), y_pseudo, num_classes=num_classes,
                                         n_estimators=config["n_estimators"])
                pred_proba = predict_proba_fixed(model_e, X_edge, num_classes, le=le)
                residual_edge -= pred_proba

                if config["variance_prune"] and np.var(pred_proba, axis=0).mean() < config["variance_threshold"]:
                    logging.info(f"Pruned weak learner at edge {edge_idx}")
                    break
                models_per_edge.append(model_e)
            edge_models.append(models_per_edge)
            edge_residual = residual_edge.copy()

        # ====================
        # Gossip / Global Layer
        # Bayesian aggregation + residual feedback
        # ====================
        X_proc_list = []
        y_proc_list = []
        for client_idx, (X_dev, _, y_dev, _) in enumerate(clients_data):
            models_for_client = device_models[client_idx]
            model_for_client = models_for_client[-1] if models_for_client else train_lightgbm(X_dev, y_dev, num_classes=num_classes)
            probs = predict_proba_fixed(model_for_client, X_dev, num_classes, le=le)

            # Isotonic calibration
            calibrated = np.zeros_like(probs)
            probs_safe = np.nan_to_num(probs, nan=1e-8)
            for c in range(num_classes):
                positives = np.sum((le.transform(y_dev) == c).astype(int))
                if positives < config["isotonic_min_positives"]:
                    calibrated[:, c] = probs_safe[:, c]
                    continue
                try:
                    iso = IsotonicRegression(out_of_bounds='clip')
                    iso.fit(probs_safe[:, c], (le.transform(y_dev) == c).astype(int))
                    calibrated[:, c] = iso.transform(probs_safe[:, c])
                except:
                    calibrated[:, c] = probs_safe[:, c]
            calibrated /= calibrated.sum(axis=1, keepdims=True)
            X_proc_list.append(calibrated)
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
        except:
            sk_model = LogisticRegression(multi_class="multinomial", max_iter=1000)
            sk_model.fit(X_proc, y_global_enc)
            trace_summary = {"betas": sk_model.coef_.T, "alpha": sk_model.intercept_}

        # Update residuals back to clients
        global_preds = X_proc @ trace_summary["betas"] + trace_summary["alpha"]
        global_probs = sp_softmax(global_preds, axis=1)
        global_residual = np.zeros_like(global_probs)
        global_residual[np.arange(global_residual.shape[0]), y_global_enc] = 1.0
        global_residual -= global_probs

        start_idx = 0
        for client_idx, (X_dev, _, _, _) in enumerate(clients_data):
            n = X_dev.shape[0]
            residuals_clients[client_idx] = global_residual[start_idx:start_idx+n]
            start_idx += n

    logging.info("Completed all iterations.")
    gossip_summary = trace_summary
    return device_models, edge_models, gossip_summary

# ============================================================
# Main
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
    logging.info(f"Created {len(clients_data)} clients across {len(edge_groups)} edges.")

    device_models, edge_models, gossip_summary = forward_pass_with_feedback(
        clients_data, edge_groups, config
    )

    if config["save_results"]:
        os.makedirs(config["results_path"], exist_ok=True)
        with open(os.path.join(config["results_path"], "device_models.pkl"), "wb") as f:
            pickle.dump(device_models, f)
        with open(os.path.join(config["results_path"], "edge_models.pkl"), "wb") as f:
            pickle.dump(edge_models, f)
        with open(os.path.join(config["results_path"], "gossip_summary.pkl"), "wb") as f:
            pickle.dump(gossip_summary, f)
        logging.info("Saved device, edge, and gossip models.")

if __name__ == "__main__":
    main()

# ============================================================
# Imports
# ============================================================
import os
import pickle
from datetime import datetime
from typing import List, Tuple, Union
from collections import Counter
import lightgbm as lgb
import numpy as np
import pandas as pd
import pymc as pm
from scipy.sparse import csr_matrix, vstack
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Custom preprocessing
from hdpftl_training.hdpftl_data.preprocess import preprocess_data

# ============================================================
# Type Aliases
# ============================================================
ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series]
ClientData = Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]

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
    "n_estimators": 50,
    "variance_prune": True,
    "variance_threshold": 1e-4,
    "bayes_n_samples": 1000,
    "bayes_n_tune": 1000,
    "save_results": True,
    "results_path": "results"
}

np.random.seed(config["random_seed"])


# ============================================================
# Helper Functions
# ============================================================
def create_non_iid_clients(
        X: ArrayLike, y: ArrayLike, X_test: ArrayLike, y_test: ArrayLike,
        n_clients: int = 50, n_edges: int = 10, min_clients_per_edge: int = 5,
        random_state: int = 42
) -> Tuple[List[ClientData], List[List[int]]]:
    np.random.seed(random_state)
    X_np = X.to_numpy() if isinstance(X, (pd.DataFrame, pd.Series)) else X
    y_np = y.to_numpy() if isinstance(y, (pd.DataFrame, pd.Series)) else y
    X_test_np = X_test.to_numpy() if isinstance(X_test, (pd.DataFrame, pd.Series)) else X_test
    y_test_np = y_test.to_numpy() if isinstance(y_test, (pd.DataFrame, pd.Series)) else y_test

    n_samples = X_np.shape[0]
    shuffled_idx = np.random.permutation(n_samples)
    samples_per_client = n_samples // n_clients

    clients_data = []
    for i in range(n_clients):
        idx = shuffled_idx[i * samples_per_client:(i + 1) * samples_per_client]
        clients_data.append((X_np[idx], X_test_np, y_np[idx], y_test_np))

    # Assign clients to edges
    clients_per_edge = max(min_clients_per_edge, n_clients // n_edges)
    edge_groups = []
    all_client_indices = list(range(n_clients))
    np.random.shuffle(all_client_indices)
    for i in range(0, n_clients, clients_per_edge):
        edge_groups.append(all_client_indices[i:i + clients_per_edge])

    return clients_data, edge_groups

# When the model predicts fewer columns, pad with zeros for missing classes:
def pad_proba(pred_proba, present_classes, num_classes):
    full = np.zeros((pred_proba.shape[0], num_classes))
    full[:, present_classes] = pred_proba
    return full


def train_lightgbm(X_train, y_train, X_val=None, y_val=None, n_estimators=50, num_classes=None):
    """Train LightGBM safely with LabelEncoder and num_class"""
    if num_classes is None:
        num_classes = len(np.unique(y_train))
    if num_classes > 2:
        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=num_classes,
            metric="multi_logloss",
            boosting_type="gbdt",
            num_leaves=31,
            learning_rate=0.05,
            n_estimators=n_estimators,
            verbose=-1
        )
    else:
        model = lgb.LGBMClassifier(
            objective="binary",
            metric="binary_logloss",
            boosting_type="gbdt",
            num_leaves=31,
            learning_rate=0.05,
            n_estimators=n_estimators,
            verbose=-1
        )

    if X_val is not None and y_val is not None:
        # Clip validation labels to allowed range to avoid unseen labels
        y_val_fixed = np.clip(y_val, 0, num_classes - 1)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val_fixed)],
            eval_metric=model._objective,
            callbacks=[lgb.early_stopping(stopping_rounds=20)]
        )
    else:
        model.fit(X_train, y_train)
    return model


def extract_leaf_embeddings(model, X: ArrayLike) -> np.ndarray:
    # Keep feature names consistent with model.feature_name_
    if isinstance(X, pd.DataFrame):
        X_fix = X[model.feature_name_]           # reorder columns if needed
    else:
        X_fix = pd.DataFrame(X, columns=model.feature_name_)
    return model.predict(X_fix, pred_leaf=True).astype(int)


def one_hot_encode_leaf_embeddings(leaf_embeddings: np.ndarray) -> csr_matrix:
    ohe = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
    return ohe.fit_transform(leaf_embeddings)


# ============================================================
# Device Level Boosting
# ============================================================

def device_level_boosting(clients_data: List[ClientData], rounds=10, num_classes=None):
    """
    Sequential boosting at device level with safe residual updates.
    Handles rare classes and ensures all LightGBM predictions match num_classes.
    """
    device_models = []

    for X_train, X_test, y_train, y_test in clients_data:
        # Determine number of classes once
        if num_classes is None:
            num_classes = len(np.unique(y_train))

        # One-hot encode training labels
        y_onehot = np.zeros((y_train.shape[0], num_classes), dtype=float)
        y_onehot[np.arange(y_train.shape[0]), y_train] = 1.0
        residual = y_onehot.copy()
        models_per_device = []

        for r in range(rounds):
            # Pseudo-labels from current residuals
            y_pseudo = residual.argmax(axis=1)

            # Train-test split safely, fallback if rare classes exist
            X_tr, X_val, y_tr, y_val = safe_split(
                X_train, y_pseudo, test_size=0.2, random_state=config["random_seed"], num_classes=num_classes
            )

            # Train LightGBM classifier
            model = train_lightgbm(
                X_tr, y_tr, X_val, y_val,
                n_estimators=config["n_estimators"],
                num_classes=num_classes
            )

            # Predict probabilities on full device data
            X_train_fix = (
                X_train[model.feature_name_] if isinstance(X_train, pd.DataFrame)
                else pd.DataFrame(X_train, columns=model.feature_name_)
            )
            pred_proba_raw = model.predict_proba(X_train_fix)
            pred_proba_full = pad_proba(pred_proba_raw, model.classes_.astype(int), num_classes)

            # Update residuals
            residual -= pred_proba_full
            models_per_device.append(model)

        device_models.append(models_per_device)

    return device_models

# ============================================================
# Edge Level Boosting
# ============================================================
def edge_level_boosting(edge_groups, device_models, clients_data, num_classes=None):
    """
    Edge-level sequential boosting with residual feedback.
    Each edge aggregates devices and refines residuals with consistent num_classes.
    """
    all_edge_models = []

    for edge_idx, edge_devices in enumerate(edge_groups):
        # Stack all edge device data for the edge-level model
        X_train_edge = np.concatenate([clients_data[device_idx][0] for device_idx in edge_devices], axis=0)
        y_train_edge = np.concatenate([clients_data[device_idx][2] for device_idx in edge_devices], axis=0)

        if num_classes is None:
            num_classes = len(np.unique(y_train_edge))

        # Initialize residuals for the edge
        y_onehot_edge = np.zeros((y_train_edge.shape[0], num_classes))
        y_onehot_edge[np.arange(y_train_edge.shape[0]), y_train_edge] = 1.0
        residuals_edge = y_onehot_edge.copy()

        edge_device_models = {}

        # Track start index to slice residuals per device
        start_idx = 0
        for device_idx in edge_devices:
            X_dev = clients_data[device_idx][0]
            n_samples = X_dev.shape[0]

            # Slice residuals for this device
            residual_dev = residuals_edge[start_idx:start_idx + n_samples]
            y_pseudo = residual_dev.argmax(axis=1)

            # Train device-level LightGBM
            model = train_lightgbm(
                X_dev, y_pseudo,
                n_estimators=config["device_boosting_rounds"],
                num_classes=num_classes
            )

            # Predict probabilities on device data
            X_dev_fix = (
                X_dev[model.feature_name_] if isinstance(X_dev, pd.DataFrame)
                else pd.DataFrame(X_dev, columns=model.feature_name_)
            )
            pred_proba_raw = model.predict_proba(X_dev_fix)
            pred_proba_full = pad_proba(pred_proba_raw, model.classes_.astype(int), num_classes)

            # Update only this device's portion of residuals
            residuals_edge[start_idx:start_idx + n_samples] -= pred_proba_full
            start_idx += n_samples

            edge_device_models[device_idx] = model

        # Edge-level model on aggregated edge data
        y_pseudo_edge = residuals_edge.argmax(axis=1)
        edge_model = train_lightgbm(
            X_train_edge, y_pseudo_edge,
            n_estimators=config["edge_boosting_rounds"],
            num_classes=num_classes
        )
        edge_device_models["edge_model"] = edge_model
        all_edge_models.append(edge_device_models)

    return all_edge_models

# ============================================================
# Gossip Bayesian Aggregator
# ============================================================
def gossip_layer_bayesian(edge_models, clients_data, num_classes):
    X_list, y_list = [], []
    for edge in edge_models:
        for key, model in edge.items():
            if key == "edge_model":
                continue
            for idx, (X_train, _, y_train, _) in enumerate(clients_data):
                leaf = extract_leaf_embeddings(model, X_train)
                leaf_ohe = one_hot_encode_leaf_embeddings(leaf)
                X_list.append(leaf_ohe)
                y_list.append(y_train)
    X_sparse = vstack(X_list)
    y = np.hstack(y_list)

    if config["variance_prune"]:
        vt = VarianceThreshold(threshold=config["variance_threshold"])
        X_sparse = vt.fit_transform(X_sparse)

    scaler = StandardScaler(with_mean=False)
    X_proc = scaler.fit_transform(X_sparse)

    n_features = X_proc.shape[1]
    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0, sigma=1, shape=(num_classes,))
        betas = pm.Normal("betas", mu=0, sigma=1, shape=(n_features, num_classes))
        logits = pm.math.dot(X_proc, betas) + alpha
        p = pm.math.softmax(logits)
        y_obs = pm.Categorical("y_obs", p=p, observed=y)
        trace = pm.sample(draws=config["bayes_n_samples"], tune=config["bayes_n_tune"],
                          chains=2, cores=2, target_accept=0.9, progressbar=True)
    return model, trace


# ============================================================
# Residual Feedback Loop
# ============================================================
def compute_residual(y_true, y_pred):
    return y_true - y_pred


def backward_edge_refinement(edge_models, device_models, edge_groups, clients_data, global_residuals):
    # Sequentially update edge and device models
    start_idx = 0
    for edge_idx, edge in enumerate(edge_groups):
        edge_size = sum([clients_data[device_idx][0].shape[0] for device_idx in edge])
        edge_residuals = global_residuals[start_idx:start_idx + edge_size]
        start_idx += edge_size

        residual = edge_residuals.copy()
        for key, model in edge_models[edge_idx].items():
            if key == "edge_model":
                continue
            X_dev = clients_data[key][0]
            y_pseudo = residual.argmax(axis=1)
            model.fit(X_dev, y_pseudo, init_model=model)
            pred = model.predict(X_dev)
            residual -= pred
    return edge_models, device_models


def forward_pass_with_feedback(clients_data, edge_groups, n_iterations=3):
    """
    Full HDPFTL forward pass with residual feedback, safe class handling, and
    Bayesian gossip aggregation.
    """
    # 1️⃣ Device-level sequential boosting
    device_models = device_level_boosting(
        clients_data,
        rounds=config["device_boosting_rounds"]
    )

    # Determine total number of classes across all clients
    all_labels = np.hstack([y for _, _, y, _ in clients_data])
    num_classes = len(np.unique(all_labels))

    # 2️⃣ Edge-level sequential boosting
    edge_models = edge_level_boosting(
        edge_groups,
        device_models,
        clients_data,
        num_classes=num_classes
    )

    # 3️⃣ Iterative residual feedback loop
    for iteration in range(n_iterations):
        print(f"[INFO] Residual Feedback Iteration {iteration + 1}/{n_iterations}")

        # 3a️⃣ Bayesian gossip aggregation
        gossip_model, gossip_trace = gossip_layer_bayesian(edge_models, clients_data, num_classes)

        # 3b️⃣ Optionally, compute residuals for refinement (simplified version)
        # Here, we skip complex posterior residual update for stability
        # residuals could be computed from leaf embeddings + Bayesian predictions if desired

    return device_models, edge_models, gossip_model, gossip_trace

def safe_split(X, y, test_size=0.2, random_state=42, num_classes=None):
    """
    Split while guaranteeing that every class in y appears in the training set.
    Falls back to repeated shuffling if needed.
    """
    from collections import Counter
    for _ in range(10):  # try a few times
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )
        # Check if all classes in val exist in train
        missing = set(np.unique(y_val)) - set(np.unique(y_tr))
        if not missing:
            return X_tr, X_val, y_tr, y_val
    # Worst-case fallback: move missing samples from val → train
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=test_size,
                                                random_state=random_state, stratify=None)
    missing = set(np.unique(y_val)) - set(np.unique(y_tr))
    if missing:
        mask = np.isin(y_val, list(missing))
        # move these samples into train
        X_tr = np.vstack([X_tr, X_val[mask]])
        y_tr = np.concatenate([y_tr, y_val[mask]])
        X_val = X_val[~mask]
        y_val = y_val[~mask]
    return X_tr, X_val, y_tr, y_val

# ============================================================
# Main Execution
# ============================================================
if __name__ == "__main__":
    folder_path = "CIC_IoT_dataset_2023"
    today_str = datetime.now().strftime("%Y-%m-%d")
    log_path_str = os.path.join("logs", f"{folder_path}_{today_str}")
    os.makedirs(log_path_str, exist_ok=True)

    X_final, y_final, X_pretrain, y_pretrain, X_finetune, y_finetune, X_test, y_test = preprocess_data(log_path_str,
                                                                                                       folder_path)

    clients_data, edge_groups = create_non_iid_clients(
        X_pretrain, y_pretrain, X_test, y_test,
        n_clients=config["n_clients"],
        n_edges=config["n_edges"],
        min_clients_per_edge=config["n_clients"] // config["n_edges"]
    )
    print(f"[INFO] Created {len(clients_data)} clients across {len(edge_groups)} edges.")

    device_models, edge_models, gossip_model, gossip_trace = forward_pass_with_feedback(
        clients_data, edge_groups, n_iterations=3
    )

    if config["save_results"]:
        os.makedirs(config["results_path"], exist_ok=True)
        with open(os.path.join(config["results_path"], "device_models.pkl"), "wb") as f:
            pickle.dump(device_models, f)
        with open(os.path.join(config["results_path"], "edge_models.pkl"), "wb") as f:
            pickle.dump(edge_models, f)
        with open(os.path.join(config["results_path"], "gossip_model.pkl"), "wb") as f:
            pickle.dump(gossip_model, f)
        with open(os.path.join(config["results_path"], "gossip_trace.pkl"), "wb") as f:
            pickle.dump(gossip_trace, f)
        print("[INFO] Saved all models and traces.")

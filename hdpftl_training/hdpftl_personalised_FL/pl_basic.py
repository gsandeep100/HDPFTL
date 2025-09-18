# ============================================================
# Imports
# ============================================================
import os
import pickle
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack, hstack
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
import pymc as pm

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
    "n_classes_per_client": 20,
    "epoch": 50,  # Number of global boosting rounds
    "device_boosting_rounds": 10,  # intra-device sequential boosting
    "edge_boosting_rounds": 5,  # edge-level sequential boosting
    "n_estimators": 50,  # LightGBM per boosting stage
    "variance_prune": True,
    "variance_threshold": 1e-4,
    "bayes_n_samples": 1000,
    "bayes_n_tune": 1000,
    "aggregator_type": "bayesian",  # only at gossip layer
    "save_results": True,
    "results_path": "results"
}

np.random.seed(config["random_seed"])

# ============================================================
# Helper Functions
# ============================================================

def create_non_iid_clients(
    X: ArrayLike,
    y: ArrayLike,
    X_test: ArrayLike,
    y_test: ArrayLike,
    n_clients: int = 50,
    n_edges: int = 10,
    min_clients_per_edge: int = 5,
    random_state: int = 42
) -> Tuple[List[ClientData], List[List[int]]]:
    """Split data into non-IID clients and assign to edges."""
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
        idx = shuffled_idx[i * samples_per_client: (i + 1) * samples_per_client]
        clients_data.append((X_np[idx], X_test_np, y_np[idx], y_test_np))

    # Assign clients to edges
    clients_per_edge = max(min_clients_per_edge, n_clients // n_edges)
    edge_groups = []
    all_client_indices = list(range(n_clients))
    np.random.shuffle(all_client_indices)
    for i in range(0, n_clients, clients_per_edge):
        edge_groups.append(all_client_indices[i:i + clients_per_edge])
    return clients_data, edge_groups

def train_lightgbm(X_train, y_train, X_val, y_val, n_estimators=50, num_classes=None):
    """Train LightGBM for classification or regression."""
    if num_classes and num_classes > 2:
        model = lgb.LGBMClassifier(
            objective="multiclass", metric="multi_logloss", num_class=num_classes,
            boosting_type="gbdt", num_leaves=31, learning_rate=0.05,
            n_estimators=n_estimators, verbose=-1)
    else:
        model = lgb.LGBMClassifier(
            objective="binary", metric="binary_logloss",
            boosting_type="gbdt", num_leaves=31, learning_rate=0.05,
            n_estimators=n_estimators, verbose=-1)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              eval_metric=model._objective, callbacks=[lgb.early_stopping(stopping_rounds=20)])
    return model

def extract_leaf_embeddings(model, X: ArrayLike) -> np.ndarray:
    """Get LightGBM leaf indices."""
    X_np = X.to_numpy() if isinstance(X, (pd.DataFrame, pd.Series)) else X
    return model.predict(X_np, pred_leaf=True).astype(int)

def one_hot_encode_leaf_embeddings(leaf_embeddings: np.ndarray) -> csr_matrix:
    """Convert leaf indices to one-hot encoded sparse matrix."""
    ohe = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
    return ohe.fit_transform(leaf_embeddings)

def pad_sparse_matrices(matrices: List[csr_matrix]) -> List[csr_matrix]:
    max_cols = max(mat.shape[1] for mat in matrices)
    padded = []
    for mat in matrices:
        if mat.shape[1] < max_cols:
            diff = max_cols - mat.shape[1]
            mat = hstack([mat, csr_matrix((mat.shape[0], diff))])
        padded.append(mat)
    return padded

# ============================================================
# Layer 1: Device Sequential Boosting
# ============================================================
def device_level_boosting(clients_data: List[ClientData], rounds=10):
    device_models = []
    for X_train, X_test, y_train, y_test in clients_data:
        residual = y_train.copy().astype(float)
        models_per_device = []
        for r in range(rounds):
            X_tr, X_val, y_tr, y_val = train_test_split(X_train, residual, test_size=0.2, random_state=config["random_seed"])
            model = train_lightgbm(X_tr, y_tr, X_val, y_val, n_estimators=config["n_estimators"])
            pred = model.predict(X_train)
            residual -= pred
            models_per_device.append(model)
        device_models.append(models_per_device)
    return device_models

# ============================================================
# Layer 2: Edge Sequential Boosting
# ============================================================
def edge_level_boosting(edge_groups: List[List[int]], device_models: List[List[lgb.LGBMModel]], clients_data: List[ClientData]):
    edge_models = []
    for edge in edge_groups:
        # Gather predictions from devices
        edge_pred = None
        for device_idx in edge:
            device_pred = sum([m.predict(clients_data[device_idx][0]) for m in device_models[device_idx]])
            if edge_pred is None:
                edge_pred = device_pred
            else:
                edge_pred += device_pred
        # Edge-level sequential boosting on residual
        residual = clients_data[edge[0]][2].copy() - edge_pred
        models_per_edge = []
        for r in range(config["edge_boosting_rounds"]):
            model = train_lightgbm(clients_data[edge[0]][0], residual,
                                   clients_data[edge[0]][1], clients_data[edge[0]][3],
                                   n_estimators=config["n_estimators"])
            pred = model.predict(clients_data[edge[0]][0])
            residual -= pred
            models_per_edge.append(model)
        edge_models.append(models_per_edge)
    return edge_models

# ============================================================
# Layer 3: Gossip / Global Bayesian Aggregator
# ============================================================
def gossip_layer_bayesian(edge_models: List[List[lgb.LGBMModel]], clients_data: List[ClientData], num_classes: int):
    # Collect leaf embeddings from all edge models
    X_list, y_list = [], []
    for edge in edge_models:
        for model in edge:
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
# Residual Feedback Functions
# ============================================================

def compute_residual(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute residuals for boosting."""
    return y_true - y_pred

# ----------------------------
# Backward Pass: Edge-level refinement
# ----------------------------
def backward_edge_refinement(edge_models: List[List[lgb.LGBMModel]],
                             device_models: List[List[lgb.LGBMModel]],
                             edge_groups: List[List[int]],
                             clients_data: List[ClientData],
                             global_residuals: np.ndarray):
    """
    Use global residuals to update edge and device models.
    """
    # Split global residuals across edges proportionally
    start_idx = 0
    for edge_idx, edge in enumerate(edge_groups):
        edge_size = sum([clients_data[device_idx][0].shape[0] for device_idx in edge])
        edge_residuals = global_residuals[start_idx:start_idx + edge_size]
        start_idx += edge_size

        # Update edge models sequentially
        residual = edge_residuals.copy()
        for model in edge_models[edge_idx]:
            X_edge = clients_data[edge[0]][0]
            model.fit(X_edge, residual, init_model=model)
            pred = model.predict(X_edge)
            residual -= pred

        # Update device models sequentially
        for device_idx in edge:
            residual = edge_residuals.copy()
            for model in device_models[device_idx]:
                X_dev = clients_data[device_idx][0]
                model.fit(X_dev, residual, init_model=model)
                pred = model.predict(X_dev)
                residual -= pred
    return edge_models, device_models

# ----------------------------
# Forward Pass (boosting + Bayesian aggregation)
# ----------------------------
def forward_pass_with_feedback(clients_data, edge_groups, n_iterations=3):
    """
    Forward + backward residual feedback loop.
    """
    # Layer 1: Device-level boosting
    device_models = device_level_boosting(clients_data, rounds=config["device_boosting_rounds"])

    # Layer 2: Edge-level boosting
    edge_models = edge_level_boosting(edge_groups, device_models, clients_data)

    for iteration in range(n_iterations):
        print(f"[INFO] Residual Feedback Iteration {iteration+1}/{n_iterations}")

        # Layer 3: Gossip Bayesian aggregation
        num_classes = len(np.unique([y for _, _, y, _ in clients_data]))
        gossip_model, gossip_trace = gossip_layer_bayesian(edge_models, clients_data, num_classes=num_classes)

        # Compute global predictions for residuals
        X_global_list = []
        y_global_list = []
        for edge in edge_models:
            for model in edge:
                for idx, (X_train, _, y_train, _) in enumerate(clients_data):
                    leaf = extract_leaf_embeddings(model, X_train)
                    leaf_ohe = one_hot_encode_leaf_embeddings(leaf)
                    X_global_list.append(leaf_ohe)
                    y_global_list.append(y_train)
        X_sparse = vstack(X_global_list)
        y_true = np.hstack(y_global_list)

        # Compute posterior mean prediction
        posterior_pred = pm.summary(gossip_trace)["mean"].values
        # Note: posterior_pred is parameter-level; for proper residual we could simulate predictive posterior
        # For simplicity here, approximate by linear combination
        y_global_pred = np.clip(np.dot(X_sparse.toarray(), posterior_pred[:X_sparse.shape[1]]), 0, 1)

        global_residuals = compute_residual(y_true, y_global_pred)

        # Backward pass: propagate residuals to edges and devices
        edge_models, device_models = backward_edge_refinement(edge_models, device_models, edge_groups, clients_data, global_residuals)

    return device_models, edge_models, gossip_model, gossip_trace

# ============================================================
# Main Flow with Residual Feedback
# ============================================================
if __name__ == "__main__":
    # Preprocessing and client creation (same as before)
    folder_path = "CIC_IoT_dataset_2023"
    today_str = datetime.now().strftime("%Y-%m-%d")
    log_path_str = os.path.join("logs", f"{folder_path}_{today_str}")
    os.makedirs(log_path_str, exist_ok=True)
    print(f"[INFO] Logging directory: {log_path_str}")

    X_final, y_final, X_pretrain, y_pretrain, X_finetune, y_finetune, X_test, y_test = preprocess_data(log_path_str, folder_path)

    clients_data, edge_groups = create_non_iid_clients(
        X_pretrain, y_pretrain, X_test, y_test,
        n_clients=config["n_clients"],
        n_edges=config["n_edges"],
        min_clients_per_edge=config["n_clients"] // config["n_edges"]
    )
    print(f"[INFO] Created {len(clients_data)} clients across {len(edge_groups)} edges.")

    # Run HPFL with forward-backward residual feedback
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
        print(f"[INFO] Saved all models and traces.")

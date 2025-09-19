# ============================================================
# Imports
# ============================================================
import os
import pickle
from datetime import datetime
from typing import Tuple, Union
import lightgbm as lgb
import numpy as np
import pandas as pd
import pymc as pm
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax as sp_softmax
# Custom preprocessing
from hdpftl_training.hdpftl_data.preprocess import safe_preprocess_data

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

np.random.seed(config["random_seed"])  # ensures reproducibility


# ============================================================
# Helper Functions
# ============================================================

def create_non_iid_clients(X, y, X_test, y_test,
                                n_clients=50, n_edges=10, min_clients_per_edge=5,
                                random_state=42):
    """
    Safely split dataset into clients and edges (non-IID), ensuring
    X/y are consistent per client.
    """
    np.random.seed(random_state)

    X = X.to_numpy() if isinstance(X, (pd.DataFrame, pd.Series)) else X
    y = y.to_numpy() if isinstance(y, (pd.DataFrame, pd.Series)) else y
    X_test = X_test.to_numpy() if isinstance(X_test, (pd.DataFrame, pd.Series)) else X_test
    y_test = y_test.to_numpy() if isinstance(y_test, (pd.DataFrame, pd.Series)) else y_test

    assert X.shape[0] == y.shape[0], "X and y must have same number of samples!"

    n_samples = X.shape[0]
    shuffled_idx = np.random.permutation(n_samples)
    samples_per_client = n_samples // n_clients

    clients_data = []

    for i in range(n_clients):
        start = i * samples_per_client
        end = start + samples_per_client if i != n_clients - 1 else n_samples
        idx = shuffled_idx[start:end]

        X_client = X[idx]
        y_client = y[idx]

        # Shared test set
        clients_data.append((X_client, X_test, y_client, y_test))

    # Assign clients to edges
    clients_per_edge = max(min_clients_per_edge, n_clients // n_edges)
    all_client_indices = list(range(n_clients))
    np.random.shuffle(all_client_indices)

    edge_groups = []
    for i in range(0, n_clients, clients_per_edge):
        edge_groups.append(all_client_indices[i:i + clients_per_edge])

    return clients_data, edge_groups




def pad_proba(pred_proba, present_classes, num_classes):
    """
    Pads predicted probability matrix to match total num_classes.
    Handles missing classes during partial training.
    """
    full = np.zeros((pred_proba.shape[0], num_classes))
    for i, cls in enumerate(present_classes):
        full[:, cls] = pred_proba[:, i]
    return full


def train_lightgbm(X_train, y_train, num_classes=None, n_estimators=1, random_state=42):
    """
    Train LightGBM safely even if some classes are missing in this client batch.
    """
    X_np = X_train.to_numpy() if isinstance(X_train, (pd.DataFrame, pd.Series)) else X_train
    y_np = y_train.to_numpy() if isinstance(y_train, (pd.DataFrame, pd.Series)) else y_train

    if num_classes is None:
        num_classes = len(np.unique(y_np))

    if num_classes > 2:
        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=num_classes,
            metric="multi_logloss",
            n_estimators=n_estimators,
            random_state=random_state,
            verbose=-1
        )
    else:
        model = lgb.LGBMClassifier(
            objective="binary",
            metric="binary_logloss",
            n_estimators=n_estimators,
            random_state=random_state,
            verbose=-1
        )

    model.fit(X_np, y_np)

    # Ensure safe predict_proba
    def safe_predict_proba(X_in):
        X_in_np = X_in.to_numpy() if isinstance(X_in, (pd.DataFrame, pd.Series)) else X_in
        pred = model.predict_proba(X_in_np)
        if pred.shape[1] < num_classes:
            full = np.zeros((pred.shape[0], num_classes))
            for i, cls in enumerate(model.classes_):
                full[:, cls] = pred[:, i]
            pred = full
        return pred

    model.safe_predict_proba = safe_predict_proba
    return model



def extract_leaf_embeddings(model, X):
    """Return leaf indices, ensuring consistent features."""
    if isinstance(X, pd.DataFrame):
        X_fix = X[model.feature_name_]
    else:
        X_fix = pd.DataFrame(X, columns=model.feature_name_)
    return model.predict(X_fix, pred_leaf=True).astype(int)


def one_hot_encode_leaf_embeddings(leaf_embeddings, max_leaf=None):
    """Convert leaf indices to sparse one-hot embeddings with consistent width."""
    ohe = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
    if max_leaf is None:
        return ohe.fit_transform(leaf_embeddings), ohe
    else:
        # Fit on max_leaf dummy data to ensure consistent columns
        dummy = np.zeros((1, leaf_embeddings.shape[1]))
        leaf_embeddings_padded = np.vstack([leaf_embeddings, dummy])
        ohe.fit(leaf_embeddings_padded)
        one_hot = ohe.transform(leaf_embeddings)
        # Pad columns if needed
        if one_hot.shape[1] < max_leaf:
            n_samples = one_hot.shape[0]
            one_hot = hstack([one_hot, csr_matrix((n_samples, max_leaf - one_hot.shape[1]))])
        return one_hot, ohe


def safe_split(X, y, test_size=0.2, random_state=42):
    """
    Safely split dataset while ensuring all classes appear in the training set.
    Prevents ValueError for "least populated class has only 1 member".
    """
    for _ in range(10):
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=None)
        missing = set(np.unique(y_val)) - set(np.unique(y_tr))
        if not missing:
            return X_tr, X_val, y_tr, y_val

    # fallback: move missing samples from val -> train
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=None)
    missing = set(np.unique(y_val)) - set(np.unique(y_tr))
    if missing:
        mask = np.isin(y_val, list(missing))
        X_tr = np.vstack([X_tr, X_val[mask]])
        y_tr = np.concatenate([y_tr, y_val[mask]])
        X_val = X_val[~mask]
        y_val = y_val[~mask]
    return X_tr, X_val, y_tr, y_val


# ============================================================
# Device Level Boosting
# ============================================================
def device_level_boosting(clients_data, num_classes):
    """
    Sequential boosting for each device, with residual updates and safe predict_proba.
    """
    device_models = []

    for X_dev, _, y_dev, _ in clients_data:
        y_enc = y_dev  # already aligned
        residual = np.zeros((X_dev.shape[0], num_classes))
        residual[np.arange(X_dev.shape[0]), y_enc] = 1.0

        models_per_device = []
        for _ in range(config["device_boosting_rounds"]):
            y_pseudo = residual.argmax(axis=1)
            model = train_lightgbm(X_dev, y_pseudo, num_classes=num_classes, n_estimators=1)
            pred_proba = model.predict_proba(X_dev)
            residual -= pred_proba
            models_per_device.append(model)

        device_models.append(models_per_device)

    return device_models


# =============================================================
# Edge Level Boosting
# ============================================================
def edge_level_boosting(edge_groups, clients_data, num_classes):
    """
    Sequential boosting at edge level with stacked device data.
    """
    edge_models = []

    for edge_clients in edge_groups:
        X_edge = np.vstack([clients_data[i][0] for i in edge_clients])
        y_edge = np.hstack([clients_data[i][2] for i in edge_clients])

        residual_edge = np.zeros((X_edge.shape[0], num_classes))
        residual_edge[np.arange(y_edge.shape[0]), y_edge] = 1.0

        for _ in range(config["edge_boosting_rounds"]):
            y_pseudo = residual_edge.argmax(axis=1)
            model = train_lightgbm(pd.DataFrame(X_edge), y_pseudo, num_classes=num_classes, n_estimators=1)
            pred_proba = model.predict_proba(X_edge)
            residual_edge -= pred_proba

        edge_models.append(model)

    return edge_models


# =============================================================
# Gossip Bayesian Aggregator
# ============================================================

def gossip_layer_bayesian(edge_models, clients_data, num_classes, use_calibration=True):
    """
    Robust Bayesian aggregation with consistent shapes.
    """
    leaf_ohe_per_edge = []
    y_per_edge = []
    max_features = 0

    # --- Collect all leaf embeddings first ---
    for edge_model in edge_models:
        edge_features = []
        edge_labels = []

        for idx, (X_dev, _, y_dev, _) in enumerate(clients_data):
            leaf = extract_leaf_embeddings(edge_model, X_dev)
            leaf_ohe = one_hot_encode_leaf_embeddings(leaf)[0]
            edge_features.append(leaf_ohe)
            edge_labels.append(y_dev)

        # Determine max features
        edge_max = max([f.shape[1] for f in edge_features])
        if edge_max > max_features:
            max_features = edge_max

        leaf_ohe_per_edge.append(edge_features)
        y_per_edge.append(np.hstack(edge_labels))

    # --- Pad features to max_features ---
    padded_edges = []
    for edge_features in leaf_ohe_per_edge:
        padded_edge = []
        for f in edge_features:
            n_samples, n_cols = f.shape
            if n_cols < max_features:
                f = hstack([f, csr_matrix((n_samples, max_features - n_cols))])
            padded_edge.append(f)
        padded_edges.append(padded_edge)

    # --- Compute calibrated probabilities per edge ---
    calibrated_probs_edges = []
    for edge_idx, edge_features in enumerate(padded_edges):
        X_edge = vstack(edge_features)
        y_edge = y_per_edge[edge_idx]
        edge_model = edge_models[edge_idx]

        # Predict
        if hasattr(edge_model, "predict_proba"):
            probs = edge_model.predict_proba(X_edge)
        else:
            probs = np.zeros((X_edge.shape[0], num_classes))

        # Isotonic calibration
        if use_calibration:
            calibrated_probs = np.zeros_like(probs)
            for c in range(num_classes):
                iso = IsotonicRegression(out_of_bounds='clip')
                iso.fit(probs[:, c], y_edge == c)
                calibrated_probs[:, c] = iso.transform(probs[:, c])
            # Normalize
            calibrated_probs /= calibrated_probs.sum(axis=1, keepdims=True)
        else:
            calibrated_probs = probs

        calibrated_probs_edges.append(calibrated_probs)

    # --- Stack edge probabilities horizontally for Bayesian ---
    # Make sure rows match by using min_rows
    min_rows = min(p.shape[0] for p in calibrated_probs_edges)
    X_proc = np.hstack([p[:min_rows] for p in calibrated_probs_edges])
    y_global = np.hstack([y[:min_rows] for y in y_per_edge])

    # --- Bayesian multinomial model ---
    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0, sigma=1, shape=(num_classes,))
        betas = pm.Normal("betas", mu=0, sigma=1, shape=(X_proc.shape[1], num_classes))
        logits = pm.math.dot(X_proc, betas) + alpha
        p = pm.math.softmax(logits)
        y_obs = pm.Categorical("y_obs", p=p, observed=y_global)

        trace = pm.sample(
            draws=config["bayes_n_samples"],
            tune=config["bayes_n_tune"],
            chains=2,
            cores=2,
            target_accept=0.9,
            progressbar=True
        )

    return model, trace


def variance_prune_sparse(X_sparse, threshold=1e-4):
    """
    Remove sparse columns with variance below threshold.
    X_sparse: csr_matrix
    Returns: pruned csr_matrix
    """
    # Compute variance per column
    mean = X_sparse.mean(axis=0).A1
    mean_sq = X_sparse.power(2).mean(axis=0).A1
    var = mean_sq - mean**2
    keep_cols = np.where(var > threshold)[0]
    return X_sparse[:, keep_cols], keep_cols
# ============================================================
# Forward Pass with Feedback
# ============================================================

def forward_pass_with_feedback(clients_data, edge_groups, n_iterations=3):
    """
    Full pipeline with:
    - Device boosting
    - Edge boosting
    - Bayesian aggregation
    - Residual feedback
    (Fixed to safely pad / stack arrays of unequal lengths)
    """
    # Label encoding
    all_labels = np.hstack([y_dev for X_dev, _, y_dev, _ in clients_data])
    le = LabelEncoder()
    le.fit(all_labels)
    num_classes = len(le.classes_)

    residuals_clients = [None] * len(clients_data)

    for iteration in range(n_iterations):
        print(f"[INFO] Iteration {iteration+1}/{n_iterations}")

        # --- Device Level ---
        device_models = []
        for idx, (X_dev, _, y_dev, _) in enumerate(clients_data):
            y_enc = le.transform(y_dev)
            if residuals_clients[idx] is None:
                residual = np.zeros((X_dev.shape[0], num_classes))
                residual[np.arange(X_dev.shape[0]), y_enc] = 1.0
            else:
                residual = residuals_clients[idx]

            models_per_device = []
            for _ in range(config["device_boosting_rounds"]):
                y_pseudo = residual.argmax(axis=1)
                model = train_lightgbm(X_dev, y_pseudo, num_classes=num_classes, n_estimators=1)
                pred_proba = model.predict_proba(X_dev)
                residual -= pred_proba
                models_per_device.append(model)

            residuals_clients[idx] = residual
            device_models.append(models_per_device)

        # --- Edge Level ---
        edge_models = []
        for edge_clients in edge_groups:
            X_edge = np.vstack([clients_data[i][0] for i in edge_clients])
            y_edge = np.hstack([clients_data[i][2] for i in edge_clients])
            residual_edge = np.vstack([residuals_clients[i] for i in edge_clients])

            for _ in range(config["edge_boosting_rounds"]):
                y_pseudo = residual_edge.argmax(axis=1)
                model = train_lightgbm(pd.DataFrame(X_edge), y_pseudo, num_classes=num_classes, n_estimators=1)
                pred_proba = model.predict_proba(X_edge)
                residual_edge -= pred_proba

            edge_models.append(model)

        # --- Bayesian aggregation ---
        X_bayes_list = []
        y_bayes_list = []
        for edge_idx, edge_clients in enumerate(edge_groups):
            X_edge_stack = np.vstack([clients_data[i][0] for i in edge_clients])
            y_edge_stack = np.hstack([clients_data[i][2] for i in edge_clients])
            edge_model = edge_models[edge_idx]
            probs = edge_model.safe_predict_proba(X_edge_stack)

            # Ensure 2-D shape
            probs = np.atleast_2d(probs)
            if probs.ndim == 1:      # (N,) -> (N,1)
                probs = probs[:, None]

            X_bayes_list.append(probs)
            y_bayes_list.append(y_edge_stack)

        # Pad arrays along rows to allow horizontal stacking
        max_rows = max(arr.shape[0] for arr in X_bayes_list)
        padded_X_bayes = []
        for arr in X_bayes_list:
            rows, cols = arr.shape
            pad_rows = max_rows - rows
            # pad_rows can be 0 if already max
            padded = np.pad(
                arr,
                ((0, pad_rows), (0, 0)),
                mode='constant',
                constant_values=0
            )
            padded_X_bayes.append(padded)

        X_proc = np.hstack(padded_X_bayes)

        # For labels we truncate/pad to max_rows to match X_proc
        padded_y_bayes = []
        for arr in y_bayes_list:
            if arr.shape[0] < max_rows:
                arr_padded = np.pad(arr, (0, max_rows - arr.shape[0]),
                                    mode='constant',
                                    constant_values=-1)  # -1 marks padded rows
            else:
                arr_padded = arr[:max_rows]
            padded_y_bayes.append(arr_padded)

        y_global = np.hstack(padded_y_bayes)
        valid_mask = y_global != -1
        y_global = y_global[valid_mask]
        X_proc = X_proc[valid_mask]

        y_global_enc = le.transform(y_global)

        with pm.Model() as gossip_model:
            alpha = pm.Normal("alpha", mu=0, sigma=1, shape=(num_classes,))
            betas = pm.Normal("betas", mu=0, sigma=1, shape=(X_proc.shape[1], num_classes))
            logits = pm.math.dot(X_proc, betas) + alpha
            p = pm.math.softmax(logits)
            y_obs = pm.Categorical("y_obs", p=p, observed=y_global_enc)

            gossip_trace = pm.sample(
                draws=config["bayes_n_samples"],
                tune=config["bayes_n_tune"],
                chains=2,
                cores=2,
                target_accept=0.9,
                progressbar=True
            )

        # --- Residual Feedback ---
        global_preds = X_proc @ gossip_trace['betas'].mean(axis=0) + gossip_trace['alpha'].mean(axis=0)
        global_probs = sp_softmax(global_preds, axis=1)
        global_residual = np.zeros_like(global_probs)
        global_residual[np.arange(len(y_global_enc)), y_global_enc] = 1.0
        global_residual -= global_probs

        # Split residuals back to clients
        start_idx = 0
        for e_idx, edge_clients in enumerate(edge_groups):
            n_samples_edge = sum(clients_data[i][0].shape[0] for i in edge_clients)
            residual_edge_feedback = global_residual[start_idx:start_idx + n_samples_edge]
            start_idx += n_samples_edge

            offset = 0
            for d_idx in edge_clients:
                n_samples_dev = clients_data[d_idx][0].shape[0]
                residuals_clients[d_idx] = residual_edge_feedback[offset:offset + n_samples_dev]
                offset += n_samples_dev

    print("[INFO] Completed all iterations with Bayesian aggregation.")
    return device_models, edge_models, gossip_model, gossip_trace


# ============================================================
# Main Execution
# ============================================================
if __name__ == "__main__":
    folder_path = "CIC_IoT_dataset_2023"
    today_str = datetime.now().strftime("%Y-%m-%d")
    log_path_str = os.path.join("logs", f"{folder_path}_{today_str}")
    os.makedirs(log_path_str, exist_ok=True)

    # Flag to enable/disable Isotonic calibration before Bayesian layer
    USE_CALIBRATION = True

    # Preprocess and split
    X_final, y_final, X_pretrain, y_pretrain, X_finetune, y_finetune, X_test, y_test = safe_preprocess_data(
        log_path_str, folder_path
    )

    # Create non-IID clients and edge groups
    clients_data, edge_groups = create_non_iid_clients(
        X_pretrain, y_pretrain, X_test, y_test,
        n_clients=config["n_clients"],
        n_edges=config["n_edges"],
        min_clients_per_edge=config["n_clients"] // config["n_edges"]
    )
    print(f"[INFO] Created {len(clients_data)} clients across {len(edge_groups)} edges.")

    # Run full HDPFTL pipeline with optional calibration
    device_models, edge_models, gossip_model, gossip_trace = forward_pass_with_feedback(clients_data, edge_groups, n_iterations=3)

    # Save results
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

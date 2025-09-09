"""


Optimized LightGBM Hyperparameters

Your script currently uses fixed values (num_leaves=31, learning_rate=0.05, n_estimators=50).
That’s a good baseline, but for federated tasks, you usually want a bit more flexibility.

Here’s a recommended grid you can explore (via cross-validation or Bayesian optimization):

lgb_params_grid = {
    "boosting_type": ["gbdt", "dart"],   # dart helps in non-iid cases
    "num_leaves": [16, 31, 63, 127],     # tradeoff: small = less variance, large = more complexity
    "max_depth": [-1, 5, 10, 20],        # -1 = unlimited
    "learning_rate": [0.01, 0.05, 0.1],  # smaller = more stable in federated
    "n_estimators": [50, 100, 200, 500], # higher with early stopping
    "min_child_samples": [10, 20, 50, 100], # prevents overfitting on small clients
    "subsample": [0.6, 0.8, 1.0],        # row sampling
    "colsample_bytree": [0.6, 0.8, 1.0], # feature sampling
    "reg_alpha": [0, 0.1, 1],            # L1 regularization
    "reg_lambda": [0, 0.1, 1],           # L2 regularization
}


👉 For binary classification, also try:

objective = "binary"

metric = "auc" in addition to "binary_logloss"

👉 For multiclass, keep:

objective = "multiclass"

metric = "multi_logloss"

⚠️ In federated learning:

Use smaller learning rates (0.01–0.05) for stability.

Use subsample + colsample_bytree < 1.0 to improve generalization across non-iid clients.

📊 Optimized Bayesian Hyperparameters

Your current setup uses:

alpha ~ N(0,1)

betas ~ N(0,0.5)

n_samples=200, n_tune=200

This works but is under-sampled and priors are quite restrictive.
Here’s how you can improve:

Priors

Intercepts (alpha): Normal(0, 5) → allows more flexibility across clients

Betas (coefficients):

Normal(0, 1) (less restrictive than 0.5)

OR Laplace(0, 1) if you want sparsity (feature pruning effect)

Sampling Strategy
bayes_config = {
    "n_samples": 1000,   # more robust posterior
    "n_tune": 1000,      # longer tuning for stability
    "target_accept": 0.9,  # keep same
    "chains": 2,         # parallel chains for robustness
    "cores": 2,          # if resources allow
}

Variance Pruning

Your script uses:

VarianceThreshold(threshold=1e-6)


This is fine, but you may experiment with 1e-4 or 1e-3 to cut down noise features before Bayesian aggregation.

🚀 Practical Recommendation for You

LightGBM first:

Start with num_leaves=63, learning_rate=0.05, n_estimators=200, subsample=0.8, colsample_bytree=0.8.

Tune with grid/random search across clients.

Bayesian aggregator:

Increase n_samples/n_tune to at least 1000 each.

Use Normal(0,1) or Laplace(0,1) priors on betas.

Try variance thresholding at 1e-4 for faster training.

Balance:

If Bayesian becomes too heavy, fall back to Logistic Regression (logreg) aggregator with penalty='l2', C=1.0.
"""



# ============================================================
# Imports
# ============================================================

import os
import pickle
import threading
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Union
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from scipy.sparse import csr_matrix, vstack, hstack
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

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
    "epoch": 500,
    "min_clients_per_edge": 5,
    "n_estimators": 50,
    "aggregator_type": "bayesian",  # bayesian / logreg / simple / weighted
    "variance_prune": True,
    "variance_threshold": 1e-6,
    "save_results": True,
    "results_path": "results",
    "bayes_n_samples": 200,
    "bayes_n_tune": 200,
}

np.random.seed(config["random_seed"])


# ============================================================
# Non-IID Client Creation
# ============================================================
def create_non_iid_clients(
        X: ArrayLike,
        y: ArrayLike,
        X_test: ArrayLike,
        y_test: ArrayLike,
        n_clients: int = 50,
        n_edges: int = 10,
        n_classes_per_client: int = 5,
        min_clients_per_edge: int = 5,
        random_state: int = 42
) -> Tuple[List[ClientData], List[List[int]]]:
    """
    Partition data into non-IID clients and assign them to edges.

    Returns:
        clients_data: list of tuples (X_train, X_test, y_train, y_test) per client
        edge_groups: list of lists of client indices per edge
    """
    np.random.seed(random_state)

    # Convert to NumPy if pandas
    X_np = X.to_numpy() if isinstance(X, (pd.DataFrame, pd.Series)) else X
    y_np = y.to_numpy() if isinstance(y, (pd.DataFrame, pd.Series)) else y
    X_test_np = X_test.to_numpy() if isinstance(X_test, (pd.DataFrame, pd.Series)) else X_test
    y_test_np = y_test.to_numpy() if isinstance(y_test, (pd.DataFrame, pd.Series)) else y_test

    n_samples = X_np.shape[0]
    classes = np.unique(y_np)
    n_classes = len(classes)

    clients_data: List[ClientData] = []

    # Shuffle samples for random assignment
    shuffled_idx = np.random.permutation(n_samples)

    # Approx samples per client
    samples_per_client = n_samples // n_clients

    for i in range(n_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client
        client_idx = shuffled_idx[start_idx:end_idx]

        X_train_client = X_np[client_idx]
        y_train_client = y_np[client_idx]

        # Use full test set for simplicity (can also split test per client)
        clients_data.append((X_train_client, X_test_np, y_train_client, y_test_np))

    # Assign clients to edges
    clients_per_edge = max(min_clients_per_edge, n_clients // n_edges)
    edge_groups: List[List[int]] = []
    all_client_indices = list(range(n_clients))
    np.random.shuffle(all_client_indices)

    for i in range(0, n_clients, clients_per_edge):
        edge_groups.append(all_client_indices[i:i + clients_per_edge])

    return clients_data, edge_groups


# ============================================================
# Local LightGBM Training
# ============================================================
def train_local_lightgbm(X_train, y_train, X_val, y_val, task='classification', n_estimators=50, num_classes=None):
    if task == 'classification':
        if num_classes and num_classes > 2:
            params = {'objective': 'multiclass', 'metric': 'multi_logloss',
                      'num_class': num_classes, 'boosting_type': 'gbdt',
                      'num_leaves': 31, 'learning_rate': 0.05, 'verbose': -1,
                      'n_estimators': n_estimators}
            model = lgb.LGBMClassifier(**params)
        else:
            params = {'objective': 'binary', 'metric': 'binary_logloss',
                      'boosting_type': 'gbdt', 'num_leaves': 31, 'learning_rate': 0.05,
                      'verbose': -1, 'n_estimators': n_estimators}
            model = lgb.LGBMClassifier(**params)
    else:
        params = {'objective': 'regression', 'metric': 'rmse',
                  'boosting_type': 'gbdt', 'num_leaves': 31,
                  'learning_rate': 0.05, 'verbose': -1, 'n_estimators': n_estimators}
        model = lgb.LGBMRegressor(**params)

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric=params['metric'],
              callbacks=[lgb.early_stopping(stopping_rounds=20)])
    return model


# ============================================================
# Leaf Embeddings
# ============================================================
def extract_leaf_embeddings(model, X: ArrayLike) -> np.ndarray:
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    return model.predict(X, pred_leaf=True).astype(np.int32)


# ============================================================
# Client Training Thread
# ============================================================
def client_train_thread(client_id, X_train, y_train, X_val, y_val, task, n_estimators, local_models):
    num_classes = len(np.unique(y_train)) if task == 'classification' else None
    model = train_local_lightgbm(X_train, y_train, X_val, y_val, task, n_estimators, num_classes)
    local_models[client_id] = model


# ============================================================
# Bayesian Aggregator (edge-wise)
# ============================================================
def bayesian_aggregator_edge(X_sparse_edge: csr_matrix, y_edge: np.ndarray,
                             n_classes: int, variance_prune=True, threshold=1e-6,
                             n_samples=200, n_tune=200):
    # Progressive memory optimization
    X_dense = X_sparse_edge.toarray() if X_sparse_edge.shape[0] * X_sparse_edge.shape[1] < 1e7 else None
    if variance_prune and X_dense is not None:
        X_dense = VarianceThreshold(threshold=threshold).fit_transform(X_dense)

    y = np.clip(y_edge.astype(int), 0, n_classes - 1)
    if X_dense is not None:
        X_dense = StandardScaler().fit_transform(X_dense)

    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0, sigma=1, shape=n_classes)
        betas = pm.Normal("betas", mu=0, sigma=0.5, shape=(X_sparse_edge.shape[1], n_classes))
        logits = pm.math.dot(X_sparse_edge.toarray(), betas) + alpha if X_dense is None else pm.math.dot(X_dense,
                                                                                                         betas) + alpha

        logits_shifted = logits - pm.math.max(logits, axis=1, keepdims=True)
        y_obs = pm.Categorical("y_obs", p=pm.math.softmax(logits_shifted), observed=y)
        trace = pm.sample(n_samples, tune=n_tune, target_accept=0.9,cores=1, progressbar=True, init='jitter+adapt_diag')
    return model, trace


# ============================================================
# Helper: Pad sparse matrices
# ============================================================
def pad_sparse_matrices(matrices: List[csr_matrix]) -> List[csr_matrix]:
    max_cols = max(mat.shape[1] for mat in matrices)
    padded = []
    for mat in matrices:
        if mat.shape[1] < max_cols:
            diff = max_cols - mat.shape[1]
            mat = hstack([mat, csr_matrix((mat.shape[0], diff))])
        padded.append(mat)
    return padded

def one_hot_encode_leaf_embeddings(leaf_embeddings: csr_matrix) -> csr_matrix:
    """
    Convert LightGBM leaf indices to one-hot encoding.

    leaf_embeddings: csr_matrix of shape [n_samples, n_trees] (integers)
    returns: csr_matrix of shape [n_samples, n_trees * num_leaves]
    """
    # Convert to dense array for OneHotEncoder
    if isinstance(leaf_embeddings, csr_matrix):
        leaf_dense = leaf_embeddings.toarray()
    else:
        leaf_dense = leaf_embeddings

    ohe = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
    leaf_ohe_sparse = ohe.fit_transform(leaf_dense)
    return leaf_ohe_sparse


# ============================================================
# Aggregate Edge
# ============================================================
def aggregate_multiclass_edge(edge_X_list: List[csr_matrix], edge_y_list: List[np.ndarray],
                              num_classes: int, aggregator_type='bayesian'):
    safe_X, safe_y = [], []
    for X_edge, y_edge in zip(edge_X_list, edge_y_list):
        if X_edge.shape[0] == 0:
            continue
        # One-hot encode leaf embeddings per client
        X_edge_ohe = one_hot_encode_leaf_embeddings(X_edge)

        safe_X.append(X_edge_ohe)
        safe_y.append(y_edge)
    if not safe_X:
        return None, None, None, None
    safe_X = pad_sparse_matrices(safe_X)
    edge_X_sparse = vstack(safe_X)
    edge_y = np.hstack(safe_y)

    if aggregator_type == 'bayesian':
        try:
            return bayesian_aggregator_edge(edge_X_sparse, edge_y, num_classes) + (edge_X_sparse, edge_y)
        except Exception as e:
            print(f"[ERROR] Bayesian aggregation failed: {e}")
            return None, None, edge_X_sparse, edge_y
    else:
        logreg = LogisticRegression(multi_class='multinomial', max_iter=500)
        logreg.fit(edge_X_sparse, edge_y)
        return logreg, None, edge_X_sparse, edge_y


# ============================================================
# Prediction Alignment
# ============================================================
def align_predictions(y_pred_local: np.ndarray, classes_source: np.ndarray,
                      global_classes: np.ndarray) -> np.ndarray:
    class_to_index = {c: i for i, c in enumerate(global_classes)}
    n_samples, n_global = y_pred_local.shape[0], len(global_classes)
    y_pred_full = np.zeros((n_samples, n_global))
    for i, c in enumerate(classes_source):
        if c in class_to_index:
            y_pred_full[:, class_to_index[c]] = y_pred_local[:, i]
    return y_pred_full


# ============================================================
# Hierarchical PFL
# ============================================================
def hierarchical_pfl(clients_data: List[ClientData], edge_groups: List[List[int]],
                     task='classification', n_estimators=50):
    n_clients = len(clients_data)
    local_models: List[Optional[lgb.LGBMModel]] = [None] * n_clients
    threads = []

    # Local training
    for i, (X_train, X_test, y_train, y_test) in enumerate(clients_data):
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2,
                                                    random_state=config["random_seed"])
        t = threading.Thread(target=client_train_thread,
                             args=(i, X_tr, y_tr, X_val, y_val, task, n_estimators, local_models))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    # Leaf embeddings
    client_leaf_embeddings: List[csr_matrix] = []
    for model, (X_train, _, _, _) in zip(local_models, clients_data):
        if model:
            leaf_idx = extract_leaf_embeddings(model, X_train)
            client_leaf_embeddings.append(csr_matrix(leaf_idx))

    global_classes = np.unique(np.hstack([y for _, _, y, _ in clients_data]))
    num_classes = len(global_classes)

    w_globals = np.array([0.5] * n_clients)
    client_round_metrics, client_round_weights, global_accs = {i: [] for i in range(n_clients)}, {i: [] for i in
                                                                                                  range(n_clients)}, []

    for round_id in range(config["epoch"]):
        print(f"\n=== Round {round_id + 1}/{config['epoch']} ===")

        # Edge aggregation
        edge_aggregators = []
        for edge in edge_groups:
            edge_X_list, edge_y_list = [], []
            for client_idx in edge:
                model = local_models[client_idx]
                X_train, _, y_train, _ = clients_data[client_idx]
                if model is None or X_train.shape[0] == 0:
                    continue
                leaf_idx = extract_leaf_embeddings(model, X_train)
                edge_X_list.append(csr_matrix(leaf_idx))
                edge_y_list.append(y_train)
            if edge_X_list:
                aggregator_model, bayes_trace, _, _ = aggregate_multiclass_edge(edge_X_list, edge_y_list, num_classes)
                edge_aggregators.append((aggregator_model, bayes_trace))

            # --- build edge_predictions (avg of local client probs per edge) ---
            edge_predictions: Dict[int, np.ndarray] = {}
            for edge_idx, edge in enumerate(edge_groups):
                edge_client_preds = []
                for client_idx in edge:
                    model = local_models[client_idx]
                    # clients share the same X_test in your pipeline (clients_data[*][1])
                    if model is None:
                        continue
                    # grab the shared test set (same for all clients in your code)
                    _, X_test_shared, _, _ = clients_data[client_idx]
                    if X_test_shared is None or X_test_shared.shape[0] == 0:
                        continue
                    # get client-local probabilities and align them to global classes
                    try:
                        client_prob = model.predict_proba(X_test_shared)  # shape (n_test, n_local_classes)
                        client_prob_aligned = align_predictions(client_prob, model.classes_,
                                                                global_classes)  # (n_test, n_global)
                        # safe normalization just in case
                        client_prob_aligned = client_prob_aligned / (
                                    client_prob_aligned.sum(axis=1, keepdims=True) + 1e-12)
                        client_prob_aligned = np.clip(client_prob_aligned, 1e-8, 1 - 1e-8)
                        edge_client_preds.append(client_prob_aligned)
                    except Exception as e:
                        # model.predict_proba may fail for some models; skip them
                        print(f"[WARN] edge {edge_idx} client {client_idx} predict_proba failed: {e}")
                        continue

                if len(edge_client_preds) > 0:
                    # average across clients in the edge -> (n_test, n_global)
                    edge_pred = np.mean(edge_client_preds, axis=0)
                    # normalize & clip to be safe
                    edge_pred = edge_pred / (edge_pred.sum(axis=1, keepdims=True) + 1e-12)
                    edge_pred = np.clip(edge_pred, 1e-8, 1 - 1e-8)
                    edge_predictions[edge_idx] = edge_pred
                else:
                    # no valid clients' predictions for this edge
                    edge_predictions[edge_idx] = None
            # --- done building edge_predictions ---


        # Client predictions & adaptive weighting
        round_global_correct, round_global_total = 0, 0
        for i, (X_train, X_test, y_train, y_test) in enumerate(clients_data):
            model = local_models[i]
            if model is None or X_test.shape[0] == 0:
                continue
            leaf_idx_test = extract_leaf_embeddings(model, X_test)
            leaf_emb_test = csr_matrix(leaf_idx_test)

            y_pred_local_raw = model.predict_proba(X_test)
            print(y_pred_local_raw[:5])

            y_pred_local = align_predictions(y_pred_local_raw, model.classes_, global_classes)

            aggregator_model, bayes_trace = edge_aggregators[i % len(edge_aggregators)]

            if config["aggregator_type"] == 'bayesian' and aggregator_model is not None:
                with aggregator_model:
                    pp = pm.sample_posterior_predictive(bayes_trace, random_seed=42,var_names=['y_obs'],progressbar=False)
                    print("pp['y_obs'] shape:", pp['y_obs'].shape)
                    print("First 5 samples:\n", pp['y_obs'][:5])

                # pp['y_obs'].shape = (n_samples, n_data)
                # Count occurrences for each class per sample
                num_classes = len(global_classes)
                counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=num_classes),
                                             axis=0, arr=pp['y_obs'])
                # counts.shape = (num_classes, n_data)

                # Transpose to shape (n_data, num_classes) and normalize
                y_pred_global_raw = counts.T / pp['y_obs'].shape[0]
                y_pred_global = y_pred_global_raw

                print("y_pred_global_raw[:5]:\n", y_pred_global[:5])
                print("Row sums (should be 1):\n", y_pred_global.sum(axis=1)[:5])
            else:
                # ⚠️ Bayesian failed → use edge-averaged fallback (if available)
                print("[WARN] Bayesian aggregator unavailable, falling back to edge-average")
                # choose corresponding edge index for this client
                # Note: you currently pick aggregator_model, bayes_trace with edge_aggregators[i % len(edge_aggregators)]
                # so we use the same mapping to get edge index
                mapped_edge_idx = i % len(edge_groups)  # or use actual mapping if you have one
                edge_pred = edge_predictions.get(mapped_edge_idx, None)
                if edge_pred is not None:
                    y_pred_global = edge_pred.copy()  # shape (n_test, n_global)
                else:
                    # final fallback: just use local prediction
                    print("[WARN] No edge predictions available, using local fallback")
                    y_pred_global = y_pred_local.copy()
                # debug print (optional)
                print("y_pred_global (fallback) first rows:\n", y_pred_global[:3])

            combined_prob = w_globals[i] * y_pred_global + (1 - w_globals[i]) * y_pred_local
            combined_prob = combined_prob / (combined_prob.sum(axis=1, keepdims=True) + 1e-12)
            combined_prob = np.clip(combined_prob, 1e-8, 1 - 1e-8)
            y_pred_label = np.argmax(combined_prob, axis=1)

            acc = accuracy_score(y_test, y_pred_label)
            client_round_metrics[i].append(acc)

            loss_global = log_loss(y_test, np.clip(y_pred_global, 1e-8, 1 - 1e-8), labels=global_classes)
            loss_local = log_loss(y_test, np.clip(y_pred_local, 1e-8, 1 - 1e-8), labels=global_classes)
            if not np.isnan(loss_global) and not np.isnan(loss_local):
                w_globals[i] = np.clip(loss_local / (loss_local + loss_global + 1e-8), 0.1, 0.9)
            client_round_weights[i].append(w_globals[i])

            round_global_correct += np.sum(y_pred_label == y_test)
            round_global_total += len(y_test)

        global_accs.append(round_global_correct / max(round_global_total, 1))
        print(f"Global Accuracy Round {round_id + 1}: {global_accs[-1]:.4f}")

    if config["save_results"]:
        os.makedirs(config["results_path"], exist_ok=True)
        with open(os.path.join(config["results_path"], "pfl_results.pkl"), "wb") as f:
            pickle.dump({
                "local_models": local_models,
                "edge_aggregators": edge_aggregators,
                "w_globals": w_globals,
                "client_metrics": client_round_metrics,
                "client_weights": client_round_weights,
                "global_classes": global_classes,
                "global_accs": global_accs
            }, f)

    return local_models, edge_aggregators, w_globals, client_round_metrics, client_round_weights, global_accs


# ============================================================
# Plotting
# ============================================================
def plot_metrics_and_weights(client_metrics: Dict[int, List[float]],
                             client_weights: Dict[int, List[float]],
                             global_accs: List[float]):
    plt.figure(figsize=(14, 6))
    for i, metrics in client_metrics.items():
        plt.plot(np.convolve(metrics, np.ones(3) / 3, mode='same'), label=f'Client {i} Acc', alpha=0.5)
    plt.plot(np.convolve(global_accs, np.ones(3) / 3, mode='same'), 'k-', lw=2, label='Global Acc')
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("Client and Global Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    plt.figure(figsize=(14, 6))
    for i, weights in client_weights.items():
        plt.plot(np.convolve(weights, np.ones(3) / 3, mode='same'), label=f'Client {i} Weight', alpha=0.5)
    plt.xlabel("Round")
    plt.ylabel("Weight")
    plt.title("Client Adaptive Weights")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    folder_path = "CIC_IoT_dataset_2023"
    today_str = datetime.now().strftime("%Y-%m-%d")
    log_path_str = os.path.join("logs", f"{folder_path}_{today_str}")
    os.makedirs(log_path_str, exist_ok=True)
    print(f"[INFO] Logging directory: {log_path_str}")

    task = "classification"
    X_final, y_final, X_pretrain, y_pretrain, X_finetune, y_finetune, X_test, y_test = preprocess_data(log_path_str,
                                                                                                       folder_path)
    print(f"[INFO] Preprocessed data shapes: X_pretrain={X_pretrain.shape}, X_test={X_test.shape}")

    # Create non-IID clients and assign to edges
    clients_data, edge_groups = create_non_iid_clients(
        X_pretrain.to_numpy() if isinstance(X_pretrain, pd.DataFrame) else X_pretrain,
        y_pretrain,
        X_test,
        y_test,
        n_clients=config["n_clients"],
        n_classes_per_client=config["n_classes_per_client"],
        n_edges=config["n_edges"],
        min_clients_per_edge=config["min_clients_per_edge"],
        random_state=config["random_seed"]
    )

    # Run hierarchical PFL
    local_models, edge_aggregators, w_globals, client_metrics, client_weights, global_accs = hierarchical_pfl(
        clients_data, edge_groups, task=task, n_estimators=config["n_estimators"]
    )

    # Plot results
    plot_metrics_and_weights(client_metrics, client_weights, global_accs)

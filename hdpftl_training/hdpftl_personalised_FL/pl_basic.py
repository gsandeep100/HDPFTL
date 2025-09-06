# ============================================================
# Imports
# ============================================================
import threading
from typing import Optional, List, Tuple, Union, Dict

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import scipy as sp
from scipy.sparse import vstack, csr_matrix
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import hdpftl_training.hdpftl_pipeline as pipeline

import hdpftl_utility.config as config
import hdpftl_utility.log as log_util
import hdpftl_utility.utils as util
from hdpftl_training.hdpftl_data import preprocess

# ============================================================
# Type Aliases
# ============================================================
ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series]
ClientData = Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]

# ============================================================
# Global Config
# ============================================================
np.random.seed(42)
n_edges:int = 10
n_clients: int = 50
n_classes_per_client: int = 5
epoch: int = 10
edge_data: List[List[int]] = []


# ============================================================
# Preprocessing
# ============================================================
def preprocess_for_lightgbm_fast(
    X_train: ArrayLike, X_test: ArrayLike
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]:
    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train)
    if isinstance(X_test, np.ndarray):
        X_test = pd.DataFrame(X_test)

    encoders: Dict[str, List[str]] = {}
    for col in X_train.columns:
        if X_train[col].dtype == "object" or X_test[col].dtype == "object":
            X_train[col] = X_train[col].astype("category")
            X_test[col] = X_test[col].astype("category")
            encoders[col] = list(X_train[col].cat.categories)
    return X_train, X_test, encoders

# ============================================================
# Local LightGBM Training
# ============================================================
def train_local_lightgbm(
    X_train: pd.DataFrame,
    y_train: ArrayLike,
    X_val: pd.DataFrame,
    y_val: ArrayLike,
    task: str = 'classification',
    n_estimators: int = 100,
    num_classes: Optional[int] = None
) -> lgb.LGBMModel:
    if task == 'classification':
        if num_classes and num_classes > 2:
            params = {'objective': 'multiclass', 'metric': 'multi_logloss', 'num_class': num_classes,
                      'boosting_type': 'gbdt', 'num_leaves': 31, 'learning_rate': 0.05, 'verbose': -1,
                      'n_estimators': n_estimators}
        else:
            params = {'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt', 'num_leaves': 31,
                      'learning_rate': 0.05, 'verbose': -1, 'n_estimators': n_estimators}
        model = lgb.LGBMClassifier(**params)
    else:
        params = {'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt', 'num_leaves': 31,
                  'learning_rate': 0.05, 'verbose': -1, 'n_estimators': n_estimators}
        model = lgb.LGBMRegressor(**params)

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric=params['metric'],
              callbacks=[lgb.early_stopping(stopping_rounds=20)])
    return model

# ============================================================
# Leaf Embeddings
# ============================================================
def extract_leaf_embeddings(model: Optional[lgb.LGBMClassifier], X: ArrayLike) -> np.ndarray:
    if model is None:
        raise ValueError("Model is None, cannot extract leaf embeddings")
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    leaf_indices: np.ndarray = model.predict(X, pred_leaf=True)
    return np.array(leaf_indices, dtype=np.int32)

# ============================================================
# Client Training Thread
# ============================================================
def client_train_thread(
    client_id: int,
    X_train: ArrayLike,
    y_train: ArrayLike,
    X_val: ArrayLike,
    y_val: ArrayLike,
    task: str,
    n_estimators: int,
    local_models: List[Optional[lgb.LGBMClassifier]],
    done_event: threading.Event
) -> None:
    try:
        X_train_enc, X_val_enc, _ = preprocess_for_lightgbm_fast(X_train, X_val)
        num_classes = len(np.unique(y_train)) if task == 'classification' else None
        model = train_local_lightgbm(X_train_enc, y_train, X_val_enc, y_val, task, n_estimators, num_classes)
        local_models[client_id] = model
    finally:
        done_event.set()

# ============================================================
# Bayesian Aggregator Edge
# ============================================================
def bayesian_aggregator_multiclass_edge(
    X_sparse_edge: csr_matrix,
    y_edge: np.ndarray,
    n_classes: int,
    n_samples: int = 200,
    n_tune: int = 200
) -> Tuple[pm.Model, pm.backends.base.MultiTrace, np.ndarray]:

    # 1. Prune zero/low-variance columns
    pruner = VarianceThreshold(threshold=1e-6)
    X_pruned = pruner.fit_transform(X_sparse_edge)
    non_zero_cols = pruner.get_support()

    if X_pruned.shape[1] == 0:
        raise ValueError("All features removed after pruning (zero/low variance)")

    # 2. Convert to dense
    X_dense = X_pruned.toarray()

    # 3. Clip labels
    y = np.clip(y_edge.astype(int), 0, n_classes - 1)

    # 4. Optional scaling
    X_dense = StandardScaler().fit_transform(X_dense)

    with pm.Model() as model:
        # safer priors
        alpha = pm.Normal("alpha", mu=0, sigma=1, shape=n_classes)
        betas = pm.Normal("betas", mu=0, sigma=0.5, shape=(X_dense.shape[1], n_classes))

        logits = pm.math.dot(X_dense, betas) + alpha
        y_obs = pm.Categorical("y_obs", p=pm.math.softmax(logits), observed=y)

        trace = pm.sample(
            n_samples,
            tune=n_tune,
            target_accept=0.9,
            cores=1,
            progressbar=True,
            init='adapt_diag'
        )

    return model, trace, non_zero_cols

def safe_sparse_to_dense(X_sparse):
    """
    Convert sparse matrix to dense safely.
    """
    if sp.issparse(X_sparse):
        return X_sparse.toarray()
    return X_sparse

# ============================================================
# Aggregator Selector
# ============================================================
def aggregate_multiclass_edge(
    edge_X_list: List[csr_matrix],
    edge_y_list: List[np.ndarray],
    num_classes: int,
    aggregator_type: str = 'bayesian',
    min_samples_per_edge: int = 5
) -> Tuple[Optional[Union[LogisticRegression, pm.Model]],
           Optional[pm.backends.base.MultiTrace],
           Optional[csr_matrix],
           Optional[np.ndarray]]:

    safe_edge_X, safe_edge_y = [], []

    # Filter edges with too few samples
    for X_edge, y_edge in zip(edge_X_list, edge_y_list):
        if X_edge.shape[0] < min_samples_per_edge:
            print(f"[WARN] Skipping edge with {X_edge.shape[0]} samples (too few)")
            continue
        safe_edge_X.append(X_edge)
        safe_edge_y.append(y_edge)

    if not safe_edge_X:
        print("[WARN] No valid edges left after filtering. Returning None.")
        return None, None, None, None

    edge_X_sparse: csr_matrix = vstack(safe_edge_X)
    edge_y: np.ndarray = np.hstack(safe_edge_y)

    if aggregator_type == 'bayesian':
        try:
            model, trace = bayesian_aggregator_multiclass_edge(edge_X_sparse, edge_y, num_classes)
        except Exception as e:
            print(f"[ERROR] Bayesian aggregation failed: {e}")
            model, trace = None, None
        return model, trace, edge_X_sparse, edge_y

    elif aggregator_type == 'logreg':
        logreg = LogisticRegression(multi_class='multinomial', max_iter=500)
        logreg.fit(edge_X_sparse, edge_y)
        return logreg, None, edge_X_sparse, edge_y

    elif aggregator_type == 'simple':
        avg_X = edge_X_sparse.mean(axis=0)
        return None, None, avg_X, edge_y

    elif aggregator_type == 'weighted':
        weights = np.ones(len(safe_edge_X)) / len(safe_edge_X)
        weighted_X = sum(w * x for w, x in zip(weights, safe_edge_X))
        return None, None, weighted_X, edge_y

    else:
        raise ValueError(f"Unknown aggregator_type {aggregator_type}")

# ============================================================
# Align Predictions
# ============================================================
def align_predictions(
    y_pred_local: np.ndarray,
    classes_source: np.ndarray,
    global_classes: np.ndarray
) -> np.ndarray:
    class_to_index: Dict[int, int] = {c: i for i, c in enumerate(global_classes)}
    n_samples, n_global = y_pred_local.shape[0], len(global_classes)
    y_pred_full: np.ndarray = np.zeros((n_samples, n_global))
    for i, c in enumerate(classes_source):
        if c in class_to_index:
            y_pred_full[:, class_to_index[c]] = y_pred_local[:, i]
    return y_pred_full


# ============================================================
# Hierarchical PFL
# ============================================================
def hierarchical_pfl(
    clients_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    edge_groups: List[List[int]],
    task: str = 'classification',
    n_estimators: int = 50,
    aggregator_type: str = 'bayesian',
):
    n_clients = len(clients_data)

    local_models: List[Optional[lgb.LGBMClassifier]] = [None] * n_clients
    threads: List[threading.Thread] = []
    events: List[threading.Event] = []

    # --------------------
    # Local training
    # --------------------
    for i, (X_train, X_test, y_train, y_test) in enumerate(clients_data):
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        done_event = threading.Event()
        t = threading.Thread(
            target=client_train_thread,
            args=(i, X_tr, y_tr, X_val, y_val, task, n_estimators, local_models, done_event)
        )
        threads.append(t)
        events.append(done_event)
        t.start()

    for e in events:
        e.wait()

    # --------------------
    # Prepare global OneHotEncoder for leaf embeddings
    # --------------------
    all_leaf_indices = []
    for model, (X_train, _, _, _) in zip(local_models, clients_data):
        if model is not None:
            leaf_idx = extract_leaf_embeddings(model, X_train)
            all_leaf_indices.append(leaf_idx)
    if not all_leaf_indices:
        raise RuntimeError("No valid leaf embeddings found from local models.")

    all_leaf_indices_stacked = np.vstack(all_leaf_indices)
    global_leaf_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    global_leaf_encoder.fit(all_leaf_indices_stacked)

    # Global classes
    global_classes = np.unique(np.hstack([y for _, _, y, _ in clients_data]))
    num_classes = len(global_classes)

    # Initialize adaptive weights
    w_globals = np.array([0.5] * n_clients)
    client_round_metrics: Dict[int, List[float]] = {i: [] for i in range(n_clients)}
    client_round_weights: Dict[int, List[float]] = {i: [] for i in range(n_clients)}
    global_accs: List[float] = []

    # --------------------
    # Federated rounds
    # --------------------
    for round_id in range(epoch):
        print(f"\n=== Round {round_id + 1}/{epoch} ===")

        # Build edge-level data
        edge_X_list, edge_y_list = [], []
        for edge in edge_groups:
            edge_client_X, edge_client_y = [], []
            for client_idx in edge:
                X_train, _, y_train, _ = clients_data[client_idx]
                model = local_models[client_idx]
                if model is None or X_train.shape[0] == 0:
                    continue
                leaf_idx = extract_leaf_embeddings(model, X_train)
                leaf_emb = global_leaf_encoder.transform(leaf_idx)
                edge_client_X.append(leaf_emb)
                edge_client_y.append(y_train)
            if edge_client_X:
                edge_X_list.append(vstack(edge_client_X))
                edge_y_list.append(np.hstack(edge_client_y))

        # Aggregate edges
        if edge_X_list:
            aggregator_model, bayes_trace, global_X_sparse, global_y = aggregate_multiclass_edge(
                edge_X_list, edge_y_list, num_classes, aggregator_type
            )
        else:
            aggregator_model, bayes_trace, global_X_sparse, global_y = None, None, None, None
            print("[WARN] No valid edges for aggregation this round.")

        # Evaluate per-client
        round_global_correct = 0
        round_global_total = 0

        for i, (X_train, X_test, y_train, y_test) in enumerate(clients_data):
            model = local_models[i]
            if model is None or X_test.shape[0] == 0:
                continue

            # Leaf embeddings for test
            leaf_idx_test = extract_leaf_embeddings(model, X_test)
            leaf_emb_test = global_leaf_encoder.transform(leaf_idx_test)

            # Local prediction
            X_test_dense = X_test.to_numpy() if isinstance(X_test, pd.DataFrame) else np.array(X_test)
            y_pred_local_raw = model.predict_proba(X_test_dense)
            y_pred_local = align_predictions(y_pred_local_raw, model.classes_, global_classes)

            # Global prediction
            if aggregator_type == 'bayesian' and aggregator_model is not None:
                with aggregator_model:
                    pp = pm.sample_posterior_predictive(bayes_trace, var_names=['y_obs'], samples=50, progressbar=False)
                y_pred_global_raw = np.zeros((X_test_dense.shape[0], len(global_classes)))
                for c in global_classes:
                    y_pred_global_raw[:, c] = np.mean(pp['y_obs'] == c, axis=0)
                y_pred_global = y_pred_global_raw
            elif aggregator_type == 'logreg' and aggregator_model is not None:
                y_pred_global_raw = aggregator_model.predict_proba(leaf_emb_test)
                y_pred_global = align_predictions(y_pred_global_raw, aggregator_model.classes_, global_classes)
            else:
                y_pred_global = np.zeros_like(y_pred_local)

            # Combine
            combined_prob = w_globals[i] * y_pred_global + (1 - w_globals[i]) * y_pred_local

            # Ensure probabilities sum to 1
            combined_prob /= combined_prob.sum(axis=1, keepdims=True)

            y_pred_label = np.argmax(combined_prob, axis=1)
            acc = accuracy_score(y_test, y_pred_label)
            client_round_metrics[i].append(acc)

            # Losses
            loss_global = log_loss(y_test, np.clip(y_pred_global, 1e-8, 1 - 1e-8), labels=global_classes)
            loss_local = log_loss(y_test, np.clip(y_pred_local, 1e-8, 1 - 1e-8), labels=global_classes)
            w_globals[i] = np.clip(loss_local / (loss_local + loss_global + 1e-8), 0.1, 0.9)
            client_round_weights[i].append(w_globals[i])

            # Global accuracy for this round
            round_global_correct += np.sum(y_pred_label == y_test)
            round_global_total += len(y_test)

        global_accs.append(round_global_correct / max(round_global_total, 1))
        print("Adaptive weights:", np.round(w_globals, 3))
        print(f"Global Accuracy Round {round_id + 1}: {global_accs[-1]:.4f}")

    return local_models, aggregator_model, bayes_trace, w_globals, client_round_metrics, client_round_weights, global_classes, global_accs


# ============================================================
# Visualization
# ============================================================
# ============================================================
# Visualization with global accuracy
# ============================================================
# ============================================================
# Visualization with secondary y-axis for global accuracy
# ============================================================
def plot_metrics_and_weights(
        round_metrics: Dict[int, List[float]],
        round_weights: Dict[int, List[float]],
        global_metrics: Optional[List[float]] = None
) -> None:
    n_clients = len(round_metrics)
    rounds = range(len(next(iter(round_metrics.values()))))

    # --- Accuracy plot ---
    fig, ax1 = plt.subplots(figsize=(14, 5))

    # Plot per-client accuracy
    for i in range(n_clients):
        ax1.plot(rounds, round_metrics[i], label=f'Client-{i} Accuracy', alpha=0.7)
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Client Accuracy")
    ax1.set_title("Per-Client Accuracy per Round")
    ax1.grid(True)

    # Plot global accuracy on secondary y-axis
    if global_metrics is not None:
        ax2 = ax1.twinx()
        ax2.plot(rounds, global_metrics, label='Global Accuracy', color='black', linewidth=2, linestyle='--')
        ax2.set_ylabel("Global Accuracy")

    # Combine legends
    handles1, labels1 = ax1.get_legend_handles_labels()
    if global_metrics is not None:
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc='lower right')
    else:
        ax1.legend(loc='lower right')

    plt.show()

    # --- Adaptive weights plot ---
    plt.figure(figsize=(14, 5))
    for i in range(n_clients):
        plt.plot(round_weights[i], label=f'Client-{i} Weight', alpha=0.7)
    plt.xlabel("Round")
    plt.ylabel("Adaptive Weight")
    plt.title("Adaptive Weights per Round")
    plt.legend()
    plt.grid(True)
    plt.show()


# ============================================================
# Client Creation
# ============================================================
def create_non_iid_clients_with_random_edges(
    X, y, X_test, y_test,
    n_edges: int,
    min_clients_per_edge: int = 5,
    random_state: Optional[int] = None
) -> Tuple[List[ClientData], List[List[int]]]:
    """
    1. Create non-IID clients.
    2. Randomly assign clients to a fixed number of edges.
       - Each edge gets at least min_clients_per_edge clients.
       - Clients are unique (not repeated across edges).

    Returns:
        clients_data: list of client data (X_train, X_test, y_train, y_test)
        edge_list: list of edges, each edge = list of client indices
    """
    if random_state is not None:
        np.random.seed(random_state)

    # --- Non-IID clients creation ---
    y_arr = y.values if isinstance(y, pd.Series) else y
    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    unique_classes = np.unique(y_arr)
    class_indices = {c: np.where(y_arr == c)[0] for c in unique_classes}

    clients_data: List[ClientData] = []

    for _ in range(n_clients):
        selected_classes = np.random.choice(unique_classes, n_classes_per_client, replace=False)
        client_idx = np.hstack([class_indices[c] for c in selected_classes])
        np.random.shuffle(client_idx)

        X_client = X_arr[client_idx]
        y_client = y_arr[client_idx]

        if isinstance(X, pd.DataFrame):
            X_client = pd.DataFrame(X_client, columns=X.columns)
        if isinstance(y, pd.Series):
            y_client = pd.Series(y_client, name=y.name)

        clients_data.append((X_client, X_test, y_client, y_test))

    # --- Random edges creation (fixed number of edges) ---
    n_clients_actual = len(clients_data)
    client_indices = np.arange(n_clients_actual)
    np.random.shuffle(client_indices)

    # Calculate remaining clients after assigning min_clients_per_edge to each edge
    remaining_clients = n_clients_actual - n_edges * min_clients_per_edge
    if remaining_clients < 0:
        raise ValueError("Not enough clients to satisfy min_clients_per_edge for all edges")

    # Start with min_clients_per_edge per edge
    edge_sizes = [min_clients_per_edge] * n_edges

    # Randomly distribute remaining clients across edges
    for _ in range(remaining_clients):
        idx = np.random.randint(0, n_edges)
        edge_sizes[idx] += 1

    # Assign clients to edges
    edge_list: List[List[int]] = []
    start = 0
    for size in edge_sizes:
        edge_clients = list(client_indices[start:start + size])
        edge_list.append(edge_clients)
        start += size

    return clients_data, edge_list


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    folder_path = "CIC_IoT_dataset_2023"
    log_path_str = config.LOGS_DIR_TEMPLATE.substitute(
        dataset=folder_path, date=util.get_today_date()
    )
    util.is_folder_exist(log_path_str)
    log_util.setup_logging(log_path_str)

    task = "classification"
    X_final, y_final, X_pretrain, y_pretrain, X_finetune, y_finetune, X_test, y_test = preprocess.preprocess_data(
        log_path_str, folder_path
    )

    clients_data, edge_groups = create_non_iid_clients_with_random_edges(
        X_pretrain, y_pretrain, X_test, y_test, n_edges=n_edges,
    min_clients_per_edge=5,random_state=42
    )
    print(edge_groups)
    """
    clients_data, hierarchical_data = pipeline.dirichlet_partition_for_edges_clients(
        X_pretrain, y_pretrain, edge_groups
    )
    client_data_dict_test, hierarchical_data_test = pipeline.dirichlet_partition_for_edges_clients(
        X_test, y_test, edge_groups
    )
"""
    local_models, aggregator_model, bayes_trace, w_globals, client_metrics, client_weights, global_classes, global_accs = hierarchical_pfl(
        clients_data, edge_groups, task=task, n_estimators=50
    )

    plot_metrics_and_weights(client_metrics, client_weights, global_metrics=global_accs)

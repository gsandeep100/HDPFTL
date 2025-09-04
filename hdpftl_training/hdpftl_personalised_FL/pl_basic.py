# ============================================================
# Imports
# ============================================================
import threading

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from scipy.sparse import vstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import hdpftl_utility.config as config
import hdpftl_utility.log as log_util
import hdpftl_utility.utils as util
from hdpftl_training.hdpftl_data import preprocess

np.random.seed(42)
n_clients = 100
n_classes_per_client = 10
rounds = 300


# ============================================================
# Preprocessing
# ============================================================
def preprocess_for_lightgbm_fast(X_train, X_test):
    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train)
    if isinstance(X_test, np.ndarray):
        X_test = pd.DataFrame(X_test)

    encoders = {}
    for col in X_train.columns:
        if X_train[col].dtype == "object" or X_test[col].dtype == "object":
            X_train[col] = X_train[col].astype("category")
            X_test[col] = X_test[col].astype("category")
            encoders[col] = list(X_train[col].cat.categories)
    return X_train, X_test, encoders


# ============================================================
# Local LightGBM Training
# ============================================================
def train_local_lightgbm(X_train, y_train, X_val, y_val, task='classification', n_estimators=100, num_classes=None):
    if task == 'classification':
        if num_classes and num_classes > 2:
            params = {'objective': 'multiclass', 'metric': 'multi_logloss', 'num_class': num_classes,
                      'boosting_type': 'gbdt', 'num_leaves': 31, 'learning_rate': 0.05, 'verbose': -1,
                      'n_estimators': n_estimators}
            model = lgb.LGBMClassifier(**params)
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
def extract_leaf_embeddings(model, X):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    leaf_indices = model.predict(X, pred_leaf=True)
    return np.array(leaf_indices, dtype=np.int32)


# ============================================================
# Client Training Thread
# ============================================================
def client_train_thread(client_id, X_train, y_train, X_val, y_val, task, n_estimators, local_models, done_event):
    try:
        X_train_enc, X_val_enc, _ = preprocess_for_lightgbm_fast(X_train, X_val)
        num_classes = len(np.unique(y_train)) if task == 'classification' else None
        model = train_local_lightgbm(X_train_enc, y_train, X_val_enc, y_val, task, n_estimators, num_classes)
        local_models[client_id] = model
    finally:
        done_event.set()


# ============================================================
# Bayesian Aggregator
# ============================================================
def bayesian_aggregator_multiclass_edge(X_sparse_edge, y_edge, n_classes, n_samples=200, n_tune=200):
    """
    Memory-efficient Bayesian aggregator for a single edge.
    Converts only the edge's sparse matrix to dense.
    """
    # Convert only this edge's sparse matrix to dense
    X_dense = X_sparse_edge.toarray()
    y = y_edge.astype(int)

    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0, sigma=10, shape=n_classes)
        betas = pm.Normal("betas", mu=0, sigma=1, shape=(X_dense.shape[1], n_classes))
        logits = pm.math.dot(X_dense, betas) + alpha
        y_obs = pm.Categorical("y_obs", p=pm.math.softmax(logits), observed=y)

        trace = pm.sample(n_samples, tune=n_tune, target_accept=0.9, cores=1, progressbar=True)

    return model, trace


# ============================================================
# Aggregator Selector
# ============================================================
def aggregate_multiclass_edge(edge_X_list, edge_y_list, num_classes, aggregator_type='bayesian'):
    """
    Aggregates all edges using the selected aggregator.
    Sparse matrices are preserved until Bayesian sampling.
    """
    edge_X_sparse = vstack(edge_X_list)  # Keep sparse
    edge_y = np.hstack(edge_y_list)

    if aggregator_type == 'bayesian':
        # Apply Bayesian aggregator edge-wise
        # Each edge individually converted to dense inside bayesian_aggregator_multiclass
        # Here we combine all edges for the aggregator
        model, trace = bayesian_aggregator_multiclass_edge(edge_X_sparse, edge_y, num_classes)
        return model, trace, edge_X_sparse, edge_y

    elif aggregator_type == 'logreg':
        logreg = LogisticRegression(multi_class='multinomial', max_iter=500)
        logreg.fit(edge_X_sparse, edge_y)
        return logreg, None, edge_X_sparse, edge_y

    elif aggregator_type == 'simple':
        avg_X = edge_X_sparse.mean(axis=0)
        return None, None, avg_X, edge_y

    elif aggregator_type == 'weighted':
        weights = np.ones(len(edge_X_list)) / len(edge_X_list)
        weighted_X = sum(w * x for w, x in zip(weights, edge_X_list))
        return None, None, weighted_X, edge_y

    else:
        raise ValueError(f"Unknown aggregator_type {aggregator_type}")


# ============================================================
# Align Predictions to Global Classes
# ============================================================
def align_predictions(y_pred_local, classes_source, global_classes):
    class_to_index = {c: i for i, c in enumerate(global_classes)}
    n_samples, n_global = y_pred_local.shape[0], len(global_classes)
    y_pred_full = np.zeros((n_samples, n_global))
    for i, c in enumerate(classes_source):
        if c in class_to_index:
            y_pred_full[:, class_to_index[c]] = y_pred_local[:, i]
    return y_pred_full


def create_fully_random_edge():
    """
    Create a fully random 2D list such that:
    - All numbers 0.total-1 appear exactly once
    - Rows have random lengths
    - Number of rows is random
    """
    numbers = np.arange(n_clients)
    np.random.shuffle(numbers)

    n_edges = []
    idx = 0
    while idx < n_clients:
        # Random row size: at least 1, at most remaining numbers
        row_size = np.random.randint(1, n_clients - idx + 1)
        n_edges.append(list(numbers[idx:idx + row_size]))
        idx += row_size

    return n_edges


# ============================================================
# Hierarchical PFL
# ============================================================
def hierarchical_pfl(clients_data, task='classification', n_estimators=50, aggregator_type='bayesian', rounds=10):

    n_clients = len(clients_data)
    local_models = [None] * n_clients
    threads, events = [], []

    edge_groups = create_fully_random_edge()
    print("Random Edge groups:", edge_groups)

    # --- Local training ---
    for i, (X_train, X_test, y_train, y_test) in enumerate(clients_data):
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        done_event = threading.Event()
        t = threading.Thread(target=client_train_thread,
                             args=(i, X_tr, y_tr, X_val, y_val, task, n_estimators, local_models, done_event))
        threads.append(t)
        events.append(done_event)
        t.start()
    for e in events: e.wait()

    # --- Global OneHotEncoder ---
    all_leaf_indices = []
    for model, (X_train, _, _, _) in zip(local_models, clients_data):
        all_leaf_indices.append(extract_leaf_embeddings(model, X_train))
    all_leaf_indices = np.vstack(all_leaf_indices)
    global_leaf_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    global_leaf_encoder.fit(all_leaf_indices)

    # Global classes
    global_classes = np.unique(np.hstack([y for _, _, y, _ in clients_data]))
    num_classes = len(global_classes)

    w_globals = np.array([0.5] * n_clients)
    client_round_metrics = {i: [] for i in range(n_clients)}
    client_round_weights = {i: [] for i in range(n_clients)}

    # --- Multi-round FL ---
    for round_id in range(rounds):
        print(f"\n=== Round {round_id + 1}/{rounds} ===")

        # Edge-level aggregation
        edge_X_list, edge_y_list = [], []
        for edge in edge_groups:
            edge_client_X, edge_client_y = [], []
            for client_idx in edge:
                X_train, _, y_train, _ = clients_data[client_idx]
                leaf_idx = extract_leaf_embeddings(local_models[client_idx], X_train)
                leaf_emb = global_leaf_encoder.transform(leaf_idx)  # sparse
                edge_client_X.append(leaf_emb)
                edge_client_y.append(y_train)
            if edge_client_X:
                edge_X_list.append(vstack(edge_client_X))
                edge_y_list.append(np.hstack(edge_client_y))

        # Global aggregation
        aggregator_model, bayes_trace, global_X_sparse, global_y = aggregate_multiclass_edge(
            edge_X_list, edge_y_list, num_classes, aggregator_type
        )

        # --- Evaluation ---
        for i, (X_train, X_test, y_train, y_test) in enumerate(clients_data):
            leaf_idx_test = extract_leaf_embeddings(local_models[i], X_test)
            leaf_emb_test = global_leaf_encoder.transform(leaf_idx_test)

            X_test_dense = X_test.to_numpy() if isinstance(X_test, pd.DataFrame) else np.array(X_test)
            y_pred_local_raw = local_models[i].predict_proba(X_test_dense)
            local_classes = local_models[i].classes_
            y_pred_local = align_predictions(y_pred_local_raw, local_classes, global_classes)

            if aggregator_type == 'bayesian':
                with aggregator_model:
                    pp = pm.sample_posterior_predictive(bayes_trace, var_names=['y_obs'], samples=50, progressbar=False)
                y_pred_global_raw = np.zeros((X_test_dense.shape[0], len(global_classes)))
                for c in global_classes:
                    y_pred_global_raw[:, c] = np.mean(pp['y_obs'] == c, axis=0)
                y_pred_global = y_pred_global_raw

            elif aggregator_type == 'logreg':
                y_pred_global_raw = aggregator_model.predict_proba(leaf_emb_test)
                y_pred_global = align_predictions(y_pred_global_raw, aggregator_model.classes_, global_classes)
            else:
                y_pred_global = np.array(global_X_sparse.todense())

            # Combine local + global
            combined_prob = w_globals[i] * y_pred_global + (1 - w_globals[i]) * y_pred_local
            y_pred_label = np.argmax(combined_prob, axis=1)

            acc = accuracy_score(y_test, y_pred_label)
            client_round_metrics[i].append(acc)
            loss_global = log_loss(y_test, np.clip(y_pred_global, 1e-8, 1 - 1e-8), labels=global_classes)
            loss_local = log_loss(y_test, np.clip(y_pred_local, 1e-8, 1 - 1e-8), labels=global_classes)

            w_globals[i] = np.clip(loss_local / (loss_local + loss_global + 1e-8), 0.1, 0.9)
            client_round_weights[i].append(w_globals[i])

        print("Adaptive weights:", np.round(w_globals, 3))

    return local_models, aggregator_model, bayes_trace, w_globals, client_round_metrics, client_round_weights, global_classes


# ============================================================
# Visualization
# ============================================================
def plot_metrics_and_weights(round_metrics, round_weights):
    n_clients = len(round_metrics)
    plt.figure(figsize=(14, 5))
    for i in range(n_clients):
        plt.plot(round_metrics[i], label=f'Client-{i} Accuracy')
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Per-Client Accuracy per Round")
    plt.show()

    plt.figure(figsize=(14, 5))
    for i in range(n_clients):
        plt.plot(round_weights[i], label=f'Client-{i} Weight')
    plt.xlabel("Round")
    plt.ylabel("Adaptive Weight")
    plt.legend()
    plt.title("Adaptive Weights per Round")
    plt.show()


# ============================================================
# Client Creation
# ============================================================
def create_non_iid_clients(X, y, X_test, y_test, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    y_arr = y.values if isinstance(y, pd.Series) else y
    X_arr = X.values if isinstance(X, pd.DataFrame) else X

    unique_classes = np.unique(y_arr)
    class_indices = {c: np.where(y_arr == c)[0] for c in unique_classes}

    clients_data = []
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
    return clients_data


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

    clients_data = create_non_iid_clients(X_pretrain, y_pretrain, X_test, y_test)

    # edge_groups = [[0, 1], [2, 3, 4], [5, 6], [7, 8, 9]]

    # aggregator_type = 'logreg'  # or 'bayesian'
    local_models, aggregator_model, bayes_trace, w_globals, round_metrics, round_weights, global_classes = hierarchical_pfl(
        clients_data, task=task, n_estimators=50
    )

    plot_metrics_and_weights(round_metrics, round_weights)

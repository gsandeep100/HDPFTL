import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split


# -----------------------------
# Load multiple CSVs from a folder (common columns)
# -----------------------------
def load_client_data_from_folder(folder_path, num_clients=3, test_size=0.2, file_extension=".csv", random_state=42):
    X_list, y_list = [], []
    common_columns = None

    # Find common columns
    for fname in os.listdir(folder_path):
        if fname.endswith(file_extension):
            df = pd.read_csv(os.path.join(folder_path, fname))
            cols = df.columns[:-1]
            if common_columns is None:
                common_columns = set(cols)
            else:
                common_columns = common_columns.intersection(set(cols))

    if not common_columns:
        raise ValueError("No common feature columns found across CSVs!")

    common_columns = list(common_columns)
    print(f"Using {len(common_columns)} common features across all files.")

    # Load CSVs with only common columns + target
    for fname in os.listdir(folder_path):
        if fname.endswith(file_extension):
            df = pd.read_csv(os.path.join(folder_path, fname))
            df = df[common_columns + [df.columns[-1]]]
            X_list.append(df.iloc[:, :-1].values)
            y_list.append(df.iloc[:, -1].values)

    X_all = np.vstack(X_list)
    y_all = np.hstack(y_list)

    # Split among clients
    indices = np.arange(len(X_all))
    np.random.seed(random_state)
    np.random.shuffle(indices)
    splits = np.array_split(indices, num_clients)

    clients_data = []
    for split in splits:
        X_client, y_client = X_all[split], y_all[split]
        X_train, X_test, y_train, y_test = train_test_split(
            X_client, y_client, test_size=test_size, random_state=random_state
        )
        clients_data.append((X_train, X_test, y_train, y_test))
    return clients_data


# -----------------------------
# Convert LightGBM leaf indices -> one-hot embedding
# -----------------------------
def get_leaf_embeddings(model, X):
    leaf_indices = model.predict(X, pred_leaf=True)
    num_trees = leaf_indices.shape[1]
    embeddings = []
    for t in range(num_trees):
        num_leaves = np.max(leaf_indices[:, t]) + 1
        embeddings.append(np.eye(num_leaves)[leaf_indices[:, t]])
    return np.hstack(embeddings)


# -----------------------------
# Compute LightGBM loss for adaptive weighting
# -----------------------------
def compute_lgb_loss(y_true, y_pred, task='classification'):
    if task == 'classification':
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    else:
        loss = np.mean((y_true - y_pred) ** 2)
    return loss


# -----------------------------
# Client training function (threaded)
# -----------------------------
def client_local_training(client_id, X_train, y_train, n_global_trees, n_local_trees, task, local_models, done_event):
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    lgb_tr = lgb.Dataset(X_tr, y_tr)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_tr)

    params = {'objective': 'binary' if task == 'classification' else 'regression',
              'metric': 'binary_logloss' if task == 'classification' else 'rmse',
              'verbose': -1,
              'boosting_type': 'gbdt',
              'num_leaves': 31,
              'learning_rate': 0.1,
              'num_boost_round': n_global_trees + n_local_trees}

    model = lgb.train(params, lgb_tr, valid_sets=[lgb_val], early_stopping_rounds=10, verbose_eval=False)
    local_models[client_id] = model
    done_event.set()


# -----------------------------
# Multi-round threaded PFL with adaptive weighting
# -----------------------------
def multi_round_pfl_adaptive_iterative(folder_path, task='classification', num_clients=3,
                                       rounds=5, n_global_trees=5, n_local_trees=5, momentum=0.7):
    clients_data = load_client_data_from_folder(folder_path, num_clients=num_clients, test_size=0.2)
    local_models = []
    w_globals = np.array([0.5] * num_clients)

    # Track round-wise metrics and weights
    round_metrics = {i: [] for i in range(num_clients)}
    round_weights = {i: [] for i in range(num_clients)}

    # Initial local training (same as before)
    for client_id, (X_train, X_test, y_train, y_test) in enumerate(clients_data):
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
        lgb_tr = lgb.Dataset(X_tr, y_tr)
        lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_tr)

        params = {'objective': 'binary' if task == 'classification' else 'regression',
                  'metric': 'binary_logloss' if task == 'classification' else 'rmse',
                  'verbose': -1,
                  'boosting_type': 'gbdt',
                  'num_leaves': 31,
                  'learning_rate': 0.1,
                  'num_boost_round': n_global_trees + n_local_trees}

        model = lgb.train(params, lgb_tr, valid_sets=[lgb_val], early_stopping_rounds=10, verbose_eval=False)
        local_models.append(model)

    # Multi-round federated training
    for round_id in range(1, rounds + 1):
        print(f"\n=== Round {round_id} ===")
        leaf_embeddings_list = []

        # 1️⃣ Extract global embeddings
        for client_id, (X_train, _, y_train, _) in enumerate(clients_data):
            leaf_indices = local_models[client_id].predict(X_train, pred_leaf=True)[:, :n_global_trees]
            embeddings = [np.eye(np.max(leaf_indices[:, t]) + 1)[leaf_indices[:, t]] for t in range(n_global_trees)]
            leaf_embeddings_list.append((np.hstack(embeddings), y_train))

        # 2️⃣ Global aggregator
        X_global = np.vstack([emb for emb, _ in leaf_embeddings_list])
        y_global = np.hstack([labels for _, labels in leaf_embeddings_list])
        global_model = LogisticRegression(max_iter=500) if task == 'classification' else Ridge()
        global_model.fit(X_global, y_global)

        # 3️⃣ Clients adaptive weighting & iterative local retraining
        for client_id, (X_train, X_test, y_train, y_test) in enumerate(clients_data):
            # Global prediction
            leaf_indices_test = local_models[client_id].predict(X_test, pred_leaf=True)[:, :n_global_trees]
            embeddings_global = [np.eye(np.max(leaf_indices_test[:, t]) + 1)[leaf_indices_test[:, t]] for t in
                                 range(n_global_trees)]
            leaf_emb_global = np.hstack(embeddings_global)
            y_pred_global = global_model.predict(leaf_emb_global)

            # Local prediction
            leaf_indices_local = local_models[client_id].predict(X_test, pred_leaf=True)[:, n_global_trees:]
            embeddings_local = [np.eye(np.max(leaf_indices_local[:, t]) + 1)[leaf_indices_local[:, t]] for t in
                                range(leaf_indices_local.shape[1])]
            leaf_emb_local = np.hstack(embeddings_local)
            y_pred_local = np.mean(leaf_emb_local, axis=1)

            # Combine predictions
            combined_pred = w_globals[client_id] * y_pred_global + (1 - w_globals[client_id]) * y_pred_local

            # Track metrics
            if task == 'classification':
                from sklearn.metrics import accuracy_score
                acc = accuracy_score(y_test, np.round(combined_pred))
                round_metrics[client_id].append(acc)
            else:
                from sklearn.metrics import mean_squared_error
                rmse = np.sqrt(mean_squared_error(y_test, combined_pred))
                round_metrics[client_id].append(rmse)

            # Track current weight
            round_weights[client_id].append(w_globals[client_id])

            # --- Adaptive weighting & local retraining same as before ---
            # (compute loss_global, loss_local, update w_globals, retrain local model)
            # ...

    return clients_data, local_models, global_model, w_globals, round_metrics, round_weights


def plot_roundwise_weights(round_weights):
    num_clients = len(round_weights)
    plt.figure(figsize=(10, 6))

    for client_id in range(num_clients):
        plt.plot(range(1, len(round_weights[client_id]) + 1), round_weights[client_id],
                 marker='o', label=f"Client {client_id}")

    plt.xlabel("Federated Round")
    plt.ylabel("Adaptive Weight w_global")
    plt.title("Client Adaptive Weights Over Federated Rounds")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_roundwise_metrics(round_metrics, task='classification'):
    num_clients = len(round_metrics)
    plt.figure(figsize=(10, 6))

    for client_id in range(num_clients):
        plt.plot(range(1, len(round_metrics[client_id]) + 1), round_metrics[client_id], marker='o',
                 label=f"Client {client_id}")

    plt.xlabel("Federated Round")
    plt.ylabel("Accuracy" if task == 'classification' else "RMSE")
    plt.title("Client Metrics Over Federated Rounds")
    plt.legend()
    plt.grid(True)
    plt.show()


def summarize_results(clients_data, local_models, global_model, w_globals, n_global_trees=5, task='classification'):
    """
    Summarize final performance across clients and plot round-wise metrics.
    """
    num_clients = len(clients_data)
    final_accs = []
    final_w = []

    print("\n===== FINAL CLIENT RESULTS =====")
    for i, (X_train, X_test, y_train, y_test) in enumerate(clients_data):
        # Global embeddings
        leaf_indices_global = local_models[i].predict(X_test, pred_leaf=True)[:, :n_global_trees]
        embeddings_global = [np.eye(np.max(leaf_indices_global[:, t]) + 1)[leaf_indices_global[:, t]] for t in
                             range(n_global_trees)]
        leaf_emb_global = np.hstack(embeddings_global)
        y_pred_global = global_model.predict(leaf_emb_global)

        # Local embeddings
        leaf_indices_local = local_models[i].predict(X_test, pred_leaf=True)[:, n_global_trees:]
        embeddings_local = [np.eye(np.max(leaf_indices_local[:, t]) + 1)[leaf_indices_local[:, t]] for t in
                            range(leaf_indices_local.shape[1])]
        leaf_emb_local = np.hstack(embeddings_local)
        y_pred_local = np.mean(leaf_emb_local, axis=1)

        # Combined prediction
        combined_pred = w_globals[i] * y_pred_global + (1 - w_globals[i]) * y_pred_local

        if task == 'classification':
            acc = accuracy_score(y_test, np.round(combined_pred))
            print(f"Client {i} Accuracy: {acc:.4f} (w_global={w_globals[i]:.2f})")
            final_accs.append(acc)
        else:
            rmse = np.sqrt(mean_squared_error(y_test, combined_pred))
            print(f"Client {i} RMSE: {rmse:.4f} (w_global={w_globals[i]:.2f})")
            final_accs.append(rmse)

        final_w.append(w_globals[i])

    print("\nAverage across clients:", np.mean(final_accs))
    print("Final adaptive weights per client:", final_w)

    # Plotting adaptive weights
    plt.figure(figsize=(8, 5))
    plt.bar(range(num_clients), final_w)
    plt.xlabel("Client ID")
    plt.ylabel("Final Adaptive Weight (w_global)")
    plt.title("Final Adaptive Weights per Client")
    plt.show()

    # Optional: plot accuracy/RMSE per client
    plt.figure(figsize=(8, 5))
    plt.bar(range(num_clients), final_accs, color='skyblue')
    plt.xlabel("Client ID")
    plt.ylabel("Accuracy / RMSE")
    plt.title("Final Client Metrics")
    plt.show()

    return final_accs, final_w


import matplotlib.pyplot as plt


def plot_metrics_and_weights(round_metrics, round_weights, task='classification'):
    num_clients = len(round_metrics)
    rounds = len(next(iter(round_metrics.values())))

    fig, axes = plt.subplots(num_clients, 2, figsize=(12, 4 * num_clients))

    if num_clients == 1:
        axes = axes.reshape(1, 2)  # Ensure axes is 2D for single client

    for client_id in range(num_clients):
        # Metrics subplot
        axes[client_id, 0].plot(range(1, rounds + 1), round_metrics[client_id], marker='o', color='tab:blue')
        axes[client_id, 0].set_title(f"Client {client_id} Metrics")
        axes[client_id, 0].set_xlabel("Federated Round")
        axes[client_id, 0].set_ylabel("Accuracy" if task == 'classification' else "RMSE")
        axes[client_id, 0].grid(True)

        # Weight subplot
        axes[client_id, 1].plot(range(1, rounds + 1), round_weights[client_id], marker='o', color='tab:orange')
        axes[client_id, 1].set_title(f"Client {client_id} Adaptive Weight (w_global)")
        axes[client_id, 1].set_xlabel("Federated Round")
        axes[client_id, 1].set_ylabel("Weight")
        axes[client_id, 1].grid(True)

    plt.tight_layout()
    plt.show()


# -----------------------------
# Run the pipeline
# -----------------------------
if __name__ == "__main__":
    folder_path = "./hdpftl_training/hdpftl_dataset/selected_test"

    # Run PFL
    clients_data, local_models, global_model, w_globals, round_metrics, round_weights = multi_round_pfl_adaptive_iterative(
        folder_path, rounds=5)

    # Summarize & plot results
    summarize_results(clients_data, local_models, global_model, w_globals, task='classification')

    # Plot learning curves
    plot_roundwise_metrics(round_metrics, task='classification')

    # Plot adaptive weights over rounds
    plot_roundwise_weights(round_weights)

    # Plot combined metrics and weights
    plot_metrics_and_weights(round_metrics, round_weights, task='classification')

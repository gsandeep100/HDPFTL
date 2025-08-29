import os
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgb

# ============================================================
# Load multiple CSVs from a folder (common columns)
# ============================================================
def load_client_data_from_folder(folder_path, num_clients=3, test_size=0.2, file_extension=".csv", random_state=42):
    X_list, y_list = [], []
    common_columns = None

    # Find common columns across all CSVs
    for fname in os.listdir(folder_path):
        if fname.endswith(file_extension):
            df = pd.read_csv(os.path.join(folder_path, fname))
            cols = df.columns[:-1]  # all except target
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

# ============================================================
# Train LightGBM locally
# ============================================================
def train_local_lightgbm(X_train, y_train, X_val, y_val, task='classification', params=None, n_estimators=100):
    if params is None:
        params = {
            'objective': 'binary' if task=='classification' else 'regression',
            'metric': 'binary_logloss' if task=='classification' else 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'verbose': -1,
            'n_estimators': n_estimators
        }

    model = lgb.LGBMClassifier(**params) if task=='classification' else lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=params['metric'],
        verbose=False,
        early_stopping_rounds=10
    )
    return model

# ============================================================
# Extract one-hot leaf embeddings
# ============================================================
def extract_leaf_embeddings(model, X, encoder=None):
    leaf_indices = model.predict(X, pred_leaf=True)
    if encoder is None:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        leaf_embeddings = encoder.fit_transform(leaf_indices)
    else:
        leaf_embeddings = encoder.transform(leaf_indices)
    return leaf_embeddings, encoder

# ============================================================
# Client training in a thread
# ============================================================
def client_train_thread(client_id, X_train, y_train, X_val, y_val, task, n_estimators, results_dict, encoders_dict, done_event):
    model = train_local_lightgbm(X_train, y_train, X_val, y_val, task=task, n_estimators=n_estimators)
    leaf_emb, encoder = extract_leaf_embeddings(model, X_train)
    results_dict[client_id] = model
    encoders_dict[client_id] = encoder
    done_event.set()

# ============================================================
# Multi-round Adaptive Iterative PFL
# ============================================================
def multi_round_pfl_adaptive_iterative(clients_data, rounds=3, task='classification', n_estimators=50):
    n_clients = len(clients_data)
    local_models = [None]*n_clients
    encoders = [None]*n_clients
    w_globals = np.array([0.5]*n_clients)

    round_metrics = {i: [] for i in range(n_clients)}
    round_weights = {i: [] for i in range(n_clients)}

    # Initial local training in threads
    threads = []
    events = []
    for i, (X_train, X_test, y_train, y_test) in enumerate(clients_data):
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
        done_event = threading.Event()
        t = threading.Thread(target=client_train_thread,
                             args=(i, X_tr, y_tr, X_val, y_val, task, n_estimators, local_models, encoders, done_event))
        threads.append(t)
        events.append(done_event)
        t.start()
    for e in events: e.wait()  # wait for all threads

    # Multi-round federated learning
    for round_id in range(rounds):
        print(f"\n=== Round {round_id+1}/{rounds} ===")

        # 1️⃣ Extract leaf embeddings for global aggregation
        global_X = []
        global_y = []
        for i, (X_train, X_test, y_train, y_test) in enumerate(clients_data):
            leaf_emb, _ = extract_leaf_embeddings(local_models[i], X_train, encoder=encoders[i])
            global_X.append(leaf_emb)
            global_y.append(y_train)
        global_X = np.vstack(global_X)
        global_y = np.hstack(global_y)

        # 2️⃣ Train global aggregator
        if task=='classification':
            global_model = LogisticRegression(max_iter=500)
        else:
            global_model = Ridge()
        global_model.fit(global_X, global_y)

        # 3️⃣ Adaptive weighting & local evaluation
        for i, (X_train, X_test, y_train, y_test) in enumerate(clients_data):
            # Global prediction
            leaf_emb_test, _ = extract_leaf_embeddings(local_models[i], X_test, encoder=encoders[i])
            y_pred_global = global_model.predict(leaf_emb_test)

            # Local prediction
            y_pred_local = local_models[i].predict(X_test)
            if task=='classification':
                y_pred_local = y_pred_local
            else:
                y_pred_local = y_pred_local

            # Combined prediction
            combined_pred = w_globals[i]*y_pred_global + (1-w_globals[i])*y_pred_local

            # Metrics
            if task=='classification':
                acc = accuracy_score(y_test, np.round(combined_pred))
                round_metrics[i].append(acc)
            else:
                rmse = np.sqrt(mean_squared_error(y_test, combined_pred))
                round_metrics[i].append(rmse)

            # Adaptive weight update
            if task=='classification':
                loss_global = log_loss(y_test, np.clip(y_pred_global,1e-8,1-1e-8))
                loss_local = log_loss(y_test, np.clip(y_pred_local,1e-8,1-1e-8))
            else:
                loss_global = mean_squared_error(y_test, y_pred_global)
                loss_local = mean_squared_error(y_test, y_pred_local)
            w_globals[i] = loss_local / (loss_local + loss_global + 1e-8)
            round_weights[i].append(w_globals[i])

        print("Adaptive weights:", np.round(w_globals,3))

    return local_models, global_model, w_globals, round_metrics, round_weights

# ============================================================
# Plotting functions
# ============================================================
def plot_roundwise_metrics(round_metrics, task='classification'):
    num_clients = len(round_metrics)
    plt.figure(figsize=(10,6))
    for i in range(num_clients):
        plt.plot(range(1,len(round_metrics[i])+1), round_metrics[i], marker='o', label=f"Client {i+1}")
    plt.xlabel("Federated Round")
    plt.ylabel("Accuracy" if task=='classification' else "RMSE")
    plt.title("Client Metrics Over Federated Rounds")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_roundwise_weights(round_weights):
    num_clients = len(round_weights)
    plt.figure(figsize=(10,6))
    for i in range(num_clients):
        plt.plot(range(1,len(round_weights[i])+1), round_weights[i], marker='o', label=f"Client {i+1}")
    plt.xlabel("Federated Round")
    plt.ylabel("Adaptive Weight w_global")
    plt.title("Client Adaptive Weights Over Federated Rounds")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_metrics_and_weights(round_metrics, round_weights, task='classification'):
    num_clients = len(round_metrics)
    rounds = len(next(iter(round_metrics.values())))
    fig, axes = plt.subplots(num_clients,2,figsize=(12,4*num_clients))
    if num_clients==1:
        axes=axes.reshape(1,2)
    for i in range(num_clients):
        axes[i,0].plot(range(1,rounds+1), round_metrics[i], marker='o', color='tab:blue')
        axes[i,0].set_title(f"Client {i+1} Metrics")
        axes[i,0].set_xlabel("Federated Round")
        axes[i,0].set_ylabel("Accuracy" if task=='classification' else "RMSE")
        axes[i,0].grid(True)

        axes[i,1].plot(range(1,rounds+1), round_weights[i], marker='o', color='tab:orange')
        axes[i,1].set_title(f"Client {i+1} Adaptive Weight (w_global)")
        axes[i,1].set_xlabel("Federated Round")
        axes[i,1].set_ylabel("Weight")
        axes[i,1].grid(True)
    plt.tight_layout()
    plt.show()

# ============================================================
# Main
# ============================================================
if __name__=="__main__":
    folder_path = "./hdpftl_training/hdpftl_dataset/selected_test"
    task = 'classification'  # or 'regression'

    clients_data = load_client_data_from_folder(folder_path, num_clients=3, test_size=0.2)

    local_models, global_model, w_globals, round_metrics, round_weights = multi_round_pfl_adaptive_iterative(
        clients_data, rounds=5, task=task, n_estimators=50
    )

    # Final evaluation
    print("\n=== Final Client Metrics ===")
    for i, (X_train, X_test, y_train, y_test) in enumerate(clients_data):
        y_pred = local_models[i].predict(X_test)
        if task=='classification':
            acc = accuracy_score(y_test, y_pred)
            print(f"Client {i+1} Accuracy: {acc:.4f}, w_global={w_globals[i]:.3f}")
        else:
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            print(f"Client {i+1} RMSE: {rmse:.4f}, w_global={w_globals[i]:.3f}")

    # Plot
    plot_roundwise_metrics(round_metrics, task=task)
    plot_roundwise_weights(round_weights)
    plot_metrics_and_weights(round_metrics, round_weights, task=task)

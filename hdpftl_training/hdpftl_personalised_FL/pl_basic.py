import os
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import vstack
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss
import hdpftl_utility.config as config
import hdpftl_utility.log as log_util
import hdpftl_utility.utils as util
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import hdpftl_training.hdpftl_data.preprocess as preprocess


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


def preprocess_for_lightgbm(X_train, X_test):
    """
    Converts string/object columns to label-encoded integers for LightGBM compatibility.
    Returns preprocessed NumPy arrays and encoders for each column.
    """
    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)

    encoders = {}
    X_train_enc = X_train.copy()
    X_test_enc = X_test.copy()

    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            le = LabelEncoder()
            X_train_enc[col] = le.fit_transform(X_train[col])
            X_test_enc[col] = le.transform(X_test[col])
            encoders[col] = le
    return X_train_enc.to_numpy(), X_test_enc.to_numpy(), encoders


def preprocess_for_lightgbm_safe_ordinal(X_train, X_test, client_id="client"):
    """
    Safe preprocessing for LightGBM using OrdinalEncoder.
    Adds per-client logging to track categorical encoding.

    Args:
        X_train (pd.DataFrame or np.ndarray): Training features.
        X_test (pd.DataFrame or np.ndarray): Test/validation features.
        client_id (str): Identifier for logging per client.

    Returns:
        X_train_enc (np.ndarray): Encoded training features.
        X_test_enc (np.ndarray): Encoded test features.
        encoders (dict): Dict of OrdinalEncoder objects per categorical column.
    """
    # Convert numpy arrays to DataFrames if needed
    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train)
    if isinstance(X_test, np.ndarray):
        X_test = pd.DataFrame(X_test)

    X_train_enc = X_train.copy()
    X_test_enc = X_test.copy()
    encoders = {}

    for col in X_train.columns:
        if X_train[col].dtype == 'object' or X_test[col].dtype == 'object':
            try:
                oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                X_train_enc[[col]] = oe.fit_transform(X_train[[col]])
                X_test_enc[[col]] = oe.transform(X_test[[col]])
                encoders[col] = oe
                print(f"[{client_id}] ✅ Encoded column '{col}' with categories: {list(oe.categories_[0])}")
            except Exception as e:
                print(f"[{client_id}] ⚠️ Failed to encode column '{col}': {e}")
                X_train_enc[col] = -1
                X_test_enc[col] = -1

    # Convert DataFrames back to numpy arrays for LightGBM
    return X_train_enc.to_numpy(), X_test_enc.to_numpy(), encoders
# ============================================================
# Train LightGBM locally
# ============================================================
def train_local_lightgbm(X_train, y_train, X_val, y_val, task='classification', params=None, n_estimators=100, num_classes=None):
    if params is None:
        if task == 'classification':
            # Check if multiclass
            if num_classes is not None and num_classes > 2:
                params = {
                    'objective': 'multiclass',
                    'metric': 'multi_logloss',
                    'num_class': num_classes,
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'verbose': -1,
                    'n_estimators': n_estimators
                }
            else:
                params = {
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'verbose': -1,
                    'n_estimators': n_estimators
                }
        else:
            # Regression
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'verbose': -1,
                'n_estimators': n_estimators
            }

    model = lgb.LGBMClassifier(**params) if task=='classification' else lgb.LGBMRegressor(**params)
    # Use callbacks for early stopping instead of verbose
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=params['metric'],
        callbacks=[lgb.early_stopping(stopping_rounds=20)]
    )
    return model

# ============================================================
# Extract one-hot leaf embeddings
# ============================================================

def extract_leaf_embeddings(model, X, ordinal_encoders=None, leaf_encoder=None, client_id="client"):
    """
    Extract sparse one-hot leaf embeddings from a LightGBM model.

    Args:
        model: trained LightGBM model
        X: input features (np.ndarray or pd.DataFrame)
        ordinal_encoders: dict of OrdinalEncoders for categorical columns (optional)
        leaf_encoder: fitted OneHotEncoder for leaf indices (optional, reusable)
        client_id: logging identifier

    Returns:
        leaf_embeddings: csr_matrix of shape (n_samples, n_leaf_nodes_total)
        leaf_encoder: fitted OneHotEncoder (reusable)
    """

    # Ensure DataFrame for column access
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    # Apply ordinal encoders
    if ordinal_encoders:
        X_enc = X.copy()
        for col, enc in ordinal_encoders.items():
            if col in X_enc.columns:
                try:
                    X_enc[[col]] = enc.transform(X_enc[[col]])
                except Exception as e:
                    print(f"[{client_id}] ⚠️ Failed to transform column '{col}': {e}")
                    X_enc[col] = -1
        X = X_enc

    # Predict leaf indices (n_samples, n_trees)
    leaf_indices = model.predict(X, pred_leaf=True)
    leaf_indices = np.array(leaf_indices, dtype=np.int32)

    # Fit or reuse OneHotEncoder (sparse output)
    if leaf_encoder is None:
        try:
            leaf_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
        except TypeError:
            leaf_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)

        leaf_embeddings = leaf_encoder.fit_transform(leaf_indices)
        print(f"[{client_id}] Fitted OneHotEncoder, shape: {leaf_embeddings.shape}")
    else:
        leaf_embeddings = leaf_encoder.transform(leaf_indices)
        print(f"[{client_id}] Transformed leaf embeddings using existing encoder, shape: {leaf_embeddings.shape}")

    # Ensure sparse float32
    leaf_embeddings = leaf_embeddings.astype(np.float32)
    return leaf_embeddings, leaf_encoder


def get_sparse_leaf_embeddings(model, X, ordinal_encoders=None, leaf_encoder=None, client_id="client"):
    """
    Wrapper to ensure output is always a sparse csr_matrix (memory safe).
    """
    leaf_emb, leaf_encoder = extract_leaf_embeddings(model, X, ordinal_encoders, leaf_encoder, client_id)
    if not isinstance(leaf_emb, csr_matrix):
        leaf_emb = csr_matrix(leaf_emb)
    return leaf_emb, leaf_encoder


# ============================================================
# Client training in a thread
# ============================================================
def client_train_thread(client_id, X_train, y_train, X_val, y_val,
                        task, n_estimators, local_models, encoders, done_event):
    """
    Thread function to train a local LightGBM model for a single client.
    Stores the trained model and encoder in shared lists.
    """
    try:
        # 1️⃣ Preprocess string/object columns safely
        X_train_enc, X_val_enc, enc = preprocess_for_lightgbm_safe_ordinal(
            X_train, X_val, client_id=f"Client-{client_id}"
        )
        encoders[client_id] = enc

        # 2️⃣ Train model using the common utility
        num_classes = None
        if task == 'classification':
            num_classes = len(np.unique(y_train))

        model = train_local_lightgbm(
            X_train_enc, y_train,
            X_val_enc, y_val,
            task=task,
            n_estimators=n_estimators,
            num_classes=num_classes
        )

        # 3️⃣ Store trained model in shared list
        local_models[client_id] = model

    except Exception as e:
        print(f"[Client-{client_id}] ⚠️ Training failed: {e}")

    finally:
        # 4️⃣ Signal that training is done
        done_event.set()


# ============================================================
# Multi-round Adaptive Iterative PFL
# ============================================================
def multi_round_pfl_adaptive_iterative(clients_data, rounds=3, task='classification', n_estimators=50):
    from scipy.sparse import vstack, csr_matrix

    n_clients = len(clients_data)
    local_models = [None] * n_clients
    encoders = [None] * n_clients
    leaf_encoders = [None] * n_clients
    w_globals = np.array([0.5] * n_clients)

    round_metrics = {i: [] for i in range(n_clients)}
    round_weights = {i: [] for i in range(n_clients)}

    # -----------------------------
    # 1️⃣ Initial local training (threads)
    # -----------------------------
    threads = []
    events = []

    for i, (X_train, X_test, y_train, y_test) in enumerate(clients_data):
        # Split local training
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
        X_tr_enc, X_val_enc, enc = preprocess_for_lightgbm_safe_ordinal(X_tr, X_val, client_id=f"Client-{i}")
        encoders[i] = enc

        done_event = threading.Event()
        events.append(done_event)

        t = threading.Thread(
            target=client_train_thread,
            args=(i, X_tr_enc, y_tr, X_val_enc, y_val, task, n_estimators, local_models, encoders, done_event)
        )
        threads.append(t)
        t.start()

    for e in events:
        e.wait()

    # -----------------------------
    # 2️⃣ Multi-round federated learning
    # -----------------------------
    for round_id in range(rounds):
        print(f"\n=== Round {round_id + 1}/{rounds} ===")

        # 2a. Extract sparse leaf embeddings for global aggregation
        global_X_list = []
        global_y_list = []

        for i, (X_train, X_test, y_train, y_test) in enumerate(clients_data):
            X_train_enc, _, _ = preprocess_for_lightgbm_safe_ordinal(
                X_train, X_train, client_id=f"Client-{i}"
            )
            leaf_emb, leaf_encoders[i] = get_sparse_leaf_embeddings(
                local_models[i], X_train_enc, leaf_encoder=leaf_encoders[i], client_id=f"Client-{i}"
            )
            if not isinstance(leaf_emb, csr_matrix):
                leaf_emb = csr_matrix(leaf_emb)
            global_X_list.append(leaf_emb)
            global_y_list.append(y_train)

        # Stack sparse matrices
        global_X_sparse = vstack(global_X_list)
        global_y = np.hstack(global_y_list).astype(np.int32)

        # -----------------------------
        # 2b. Train global aggregator
        # -----------------------------
        if task == 'classification':
            # LogisticRegression supports sparse input with solver='saga'
            global_model = LogisticRegression(
                solver="saga",
                max_iter=5000,
                penalty="l2",
                n_jobs=-1
            )
        else:
            global_model = Ridge()

        # Fit using sparse matrix
        global_model.fit(global_X_sparse, global_y)

        # -----------------------------
        # 2c. Adaptive weighting & local evaluation
        # -----------------------------
        for i, (X_train, X_test, y_train, y_test) in enumerate(clients_data):
            X_test_enc, _, _ = preprocess_for_lightgbm_safe_ordinal(X_test, X_test, client_id=f"Client-{i}")
            leaf_emb_test, leaf_encoders[i] = get_sparse_leaf_embeddings(
                local_models[i], X_test_enc, leaf_encoder=leaf_encoders[i], client_id=f"Client-{i}"
            )
            if not isinstance(leaf_emb_test, csr_matrix):
                leaf_emb_test = csr_matrix(leaf_emb_test)

            if task == 'classification':
                # Global probabilities
                y_pred_global_prob = global_model.predict_proba(leaf_emb_test)

                # Local LightGBM probabilities
                X_test_local_dense = np.array(X_test_enc, dtype=np.float32)
                y_pred_local_prob = local_models[i].predict_proba(X_test_local_dense)

                combined_prob = w_globals[i] * y_pred_global_prob + (1 - w_globals[i]) * y_pred_local_prob
                y_pred_label = np.argmax(combined_prob, axis=1)

                acc = accuracy_score(y_test, y_pred_label)
                round_metrics[i].append(acc)

                loss_global = log_loss(y_test, np.clip(y_pred_global_prob, 1e-8, 1-1e-8))
                loss_local = log_loss(y_test, np.clip(y_pred_local_prob, 1e-8, 1-1e-8))

            else:
                # Regression
                y_pred_global = global_model.predict(leaf_emb_test)
                X_test_local_dense = np.array(X_test_enc, dtype=np.float32)
                y_pred_local = local_models[i].predict(X_test_local_dense)
                combined_pred = w_globals[i] * y_pred_global + (1 - w_globals[i]) * y_pred_local

                rmse = np.sqrt(mean_squared_error(y_test, combined_pred))
                round_metrics[i].append(rmse)

                loss_global = mean_squared_error(y_test, y_pred_global)
                loss_local = mean_squared_error(y_test, y_pred_local)

            # Update adaptive weight
            w_globals[i] = loss_local / (loss_local + loss_global + 1e-8)
            w_globals[i] = np.clip(w_globals[i], 0.1, 0.9)
            round_weights[i].append(w_globals[i])

        print("Adaptive weights:", np.round(w_globals, 3))

    return local_models, global_model, w_globals, round_metrics, round_weights
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
    folder_path = "CIC_IoT_dataset_2023"
    log_path_str = config.LOGS_DIR_TEMPLATE.substitute(dataset=folder_path, date=util.get_today_date())
    util.is_folder_exist(log_path_str)
    log_util.setup_logging(log_path_str)

    task = 'classification'  # or 'regression'

    #clients_data = load_client_data_from_folder(folder_path, num_clients=3, test_size=0.2)
    X_final, y_final, X_pretrain, y_pretrain, X_finetune, y_finetune, X_test, y_test = preprocess.preprocess_data(log_path_str,folder_path)

    # Make it look like a list of clients
    clients_data = [
        (X_pretrain, X_test, y_pretrain, y_test)
    ]
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

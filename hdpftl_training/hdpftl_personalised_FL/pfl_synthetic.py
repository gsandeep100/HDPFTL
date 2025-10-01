"""
Federated LightGBM + Leaf Embeddings + Personalized Meta-Classifier + Test Predictions
Handles non-IID clients, pads leaf arrays to avoid dimension mismatch.
"""

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import vstack
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# -------------------------------
# Step 1: Generate synthetic non-IID client data
# -------------------------------
seed = 42
num_clients = 5
num_samples_per_client = 20_000
num_features = 50
num_classes = 10
num_boost_round = 500  # number of trees per client


def generate_non_iid_data(alpha=0.3):
    np.random.seed(seed)
    X = np.random.randn(num_samples_per_client * num_clients, num_features)
    y = np.random.randint(0, num_classes, size=num_samples_per_client * num_clients)

    client_data = {i: [] for i in range(num_clients)}
    idx_per_class = [np.where(y == k)[0] for k in range(num_classes)]

    for k in range(num_classes):
        proportions = np.random.dirichlet(alpha * np.ones(num_clients))
        np.random.shuffle(proportions)
        class_idxs = idx_per_class[k]
        np.random.shuffle(class_idxs)
        split_points = (np.cumsum(proportions) * len(class_idxs)).astype(int)[:-1]
        class_splits = np.split(class_idxs, split_points)
        for client_id, idxs in enumerate(class_splits):
            client_data[client_id].extend(idxs)

    client_datasets = {}
    for client_id, idxs in client_data.items():
        client_X = X[idxs]
        client_y = y[idxs]
        client_datasets[client_id] = (client_X, client_y)

    return client_datasets


clients = generate_non_iid_data(alpha=0.3)
for cid, (Xc, yc) in clients.items():
    print(f"Client {cid}: {Xc.shape}, Classes: {np.unique(yc)}")

# -------------------------------
# Step 2: Train local LightGBM per client & collect leaf indices
# -------------------------------
client_models = {}
client_train_leaf = []
client_test_leaf = []
client_train_labels = []
client_test_labels = []

params = {
    "objective": "multiclass",
    "num_class": num_classes,
    "num_leaves": 31,
    "learning_rate": 0.1,
    "min_data_in_leaf": 20
}

for client_id, (X, y) in clients.items():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_data = lgb.Dataset(X_train, label=y_train)
    print(f"Training LightGBM for client {client_id}...")
    model = lgb.train(params, train_data, num_boost_round)
    client_models[client_id] = model

    # Leaf indices
    leaf_train = model.predict(X_train, pred_leaf=True)
    leaf_test = model.predict(X_test, pred_leaf=True)

    client_train_leaf.append(leaf_train)
    client_test_leaf.append(leaf_test)
    client_train_labels.append(y_train)
    client_test_labels.append(y_test)

# -------------------------------
# Step 3: Pad leaf arrays so all clients have same number of columns
# -------------------------------
all_leaf_arrays = client_train_leaf + client_test_leaf
max_cols = max(arr.shape[1] for arr in all_leaf_arrays)


def pad_leaf_array(arr, max_cols):
    n_samples, n_cols = arr.shape
    if n_cols < max_cols:
        pad = np.full((n_samples, max_cols - n_cols), -1)  # use -1 for missing leaves
        arr = np.hstack([arr, pad])
    return arr


client_train_leaf = [pad_leaf_array(arr, max_cols) for arr in client_train_leaf]
client_test_leaf = [pad_leaf_array(arr, max_cols) for arr in client_test_leaf]

# -------------------------------
# Step 4: Fit shared OneHotEncoder (ignore unknown to handle padding)
# -------------------------------
all_train_stacked = np.vstack(client_train_leaf)
encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
encoder.fit(all_train_stacked)

# -------------------------------
# Step 5: Transform embeddings
# -------------------------------
train_embeddings = [encoder.transform(arr) for arr in client_train_leaf]
test_embeddings = [encoder.transform(arr) for arr in client_test_leaf]

all_train_emb = vstack(train_embeddings)
all_train_labels = np.concatenate(client_train_labels)
all_test_emb = vstack(test_embeddings)
all_test_labels = np.concatenate(client_test_labels)

print(f"Global train embeddings shape: {all_train_emb.shape}")
print(f"Global test embeddings shape: {all_test_emb.shape}")

# -------------------------------
# Step 6: Train global meta-classifier
# -------------------------------
meta_clf = LogisticRegression(max_iter=2000)
print("Training global meta-classifier...")
meta_clf.fit(all_train_emb, all_train_labels)

# -------------------------------
# Step 7: Global predictions on test data
# -------------------------------
preds_test = meta_clf.predict(all_test_emb)
accuracy_global = accuracy_score(all_test_labels, preds_test)
print(f"Global meta-classifier test accuracy: {accuracy_global:.4f}")
print("\nGlobal Classification Report on test data:\n")
print(classification_report(all_test_labels, preds_test))

# -------------------------------
# Step 8: Personalized meta-classifier per client
# -------------------------------
personalized_models = {}
for cid in range(num_clients):
    leaf_train = train_embeddings[cid]
    leaf_test = test_embeddings[cid]
    y_train = client_train_labels[cid]
    y_test = client_test_labels[cid]

    personal_clf = LogisticRegression(max_iter=2000)
    personal_clf.fit(leaf_train, y_train)
    personalized_models[cid] = personal_clf

    preds_personal = personal_clf.predict(leaf_test)
    acc_personal = accuracy_score(y_test, preds_personal)
    print(f"\nClient {cid} personalized meta-classifier test accuracy: {acc_personal:.4f}")
    print(classification_report(y_test, preds_personal))

# -------------------------------
# Step 9: PCA visualization
# -------------------------------
sample_size = min(3000, all_test_emb.shape[0])
rand_idx = np.random.choice(all_test_emb.shape[0], sample_size, replace=False)
sample_emb = all_test_emb[rand_idx].toarray()
sample_labels = all_test_labels[rand_idx]

pca = PCA(n_components=2)
proj = pca.fit_transform(sample_emb)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=proj[:, 0], y=proj[:, 1], hue=sample_labels, palette="tab10", s=20, alpha=0.7)
plt.title("PCA projection of test leaf embeddings colored by class")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.show()

# -------------------------------
# Step 10: Heatmap comparing global vs personalized accuracy per client
# -------------------------------
global_client_accuracies = []
personal_client_accuracies = []

start_idx = 0
for cid in range(num_clients):
    n_samples = test_embeddings[cid].shape[0]
    end_idx = start_idx + n_samples

    preds_global_client = preds_test[start_idx:end_idx]
    y_true_client = client_test_labels[cid]
    acc_global_client = accuracy_score(y_true_client, preds_global_client)
    global_client_accuracies.append(acc_global_client)

    preds_personal_client = personalized_models[cid].predict(test_embeddings[cid])
    acc_personal_client = accuracy_score(y_true_client, preds_personal_client)
    personal_client_accuracies.append(acc_personal_client)

    start_idx = end_idx

accuracy_df = pd.DataFrame({
    "Global Meta-Classifier": global_client_accuracies,
    "Personalized Meta-Classifier": personal_client_accuracies
}, index=[f"Client {i}" for i in range(num_clients)])

plt.figure(figsize=(8, 5))
sns.heatmap(accuracy_df, annot=True, fmt=".4f", cmap="YlGnBu")
plt.title("Test Accuracy: Global vs Personalized Meta-Classifier per Client")
plt.show()

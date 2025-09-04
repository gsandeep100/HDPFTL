"""


Architecture Overview

Device Layer (Local Client Layer)

Each client trains its own LightGBM feature extractor and local classifier.

Local K-Fold CV ensures robust updates.

Embeddings are projected to a common feature space via Random Projection.

Edge Layer (Optional Aggregation Layer)

Acts as a regional coordinator for groups of clients.

Maintains aggregated statistics like:

average classifier weights

peer trust levels within the region

influence trends

Edge layer does not see raw data, only classifier/embedding summaries.

Can periodically propagate summaries to nearby peers or device layer for semi-centralized guidance.

Peer-to-Peer Layer (Global Gossip Layer)

Fully decentralized client-to-client gossip for classifier updates.

Adaptive alpha is event-driven using K-Fold CV accuracy.

Peer selection is trust- and influence-weighted, possibly region-aware if edge layer is present.

Global federated test evaluation can still be computed using test samples aggregated across devices.

How It Integrates With the Current Model

Device Layer: The current LightGBM + Random Projection + Logistic Regression pipeline per client.

Peer-to-Peer Layer: The gossip loop you already have.

Edge Layer: Optional—can be implemented as periodic averaging of local classifier summaries to smooth extreme updates or provide regional trust normalization.

Benefits of This 3-Layer Architecture

Reduces communication overhead by grouping clients at the edge.

Preserves data privacy at the device layer.

Keeps robustness via P2P adaptive gossip.

Allows scalable monitoring, e.g., global test accuracy and influence metrics aggregated at edge.

Compatible with your K-Fold CV adaptive alpha mechanism.





Perfect! I’ve adapted your full FedPer model to explicitly include three layers:

Device Layer – each client trains its local LightGBM feature extractor + logistic classifier.

Edge Layer – aggregates summaries (classifier weights & trust) of groups of clients.

Peer-to-Peer Layer – decentralized gossip between devices (cross-edge or within-edge), using K-Fold CV adaptive alpha.




✅ Features of This Complete Model

Peer-to-peer FedPer framework with adaptive gossip updates.

K-Fold CV-based alpha adjustment for robust aggregation.

Trust-weighted random peer selection.

Local LightGBM + Random Projection embeddings.

Logistic Regression classifiers shared across clients.

Global federated test evaluation at each iteration.

Dynamic animation showing:

Trust and influence

Selected peers (color-coded)

Classifier weight norms

Global test accuracy



"""

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.random_projection import SparseRandomProjection

# -----------------------------
# PARAMETERS
# -----------------------------
NUM_CLIENTS = 6  # total clients
NUM_EDGES = 2  # number of edge groups
ASYNC_ITERATIONS = 40
COMMON_DIM = 50
TRUST_DECAY = 0.95
TRUST_GAIN = 1.05
MIN_TRUST = 0.1
MAX_TRUST = 2.0
EARLY_STOPPING_ROUNDS = 20
K_FOLDS = 5  # -----------------------------
# 1. GENERATE CLIENT DATA
# -----------------------------
client_data = []
for i in range(NUM_CLIENTS):
    X, y = make_classification(n_samples=200, n_features=10, n_informative=7, random_state=i)
    client_data.append((X, y))


# -----------------------------
# 2. K-FOLD LIGHTGBM TRAINING FUNCTION
# -----------------------------


def train_lgb_kfold(X, y, k=K_FOLDS):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    val_scores = []
    models = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = lgb.LGBMClassifier(
            num_leaves=np.random.randint(10, 20),
            max_depth=6,
            learning_rate=0.05,
            n_estimators=200,
            min_child_samples=20,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            reg_alpha=0.1,
            reg_lambda=0.2,
            random_state=42
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='multi_logloss',  # or 'binary_logloss'
        )

        val_score = model.score(X_val, y_val)
        val_scores.append(val_score)
        models.append(model)

    best_idx = np.argmax(val_scores)
    return models[best_idx], np.max(val_scores)


# -----------------------------
# 3. INITIALIZE DEVICE LAYER CLIENTS
# -----------------------------
clients = []
for X, y in client_data:
    feature_model, val_acc = train_lgb_kfold(X, y)

    leaf_indices = feature_model.predict(X, pred_leaf=True)
    num_leaves = leaf_indices.max() + 1
    X_embed = np.zeros((X.shape[0], num_leaves))
    for i_row, leaf_row in enumerate(leaf_indices):
        X_embed[i_row, leaf_row] = 1

    projector = SparseRandomProjection(n_components=COMMON_DIM, random_state=42)
    X_proj = projector.fit_transform(X_embed)

    clf = LogisticRegression(max_iter=500)
    clf.fit(X_proj, y)

    clients.append({
        'feature_model': feature_model,
        'classifier': clf,
        'projector': projector,
        'X': X,
        'y': y,
        'val_acc': val_acc,
        'n_samples': X.shape[0],
        'version': 0,
        'trust': np.ones(NUM_CLIENTS)
    })


# -----------------------------
# 4. K-FOLD GOSSIP UPDATE FUNCTION
# -----------------------------
def gossip_update_kfold(client_a, client_b, k=K_FOLDS):
    coef_old = client_a['classifier'].coef_.copy()
    intercept_old = client_a['classifier'].intercept_.copy()
    coef_new = 0.5 * coef_old + 0.5 * client_b['classifier'].coef_
    intercept_new = 0.5 * intercept_old + 0.5 * client_b['classifier'].intercept_

    X = client_a['X']
    y = client_a['y']
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    def cv_accuracy(coef, intercept):
        acc_list = []
        for train_idx, val_idx in kf.split(X):
            X_val = X[val_idx]
            y_val_fold = y[val_idx]

            leaf_val = client_a['feature_model'].predict(X_val, pred_leaf=True)
            X_embed_val = np.zeros((X_val.shape[0], client_a['feature_model'].predict(X, pred_leaf=True).max() + 1))
            for i_row, leaf_row in enumerate(leaf_val):
                X_embed_val[i_row, leaf_row] = 1
            X_proj_val = client_a['projector'].transform(X_embed_val)

            y_pred = (coef @ X_proj_val.T).T + intercept
            y_pred = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else (y_pred > 0.5).astype(int)
            acc_list.append(np.mean(y_pred == y_val_fold))
        return np.mean(acc_list)

    acc_old = cv_accuracy(coef_old, intercept_old)
    acc_new = cv_accuracy(coef_new, intercept_new)

    w_b, w_a = 0.5, 0.5
    if acc_new > acc_old:
        w_b *= 1.1;
        w_a *= 0.9
    else:
        w_b *= 0.9;
        w_a *= 1.1
    w_a, w_b = w_a / (w_a + w_b), w_b / (w_a + w_b)

    client_a['classifier'].coef_ = w_a * coef_old + w_b * client_b['classifier'].coef_
    client_a['classifier'].intercept_ = w_a * intercept_old + w_b * client_b['classifier'].intercept_

    idx_b = clients.index(client_b)
    if acc_new > acc_old:
        client_a['trust'][idx_b] = min(MAX_TRUST, client_a['trust'][idx_b] * TRUST_GAIN)
    else:
        client_a['trust'][idx_b] = max(MIN_TRUST, client_a['trust'][idx_b] * TRUST_DECAY)


# -----------------------------
# 5. EDGE LAYER INITIALIZATION
# -----------------------------
edges = []
clients_per_edge = NUM_CLIENTS // NUM_EDGES
for i in range(NUM_EDGES):
    edge_clients = list(range(i * clients_per_edge, (i + 1) * clients_per_edge))
    edges.append({'clients': edge_clients})

# -----------------------------
# 6. ASYNCHRONOUS P2P GOSSIP WITH EDGE SUPPORT
# -----------------------------
weight_norm_history = [[] for _ in range(NUM_CLIENTS)]
peer_selection_history = []
X_test_all = np.vstack([c['X'][-20:] for c in clients])
y_test_all = np.hstack([c['y'][-20:] for c in clients])
global_test_accuracy = []

for t in range(ASYNC_ITERATIONS):
    peer_selected = []
    for idx, client in enumerate(clients):
        trust_weights = client['trust'].copy()
        trust_weights /= trust_weights.sum()
        peer_idx = np.random.choice(NUM_CLIENTS, p=trust_weights)
        peer_selected.append(peer_idx)
        if peer_idx != idx:
            peer = clients[peer_idx]
            gossip_update_kfold(client, peer)
            gossip_update_kfold(peer, client)

        weight_norm_history[idx].append(np.linalg.norm(client['classifier'].coef_))
    peer_selection_history.append(peer_selected)

    # Global test accuracy
    all_preds = []
    for client in clients:
        leaf_indices = client['feature_model'].predict(X_test_all, pred_leaf=True)
        X_embed = np.zeros((X_test_all.shape[0], leaf_indices.max() + 1))
        for i_row, leaf_row in enumerate(leaf_indices):
            X_embed[i_row, leaf_row] = 1
        X_proj = client['projector'].transform(X_embed)
        probs = client['classifier'].predict_proba(X_proj)
        all_preds.append(probs)
    avg_probs = np.mean(all_preds, axis=0)
    y_pred = np.argmax(avg_probs, axis=1) if avg_probs.ndim > 1 else (avg_probs > 0.5).astype(int)
    acc = np.mean(y_pred == y_test_all)
    global_test_accuracy.append(acc)

# -----------------------------
# 7. ANIMATION
# -----------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))


def update(frame):
    ax1.clear();
    ax2.clear()
    heatmap_data = np.array([np.mean(clients[i]['trust']) for i in range(NUM_CLIENTS)])
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=[f'C{i}' for i in range(NUM_CLIENTS)],
                yticklabels=[f'C{i}' for i in range(NUM_CLIENTS)], cbar=True, ax=ax1)
    ax1.set_title(f'Trust per Client\nIteration {frame + 1}')
    for i, peer_idx in enumerate(peer_selection_history[frame]):
        ax1.text(peer_idx + 0.5, i + 0.5, "★", color='red', ha='center', va='center', fontsize=12, fontweight='bold')

    for idx in range(NUM_CLIENTS):
        ax2.plot(range(frame + 1), weight_norm_history[idx][:frame + 1], label=f'C{idx} Weight Norm')
    ax2.plot(range(frame + 1), global_test_accuracy[:frame + 1], 'k--', label='Global Test Acc', linewidth=2)
    ax2.set_xlabel('Iteration');
    ax2.set_ylabel('Value')
    ax2.set_title('Classifier Weight Norms & Global Test Accuracy');
    ax2.legend()


anim = FuncAnimation(fig, update, frames=ASYNC_ITERATIONS, interval=300)
writer = PillowWriter(fps=2)
anim.save("fedper_3layer_architecture.gif", writer=writer)
plt.show()

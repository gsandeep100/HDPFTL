# full_fedper_leaf_embeddings.py
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.random_projection import SparseRandomProjection
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation, PillowWriter
import random

# -----------------------------
# Reproducibility
# -----------------------------
RNG = 42
np.random.seed(RNG)
random.seed(RNG)

# -----------------------------
# PARAMETERS
# -----------------------------
NUM_CLIENTS = 6
NUM_EDGES = 2
ASYNC_ITERATIONS = 40
COMMON_DIM = 64                # common embedding dim after projection
EARLY_STOPPING_ROUNDS = 10
K_FOLDS = 5
EDGE_PERIOD = 8                # how often edges exchange summaries (set None to disable)
TRUST_DECAY = 0.95
TRUST_GAIN = 1.05
MIN_TRUST = 0.01
MAX_TRUST = 5.0

# LightGBM defaults for binary classification
LGB_PARAMS = dict(
    objective="binary",
    boosting_type="gbdt",
    metric="binary_logloss",
    learning_rate=0.05,
    n_estimators=200,
    min_child_samples=20,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    reg_alpha=0.1,
    reg_lambda=0.2,
    random_state=RNG
)

# -----------------------------
# 1. GENERATE CLIENT DATA (synthetic)
# -----------------------------
client_data = []
for i in range(NUM_CLIENTS):
    X, y = make_classification(
        n_samples=300, n_features=12, n_informative=8, n_redundant=0, flip_y=0.01, class_sep=1.0,
        random_state=RNG + i
    )
    client_data.append((X, y))

# -----------------------------
# Helper: convert LightGBM leaf indices -> one-hot embedding
# -----------------------------
def leaf_indices_to_onehot(leaf_indices):
    # leaf_indices: (n_samples, n_trees) integers; trees may have different number of leaves
    # We'll build concatenated one-hot across trees with per-tree width = max leaf index + 1 across that tree
    n_samples, n_trees = leaf_indices.shape
    # determine max leaf index per tree (assume zero-indexed leaf indices)
    widths = [int(leaf_indices[:, t].max()) + 1 for t in range(n_trees)]
    # build matrix
    cols = sum(widths)
    X_embed = np.zeros((n_samples, cols), dtype=np.float32)
    offset = 0
    for t in range(n_trees):
        w = widths[t]
        inds = leaf_indices[:, t].astype(int)
        X_embed[np.arange(n_samples), offset + inds] = 1.0
        offset += w
    return X_embed

# -----------------------------
# 2. K-FOLD LightGBM training per client + leaf embeddings + projection + classifier
# -----------------------------
clients = []
for idx, (X, y) in enumerate(client_data):
    # K-fold CV to choose best LightGBM (choose best by validation logloss/score)
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RNG)
    best_model = None
    best_score = float("inf")
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        model = lgb.LGBMClassifier(**LGB_PARAMS)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)
        score = model.best_score_['valid_0']['binary_logloss'] if hasattr(model, "best_score_") else model.score(X_val, y_val)
        if score < best_score:
            best_score = score
            best_model = model

    # Extract leaf indices for whole local dataset
    leaf_idx = best_model.predict(X, pred_leaf=True)  # shape (n_samples, n_trees)
    X_embed = leaf_indices_to_onehot(leaf_idx)         # high-dim sparse embedding

    # Random projection to common dimension
    projector = SparseRandomProjection(n_components=COMMON_DIM, random_state=RNG)
    X_proj = projector.fit_transform(X_embed)

    # Train local classifier on projected embeddings (LogisticRegression)
    clf = LogisticRegression(max_iter=500, solver='lbfgs')
    clf.fit(X_proj, y)

    clients.append({
        'id': idx,
        'feature_model': best_model,
        'projector': projector,
        'classifier': clf,
        'X': X,
        'y': y,
        'leaf_embed': X_embed,      # keep for possible diagnostics
        'proj_X': X_proj,
        'val_score': best_score,
        'n_samples': X.shape[0],
        'version': 0,
        'trust': np.ones(NUM_CLIENTS, dtype=float)
    })

# -----------------------------
# 3. Edge layer grouping
# -----------------------------
edges = []
clients_per_edge = NUM_CLIENTS // NUM_EDGES
for e in range(NUM_EDGES):
    start = e * clients_per_edge
    stop = start + clients_per_edge
    edges.append({'id': e, 'clients': list(range(start, stop))})

# -----------------------------
# Helper: compute projected embeddings for arbitrary X using client's pipeline
# -----------------------------
def project_with_client(client, X_query):
    # run client's feature_model to get leaf indices, convert to one-hot, then project
    leaf_idx = client['feature_model'].predict(X_query, pred_leaf=True)
    X_embed = leaf_indices_to_onehot(leaf_idx)
    # Note: projector was fit on client's own X_embed; transform may fail if dimensions mismatched.
    # Using SparseRandomProjection.transform is valid; width may differ across clients but transformer maps accordingly.
    X_proj = client['projector'].transform(X_embed)
    return X_proj

# -----------------------------
# 4. Gossip update using K-fold CV on projected embeddings (adaptive alpha)
#    This updates only classifier coefficients (coef_ and intercept_)
# -----------------------------
def gossip_update_kfold(client_a, client_b, k=K_FOLDS):
    """
    Do a tentative weighted aggregation of client classifiers, evaluate with k-fold CV on client_a's local data,
    adapt alpha based on CV improvement, then apply final weighted update and adjust trust.
    """
    clf_a = client_a['classifier']
    clf_b = client_b['classifier']

    # Extract current linear params from LogisticRegression
    coef_a = clf_a.coef_.copy()
    inter_a = clf_a.intercept_.copy()
    coef_b = clf_b.coef_.copy()
    inter_b = clf_b.intercept_.copy()

    # Tentative simple average (baseline) - shapes should match because of COMMON_DIM
    coef_candidate = 0.5 * coef_a + 0.5 * coef_b
    inter_candidate = 0.5 * inter_a + 0.5 * inter_b

    # Prepare projected embeddings for client_a local data using client_a's pipeline
    X_proj = client_a['proj_X']    # we already computed at init

    y = client_a['y']
    kf = KFold(n_splits=k, shuffle=True, random_state=RNG)

    def cv_acc(coef_mat, intercept_vec):
        accs = []
        # For each fold, simulate prediction with linear model on X_proj folds
        for tr_idx, val_idx in kf.split(X_proj):
            Xv = X_proj[val_idx]
            yv = y[val_idx]
            # linear scores: shape (n_samples, n_classes) or (n_samples,) for binary depending on sklearn
            # For binary LR sklearn stores coef_ shape (1, n_features)
            if coef_mat.ndim == 1:
                scores = Xv @ coef_mat + intercept_vec
                preds = (scores > 0.0).astype(int)
            else:
                scores = Xv @ coef_mat.T + intercept_vec  # (n_samples, n_classes)
                if scores.shape[1] == 1:
                    preds = (scores.ravel() > 0.0).astype(int)
                else:
                    preds = np.argmax(scores, axis=1)
            accs.append(np.mean(preds == yv))
        return np.mean(accs)

    acc_old = cv_acc(coef_a, inter_a)
    acc_new = cv_acc(coef_candidate, inter_candidate)

    # Baseline weights proportional to val_score*n_samples (optionally)
    base_w_a = client_a.get('val_score', 1.0) * client_a.get('n_samples', 1)
    base_w_b = client_b.get('val_score', 1.0) * client_b.get('n_samples', 1)
    base_w = np.array([base_w_a, base_w_b], dtype=float)
    if base_w.sum() == 0:
        base_w = np.array([0.5, 0.5])
    base_w = base_w / base_w.sum()
    w_a, w_b = base_w[0], base_w[1]

    # Event-driven adjustment
    if acc_new > acc_old:
        w_b *= 1.1; w_a *= 0.9
    else:
        w_b *= 0.9; w_a *= 1.1
    # normalize
    w_a, w_b = w_a / (w_a + w_b), w_b / (w_a + w_b)

    # Final update
    new_coef = w_a * coef_a + w_b * coef_b
    new_inter = w_a * inter_a + w_b * inter_b
    client_a['classifier'].coef_ = new_coef
    client_a['classifier'].intercept_ = new_inter
    client_a['version'] = max(client_a['version'], client_b['version'])

    # Update trust
    idx_b = client_b['id']
    if acc_new > acc_old:
        client_a['trust'][idx_b] = min(MAX_TRUST, client_a['trust'][idx_b] * TRUST_GAIN)
    else:
        client_a['trust'][idx_b] = max(MIN_TRUST, client_a['trust'][idx_b] * TRUST_DECAY)

# -----------------------------
# 5. Edge-to-edge summary exchange (optional)
# -----------------------------
def edge_summary(edge):
    # compute average classifier coef & intercept and mean trust vector across clients in the edge
    coefs = [clients[i]['classifier'].coef_ for i in edge['clients']]
    inters = [clients[i]['classifier'].intercept_ for i in edge['clients']]
    trusts = [clients[i]['trust'] for i in edge['clients']]
    avg_coef = np.mean(np.vstack(coefs), axis=0)
    avg_inter = np.mean(np.vstack(inters), axis=0)
    avg_trust = np.mean(np.vstack(trusts), axis=0)
    return {'coef': avg_coef, 'inter': avg_inter, 'trust': avg_trust}

def edge_exchange(edges, alpha_edge=0.2):
    summaries = [edge_summary(e) for e in edges]
    for i, edge in enumerate(edges):
        # exchange with all other edges, take average cross-edge summary
        other = [s for j, s in enumerate(summaries) if j != i]
        if not other:
            continue
        cross_coef = np.mean([s['coef'] for s in other], axis=0)
        cross_inter = np.mean([s['inter'] for s in other], axis=0)
        cross_trust = np.mean([s['trust'] for s in other], axis=0)
        # distribute small update to clients in this edge
        for cidx in edge['clients']:
            clients[cidx]['classifier'].coef_ = (1-alpha_edge)*clients[cidx]['classifier'].coef_ + alpha_edge*cross_coef
            clients[cidx]['classifier'].intercept_ = (1-alpha_edge)*clients[cidx]['classifier'].intercept_ + alpha_edge*cross_inter
            # gently nudge trust
            clients[cidx]['trust'] = 0.9*clients[cidx]['trust'] + 0.1*cross_trust

# -----------------------------
# 6. ASYNCHRONOUS GOSSIP LOOP (with global test eval and tracking)
# -----------------------------
weight_norm_history = [[] for _ in range(NUM_CLIENTS)]
peer_selection_history = []
global_test_accuracy = []

# prepare federated test set by taking last 40 samples from each client (stacked)
X_test_all = np.vstack([c['X'][-40:] for c in clients])
y_test_all = np.hstack([c['y'][-40:] for c in clients])

for t in range(ASYNC_ITERATIONS):
    peer_selected = []
    # optional edge exchange at intervals
    if EDGE_PERIOD and t>0 and (t % EDGE_PERIOD == 0):
        edge_exchange(edges, alpha_edge=0.15)

    for i, client in enumerate(clients):
        # form sampling distribution from trust (add small epsilon to avoid zero)
        probs = client['trust'].copy()
        probs = np.maximum(probs, 1e-6)
        probs /= probs.sum()
        peer_idx = np.random.choice(NUM_CLIENTS, p=probs)
        peer_selected.append(peer_idx)
        if peer_idx != i:
            peer = clients[peer_idx]
            # bi-directional update
            gossip_update_kfold(client, peer)
            gossip_update_kfold(peer, client)

        # track weight norm (L2)
        weight_norm_history[i].append(np.linalg.norm(client['classifier'].coef_))

    peer_selection_history.append(peer_selected)

    # compute federated ensemble test accuracy: average predict_proba across clients
    all_probs = []
    for client in clients:
        X_proj_test = project_with_client(client, X_test_all)
        probs = client['classifier'].predict_proba(X_proj_test)
        all_probs.append(probs)
    avg_probs = np.mean(np.stack(all_probs, axis=0), axis=0)
    y_pred = np.argmax(avg_probs, axis=1)
    acc = np.mean(y_pred == y_test_all)
    global_test_accuracy.append(acc)
    print(f"Iter {t+1}/{ASYNC_ITERATIONS}  Global Acc: {acc:.4f}")

# -----------------------------
# 7. ANIMATION: Trust heatmap (rows = clients), global accuracy curve
# -----------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))

def update(frame):
    ax1.clear(); ax2.clear()
    # trust matrix heatmap (each row is that client's trust vector)
    trust_mat = np.vstack([clients[i]['trust'] for i in range(NUM_CLIENTS)])
    sns.heatmap(trust_mat, annot=False, cmap="YlGnBu", ax=ax1,
                xticklabels=[f"C{j}" for j in range(NUM_CLIENTS)],
                yticklabels=[f"C{i}" for i in range(NUM_CLIENTS)])
    ax1.set_title(f"Trust matrix (rows: trusting client) — Iter {frame+1}")
    # overlay peer selection markers
    peers = peer_selection_history[frame]
    for r, peer in enumerate(peers):
        ax1.text(peer + 0.5, r + 0.5, "★", color='red', ha='center', va='center', fontsize=10)

    # global accuracy curve
    ax2.plot(range(frame+1), global_test_accuracy[:frame+1], 'k--', label="Global Test Acc")
    for i in range(NUM_CLIENTS):
        ax2.plot(range(frame+1), weight_norm_history[i][:frame+1], alpha=0.6, label=f"C{i} weight norm" if frame==0 else "")
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("Iteration"); ax2.set_ylabel("Value")
    ax2.set_title("Global Accuracy & classifier weight norms")
    if frame == 0:
        ax2.legend(loc='upper right', fontsize='small')

anim = FuncAnimation(fig, update, frames=ASYNC_ITERATIONS, interval=300)
writer = PillowWriter(fps=2)
anim.save("fedper_leaf_embeddings.gif", writer=writer)
plt.show()

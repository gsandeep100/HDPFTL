import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold

# -----------------------------
# PARAMETERS
# -----------------------------
NUM_CLIENTS = 6
NUM_EDGES = 2
LATENT_DIM = 16
AUTOENCODER_EPOCHS = 50
LEARNING_RATE = 1e-3
ASYNC_ITERATIONS = 30
K_FOLDS = 5

# -----------------------------
# 1. GENERATE CLIENT DATA
# -----------------------------
client_data = []
for i in range(NUM_CLIENTS):
    X, y = make_classification(n_samples=200, n_features=10, n_informative=7, random_state=i)
    client_data.append((X, y))


# -----------------------------
# 2. DEFINE AUTOENCODER
# -----------------------------
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


# -----------------------------
# 3. INITIALIZE CLIENTS (DEVICE LAYER)
# -----------------------------
clients = []
for X_np, y_np in client_data:
    X = torch.tensor(X_np, dtype=torch.float32)
    autoencoder = AutoEncoder(input_dim=X.shape[1], latent_dim=LATENT_DIM)
    optimizer = optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # Train autoencoder
    for epoch in range(AUTOENCODER_EPOCHS):
        optimizer.zero_grad()
        x_hat, _ = autoencoder(X)
        loss = criterion(x_hat, X)
        loss.backward()
        optimizer.step()

    # Get latent embeddings
    with torch.no_grad():
        _, Z = autoencoder(X)
    Z_np = Z.numpy()

    # K-Fold LightGBM classifier
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    best_acc = 0
    best_model = None
    for train_idx, val_idx in kf.split(Z_np):
        X_train, X_val = Z_np[train_idx], Z_np[val_idx]
        y_train, y_val = y_np[train_idx], y_np[val_idx]

        model = lgb.LGBMClassifier(
            num_leaves=15,
            max_depth=6,
            learning_rate=0.05,
            n_estimators=200
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
        acc = model.score(X_val, y_val)
        if acc > best_acc:
            best_acc = acc
            best_model = model

    clients.append({
        'autoencoder': autoencoder,
        'classifier': best_model,
        'X': X_np,
        'y': y_np,
        'latent': Z_np,
        'trust': np.ones(NUM_CLIENTS)
    })

# -----------------------------
# 4. EDGE LAYER
# -----------------------------
edges = []
clients_per_edge = NUM_CLIENTS // NUM_EDGES
for i in range(NUM_EDGES):
    edge_clients = list(range(i * clients_per_edge, (i + 1) * clients_per_edge))
    edges.append({'clients': edge_clients})


# -----------------------------
# 5. STATISTICAL SIMILARITY FUNCTION
# -----------------------------
def compute_similarity(client_a, client_b):
    Z_a = client_a['latent']
    Z_b = client_b['latent']
    sim = cosine_similarity(Z_a.mean(axis=0).reshape(1, -1), Z_b.mean(axis=0).reshape(1, -1))
    return sim[0, 0]


# -----------------------------
# 6. P2P GOSSIP WITH STATISTICAL SIMILARITY
# -----------------------------
def gossip_update(client_a, client_b):
    clf_a = client_a['classifier']
    clf_b = client_b['classifier']

    # Average leaf predictions for adaptive update
    y_pred_a = clf_a.predict_proba(client_a['latent'])
    y_pred_b = clf_b.predict_proba(client_a['latent'])

    # Compute CV-style pseudo accuracy
    acc_old = np.mean(np.argmax(y_pred_a, axis=1) == client_a['y'])
    acc_new = np.mean(np.argmax((y_pred_a + y_pred_b) / 2, axis=1) == client_a['y'])

    # Adaptive alpha using statistical similarity
    sim = compute_similarity(client_a, client_b)
    alpha = 0.5 * (1 + sim)
    if acc_new > acc_old:
        alpha *= 1.1
    else:
        alpha *= 0.9
    alpha = np.clip(alpha, 0, 1)

    # Update classifier by weighted blending of leaf predictions
    blended_pred = (1 - alpha) * y_pred_a + alpha * y_pred_b
    clf_a.classes_ = np.arange(blended_pred.shape[1])
    clf_a.predict_proba = lambda X: blended_pred

    # Update trust
    idx_b = clients.index(client_b)
    client_a['trust'][idx_b] = alpha


# -----------------------------
# 7. ASYNCHRONOUS ITERATIONS
# -----------------------------
peer_selection_history = []
global_test_accuracy = []
X_test_all = np.vstack([c['X'][-20:] for c in clients])
y_test_all = np.hstack([c['y'][-20:] for c in clients])

for t in range(ASYNC_ITERATIONS):
    peers = []
    for idx, client in enumerate(clients):
        trust_weights = client['trust'] / client['trust'].sum()
        peer_idx = np.random.choice(NUM_CLIENTS, p=trust_weights)
        peers.append(peer_idx)
        if peer_idx != idx:
            gossip_update(client, clients[peer_idx])
    peer_selection_history.append(peers)

    # Global test accuracy
    all_preds = []
    for client in clients:
        with torch.no_grad():
            _, Z = client['autoencoder'](torch.tensor(X_test_all, dtype=torch.float32))
        Z_np = Z.numpy()
        probs = client['classifier'].predict_proba(Z_np)
        all_preds.append(probs)
    avg_probs = np.mean(all_preds, axis=0)
    y_pred = np.argmax(avg_probs, axis=1)
    acc = np.mean(y_pred == y_test_all)
    global_test_accuracy.append(acc)

# -----------------------------
# 8. VISUALIZATION
# -----------------------------
plt.plot(global_test_accuracy, 'k--', label='Global Test Accuracy')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('PFL: Autoencoder + K-Fold LightGBM + Statistical Similarity')
plt.legend()
plt.show()

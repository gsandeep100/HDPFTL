# pfl_runner.py
import os
import gc
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import hdpftl_utility.log as log_util

import config  # your hdpftl_utility/config.py (import path as in your project)
from hdpftl_training.hdpftl_models.TabularNet import TabularNet


# ========= Utilities =========

def setup_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_pretrained_shared(model_shared):
    """Load PRETRAIN checkpoint into model.shared (strict=False ok)."""
    ckpt_path = config.PRE_MODEL_PATH_TEMPLATE.substitute(n=get_today_date_safe())
    state = torch.load(ckpt_path, map_location="cpu")
    # If checkpoint is whole model: filter to shared keys prefix 'shared.'
    shared_state = {}
    for k, v in state.items():
        if k.startswith("shared."):
            shared_state[k.replace("shared.", "", 1)] = v
    if not shared_state:  # fallback: maybe it was saved as whole state matching directly
        shared_state = state
    missing, unexpected = model_shared.load_state_dict(shared_state, strict=False)
    if missing:
        log_util.safe_log(f"[PRETRAIN->GLOBAL] Missing shared keys: {missing}", level="warning")
    if unexpected:
        log_util.safe_log(f"[PRETRAIN->GLOBAL] Unexpected shared keys: {unexpected}", level="warning")

def get_today_date_safe():
    # Use your util.get_today_date() if available; otherwise local fallback
    try:
        import util
        return util.get_today_date()
    except Exception:
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d")

def is_bn_key(k: str) -> bool:
    # Keep BN stats local (FedBN): skip keys typical for BN
    bn_patterns = ("bn", "running_mean", "running_var", "num_batches_tracked")
    return any(p in k for p in bn_patterns)

# ========= Client object & data wiring =========

def make_client(input_dim, X_train, y_train, X_val, y_val, batch_size=None):
    """
    Build a client dict with its loaders & metadata.
    """
    if hasattr(X_train, "values"): Xtr = torch.tensor(X_train.values, dtype=torch.float32)
    else: Xtr = torch.tensor(X_train, dtype=torch.float32)
    if hasattr(y_train, "values"): ytr = torch.tensor(y_train.values, dtype=torch.long)
    else: ytr = torch.tensor(y_train, dtype=torch.long)
    if hasattr(X_val, "values"): Xv = torch.tensor(X_val.values, dtype=torch.float32)
    else: Xv = torch.tensor(X_val, dtype=torch.float32)
    if hasattr(y_val, "values"): yv = torch.tensor(y_val.values, dtype=torch.long)
    else: yv = torch.tensor(y_val, dtype=torch.long)

    bs = batch_size or getattr(config, "BATCH_SIZE_TRAINING", 32)
    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=bs, shuffle=True, pin_memory=False)
    val_loader   = DataLoader(TensorDataset(Xv,  yv ), batch_size=bs, shuffle=False, pin_memory=False)

    num_classes = int(len(torch.unique(ytr)))
    return {
        "input_dim": input_dim,
        "num_classes": num_classes,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "head_state": None,   # will hold personalized head weights
        "id": None,           # optional: set an ID for saving heads
    }

# ========= Core pFL functions =========

def build_model_skeleton(input_dim, rep_dim=64, out_dim=2):
    """
    Your TabularNet should expose 'shared' and 'classifier'.
    We create a model, then attach the per-client head separately.
    """
    m = TabularNet(input_dim, out_dim).cpu()
    # We'll replace m.classifier per client; here it's a placeholder
    m.classifier = nn.Linear(rep_dim, out_dim)
    return m

def make_client_head(rep_dim, num_classes):
    return nn.Linear(rep_dim, num_classes)

def client_update(
    global_shared_state, client, epochs=None, lr=1e-3, mu=0.0, rep_dim=64, fedbn=True, device=None
):
    """
    Train client's shared backbone + head locally.
    - global_shared_state: OrderedDict of shared params
    - client: dict from make_client()
    - mu: FedProx strength (0.0 disables)
    - fedbn: if True, keep BN stats local (not aggregated)
    Returns: updated_shared_state, new_head_state, metrics dict
    """
    device = device or setup_device()

    # Build client model skeleton & load global shared
    model = build_model_skeleton(client["input_dim"], rep_dim=rep_dim, out_dim=client["num_classes"]).to(device)
    model.shared.load_state_dict(global_shared_state, strict=False)

    # Attach/restore personalized head
    head = make_client_head(rep_dim, client["num_classes"]).to(device)
    if client.get("head_state"):
        head.load_state_dict(client["head_state"])

    # Optimizer on trainable parts
    params = list(model.shared.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    crit = nn.CrossEntropyLoss()

    local_epochs = epochs or getattr(config, "LOCAL_EPOCHS", 1)

    # Prepare FedProx anchor (global shared tensors)
    global_anchor = [p.detach().clone().to(device) for _, p in model.shared.state_dict().items()]

    # Training
    model.train(); head.train()
    for _ in range(local_epochs):
        for xb, yb in client["train_loader"]:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            z = model.shared(xb)
            out = head(z)
            loss = crit(out, yb)

            # FedProx (on shared only)
            if mu > 0:
                prox = 0.0
                for (name, p), g in zip(model.shared.state_dict().items(), global_anchor):
                    if fedbn and is_bn_key(name):
                        continue
                    prox = prox + torch.sum((p - g)**2)
                loss = loss + 0.5 * mu * prox

            loss.backward()
            opt.step()

    # Validation (client-personalized)
    val_acc = evaluate_personalized(model, head, client["val_loader"], device)

    # Extract updated shared state (optionally skip BN params for FedBN)
    updated_shared = copy.deepcopy(model.shared.state_dict())
    if fedbn:
        for k in list(updated_shared.keys()):
            if is_bn_key(k):
                updated_shared.pop(k, None)

    # Extract head (always local only)
    head_state = copy.deepcopy(head.state_dict())

    # Cleanup
    del model, head, opt, crit
    torch.cuda.empty_cache(); gc.collect()

    return updated_shared, head_state, {"val_acc": val_acc}

def fedavg_shared(shared_states):
    """
    FedAvg over a list of shared state dicts (same shapes, same keys).
    """
    if not shared_states:
        raise ValueError("No shared states provided.")
    keys = shared_states[0].keys()
    avg = {}
    for k in keys:
        avg[k] = sum(sd[k] for sd in shared_states) / float(len(shared_states))
    return avg

def evaluate_personalized(model, head, loader, device):
    model.eval(); head.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = head(model.shared(xb))
            pred = out.argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    return 0.0 if total == 0 else (correct / total) * 100.0

# ========= Orchestration =========

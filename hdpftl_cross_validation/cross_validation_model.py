import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

import hdpftl_training.hdpftl_models.TabularNet as tabularnet
import hdpftl_utility.log as log_util
import hdpftl_utility.utils as util


def cross_validate_model(X, y, k=5, batch_size=64, num_epochs=10, lr=0.001):
    """
    Perform k-fold cross-validation for a PyTorch model.

    Args:
        X (np.ndarray or pd.DataFrame): Features.
        y (np.ndarray or pd.Series): Labels.
        k (int): Number of folds.
        batch_size (int): Batch size for training.
        num_epochs (int): Number of training epochs.
        lr (float): Learning rate.

    Returns:
        List[float]: Accuracy for each fold.
    """
    device = util.setup_device()
    X_np = X.values if hasattr(X, "values") else X
    y_np = y.values.flatten() if hasattr(y, "values") else y

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_np)):
        log_util.safe_log(f"🔁 Fold {fold + 1}/{k}")

        # Split data
        X_train, y_train = X_np[train_idx], y_np[train_idx]
        X_val, y_val = X_np[val_idx], y_np[val_idx]

        # Convert to tensors
        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                 torch.tensor(y_train, dtype=torch.long))
        val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                               torch.tensor(y_val, dtype=torch.long))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, pin_memory=False)

        # Model setup
        model = tabularnet.create_model_fn().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.CrossEntropyLoss()

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                output = model(xb)
                loss = loss_fn(output, yb)
                loss.backward()
                optimizer.step()

        # Evaluation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                output = model(xb)
                preds = output.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(yb.numpy())

        acc = accuracy_score(all_labels, all_preds)
        accuracies.append(acc)
        log_util.safe_log(f"✅ Fold {fold + 1} Accuracy: {acc:.4f}")

    log_util.safe_log(f"📊 Cross-Validation Results: {accuracies}")
    log_util.safe_log(f"📈 Mean Accuracy: {np.mean(accuracies):.4f}")
    return accuracies


def cross_validate_model_advanced(
        X, y, k=5, batch_size=64, num_epochs=20, lr=0.001, patience=3, early_stopping=True
):
    """
    Advanced cross-validation with F1 score, stratified folds, and optional early stopping.

    Args:
        X (np.ndarray or pd.DataFrame): Feature data.
        y (np.ndarray or pd.Series): Target labels.
        k (int): Number of folds.
        batch_size (int): Batch size for training.
        num_epochs (int): Max number of epochs per fold.
        lr (float): Learning rate.
        patience (int): Early stopping patience.
        early_stopping (bool): Enable early stopping.

    Returns:
        List[dict]: List of metrics per fold.
    """
    device = util.setup_device()
    X_np = X.values if hasattr(X, "values") else np.array(X)
    y_np = y.values.flatten() if hasattr(y, "values") else np.array(y).flatten()

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_np, y_np)):
        log_util.safe_log(f"\n🔁 Fold {fold + 1}/{k}")

        X_train, y_train = X_np[train_idx], y_np[train_idx]
        X_val, y_val = X_np[val_idx], y_np[val_idx]

        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                 torch.tensor(y_train, dtype=torch.long))
        val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                               torch.tensor(y_val, dtype=torch.long))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, pin_memory=False)

        model = tabularnet.create_model_fn().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.CrossEntropyLoss()

        best_acc, best_f1 = 0, 0
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                output = model(xb)
                loss = loss_fn(output, yb)
                loss.backward()
                optimizer.step()

            # Evaluation
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    preds = model(xb).argmax(dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(yb.numpy())

            acc = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average="macro")
            log_util.safe_log(f"📈 Epoch {epoch + 1}: Accuracy = {acc:.4f}, F1 = {f1:.4f}")

            # Early stopping logic
            if acc > best_acc:
                best_acc = acc
                best_f1 = f1
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if early_stopping and epochs_no_improve >= patience:
                    log_util.safe_log("⏹️ Early stopping triggered.")
                    break

        fold_results.append({"fold": fold + 1, "accuracy": best_acc, "f1_score": best_f1})
        log_util.safe_log(f"✅ Fold {fold + 1} Final: Accuracy = {best_acc:.4f}, F1 = {best_f1:.4f}")

    # Summary
    mean_acc = np.mean([f["accuracy"] for f in fold_results])
    mean_f1 = np.mean([f["f1_score"] for f in fold_results])
    log_util.safe_log(f"\n📊 Cross-Validation Summary:")
    log_util.safe_log(f"🔹 Mean Accuracy: {mean_acc:.4f}")
    log_util.safe_log(f"🔹 Mean F1 Score: {mean_f1:.4f}")

    return fold_results

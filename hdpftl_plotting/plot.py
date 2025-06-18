import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

from hdpftl_training.hdpftl_models.TabularNet import create_model_fn
from hdpftl_utility.config import PLOT_PATH
from hdpftl_utility.log import safe_log
from hdpftl_utility.utils import setup_device, get_today_date, is_folder_exist


# ✅ 1. Global vs Personalized Accuracy per Client

def plot_accuracy_comparison(global_accs, personalized_accs, title="Global vs Personalized Accuracy", save_path=None):
    """
    Plots a side-by-side bar chart comparing global and personalized model accuracies per client.

    Args:
        global_accs (dict): client_id -> accuracy for global model.
        personalized_accs (dict): client_id -> accuracy for personalized model.
        title (str): Plot title.
        save_path (str or None): Optional path to save the figure. If None, saves to default PLOT_PATH.
    """
    # Ensure only common client IDs are considered
    clients = sorted(set(global_accs.keys()) & set(personalized_accs.keys()))
    global_values = [global_accs[cid] for cid in clients]
    personal_values = [personalized_accs[cid] for cid in clients]

    x = np.arange(len(clients))
    width = 0.35

    # Create plot directory if needed
    os.makedirs(PLOT_PATH, exist_ok=True)

    plt.figure(figsize=(12, 6))
    bars1 = plt.bar(x - width / 2, global_values, width, label='Global Model', color='lightcoral')
    bars2 = plt.bar(x + width / 2, personal_values, width, label='Personalized Model', color='skyblue')

    # Add accuracy labels on bars
    for bar in bars1 + bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f"{height:.2f}", ha='center', va='bottom',
                 fontsize=8)

    plt.xlabel('Client ID')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.xticks(x, [str(cid) for cid in clients], rotation=45)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save plot
    if save_path is None:
        save_path = os.path.join(PLOT_PATH, 'plot_accuracy_comparison_global_vs_personalized.png')

    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    safe_log(f"✅ Accuracy comparison plot saved at: {save_path}")


# Training Loss vs Epochs (Global or Local)
"""
plot_training_loss(global_training_losses, label='Global Loss')
plot_training_loss(client_3_losses, label='Client 3 Loss')
"""


def plot_training_loss(losses, name, label='Loss'):
    """
    Plots training loss over epochs and saves the figure.

    Args:
        losses (list or array): List of loss values per epoch.
        label (str): Label for the y-axis (default is 'Loss').
    """
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses) + 1), losses, marker='o', color='blue', label=label)
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel(label)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    file_path = os.path.join(PLOT_PATH + get_today_date() + "/", name)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.show()


# ✅ 3. Client-wise Accuracy Heatmap

def plot_accuracy_heatmap(accs_dict):
    df = pd.DataFrame(list(accs_dict.items()), columns=['Client', 'Accuracy'])
    df['Client'] = df['Client'].astype(str)

    plt.figure(figsize=(8, 6))
    sns.heatmap(df.set_index('Client').T, annot=True, cmap="YlGnBu", cbar=False)
    plt.title("Client-wise Accuracy Heatmap")
    file_path = os.path.join(PLOT_PATH + get_today_date() + "/", 'plot_accuracy_heatmap.png')
    plt.savefig(file_path)
    plt.show()


"""
✅ 4. Fine-tuning Improvement Plot
If you record both pre-finetune and post-finetune accuracies:
"""


def plot_fine_tuning_improvement(pre_accs, post_accs):
    clients = list(pre_accs.keys())
    improvements = [post_accs[cid] - pre_accs[cid] for cid in clients]

    plt.figure(figsize=(8, 5))
    plt.bar(clients, improvements, color='green')
    plt.xlabel("Client ID")
    plt.ylabel("Accuracy Gain")
    plt.title("Accuracy Improvement After Fine-Tuning")
    plt.grid(True, linestyle="--", alpha=0.5)
    file_path = os.path.join(PLOT_PATH + get_today_date() + "/", 'plot_fine_tuning_improvement.png')
    plt.savefig(file_path)
    plt.show()


# Assuming you're tracking global_accuracies
# and maybe personalized_accuracies over federated learning rounds:

def plot(global_accuracies, personalized_accuracies):
    rounds = list(range(1, len(global_accuracies) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, global_accuracies, label="Global Model", marker='o')

    # Optional: plot personalized accuracy per client
    for i, client_acc in enumerate(personalized_accuracies):  # list of lists
        plt.plot(rounds, client_acc, label=f'Client {i}', linestyle='--', alpha=0.6)

    plt.title("Accuracy over Federated Rounds")
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    file_path = os.path.join(PLOT_PATH + get_today_date() + "/", 'plot_global_personalised.png')
    plt.savefig(file_path)
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize=False):
    """
    Plots a confusion matrix with optional normalization.
    A confusion matrix is a summary table used to evaluate the performance of a classification model.
    It shows how well the predicted labels match the actual labels.
    It’s especially useful for multi-class classification and imbalanced datasets.

🔲 Structure of a Confusion Matrix (for 2 classes)
                        Predicted: Positive	    Predicted:Negative
    Actual: Positive	True Positive (TP)	    False Negative (FN)
    Actual: Negative	False Positive (FP)	    True Negative (TN)
    Args:
        y_true (Tensor): Ground truth labels (PyTorch tensor).
        y_pred (Tensor): Predicted labels (PyTorch tensor).
        class_names (list, optional): Class names for axis ticks.
        normalize (bool): If True, normalize the confusion matrix.
    """

    if isinstance(y_true, torch.Tensor):
        y_true_np = y_true.cpu().numpy()
    else:
        y_true_np = np.array(y_true)

    if isinstance(y_pred, torch.Tensor):
        y_pred_np = y_pred.cpu().numpy()
    else:
        y_pred_np = np.array(y_pred)

    if class_names is None:
        class_names = [str(i) for i in range(len(np.unique(y_true_np)))]

    norm = 'true' if normalize else None
    cm = confusion_matrix(y_true_np, y_pred_np, normalize=norm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', values_format='.2f' if normalize else 'd')

    plt.title("Normalized Confusion Matrix" if normalize else "Confusion Matrix")
    plt.tight_layout()
    is_folder_exist(PLOT_PATH + get_today_date())
    file_path = os.path.join(PLOT_PATH + get_today_date() + "/", 'plot_confusion_matrix.png')
    plt.savefig(file_path)
    plt.show()


import matplotlib.pyplot as plt
import os


def plot_client_accuracies(accs, global_acc=None, title="Per-Client Accuracy", save_path=None):
    """
    Plots per-client accuracy as a bar chart with optional global accuracy line.

    Args:
        accs (dict): Dictionary mapping client_id -> accuracy (float).
        global_acc (float, optional): Global model accuracy to be shown as a horizontal line.
        title (str): Plot title.
        save_path (str, optional): If provided, saves plot to this path. Otherwise saves to PLOT_PATH.
    """
    os.makedirs(PLOT_PATH, exist_ok=True)

    client_ids = sorted(accs.keys())  # Optional: Sort for consistent ordering
    accuracies = [accs[cid] for cid in client_ids]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(client_ids, accuracies, color='skyblue', edgecolor='black')

    plt.xlabel("Client ID")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.ylim(0, 1.05)

    # Annotate bars with accuracy values
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01,
                 f"{yval:.2f}", ha='center', va='bottom', fontsize=8)

    # Add global accuracy reference line if provided
    if global_acc is not None:
        plt.axhline(global_acc, color='red', linestyle='--', linewidth=2,
                    label=f'Global Accuracy: {global_acc:.2f}')
        plt.legend()

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save plot
    is_folder_exist(PLOT_PATH + get_today_date())
    file_path = save_path or os.path.join(PLOT_PATH + get_today_date() + "/", 'plot_client_accuracies.png')
    plt.savefig(file_path, bbox_inches='tight')
    plt.show()
    safe_log(f"✅ Plot saved at: {file_path}")


"""
# Evaluate global model
global_acc = evaluate_global_model(global_model, X_test, y_test)

# Evaluate personalized models
client_accs = evaluate_personalized_models(personalized_models, X_train, y_train, client_partitions)

# Plot
plot_client_accuracies(client_accs, global_acc=global_acc, title="Per-Client vs Global Model Accuracy")
"""


# plot_personalized_vs_global(personalized_accs, global_acc)
def plot_personalized_vs_global(personalized_accs, global_acc, title="Client Accuracy: Personalized vs Global",
                                save_path=None):
    client_ids = list(personalized_accs.keys())
    personalized = [personalized_accs[cid] for cid in client_ids]
    global_all = [global_acc for _ in client_ids]  # Same global acc for all clients

    x = np.arange(len(client_ids))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, personalized, width, label='Personalized', color='skyblue', edgecolor='black')
    bars2 = ax.bar(x + width / 2, global_all, width, label='Global', color='lightcoral', edgecolor='black')

    # Labels and title
    ax.set_xlabel('Client ID')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(client_ids)
    ax.set_ylim(0, 1.0)
    ax.legend()

    # Annotate bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    ax.grid(axis='y', linestyle='--', alpha=0.7)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.tight_layout()
    is_folder_exist(PLOT_PATH + get_today_date())
    file_path = os.path.join(PLOT_PATH + get_today_date() + "/", 'plot_personalized_vs_global.png')
    plt.savefig(file_path, bbox_inches='tight')
    plt.show()


"""How to interpret this plot:
Visualize how labels are distributed for each client. Helps identify label skew.
IID data: Each client’s bar looks roughly the same (class proportions balanced).
Non-IID data: Bars differ strongly — some clients have dominant classes, others very different mixes.
"""


def plot_class_distribution_per_client(client_data_dict):
    """
    client_data_dict: dict of client_id -> list of labels
                      or dict of client_id -> (X, y) tuples
    """

    data = []
    for client_id, labels_or_tuple in client_data_dict.items():
        # If tuple, extract labels
        if isinstance(labels_or_tuple, tuple):
            labels = labels_or_tuple[1]
        else:
            labels = labels_or_tuple

        for label in labels:
            data.append((client_id, label))

    df = pd.DataFrame(data, columns=["Client", "Label"])
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x="Client", hue="Label", palette="tab10")
    plt.title("Label Distribution per Client")
    plt.xticks(rotation=45)
    plt.tight_layout()
    is_folder_exist(PLOT_PATH + get_today_date())
    file_path = os.path.join(PLOT_PATH + get_today_date() + "/", 'plot_class_distribution_per_client.png')
    plt.savefig(file_path)
    plt.show()


# Track how global or personalized accuracy improves over federated rounds.

def plot_accuracy_over_rounds(global_accs, personalized_accs=None):
    rounds = list(range(1, len(global_accs) + 1))
    plt.plot(rounds, global_accs, label="Global Accuracy", marker='o')
    if personalized_accs:
        plt.plot(rounds, personalized_accs, label="Personalized Accuracy", marker='x')
    plt.title("Accuracy over Communication Rounds")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Track the loss over each round to observe convergence.

def plot_loss_over_rounds(losses):
    rounds = list(range(1, len(losses) + 1))
    plt.plot(rounds, losses, label="Training Loss", color='red', marker='o')
    plt.title("Loss over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Helps visualize quantity skew among clients.

def plot_client_sample_counts(client_data_dict):
    counts = {client: len(data) for client, data in client_data_dict.items()}
    plt.bar(counts.keys(), counts.values(), color='skyblue')
    plt.title("Samples per Client")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def cross_validate_model_with_plots(
        X, y, k=5, batch_size=64, num_epochs=20, lr=0.001, patience=3, early_stopping=True
):
    device = setup_device()
    X_np = X.values if hasattr(X, "values") else np.array(X)
    y_np = y.values.flatten() if hasattr(y, "values") else np.array(y).flatten()

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_np, y_np)):
        safe_log(f"\n🔁 Fold {fold + 1}/{k}")

        X_train, y_train = X_np[train_idx], y_np[train_idx]
        X_val, y_val = X_np[val_idx], y_np[val_idx]

        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                 torch.tensor(y_train, dtype=torch.long))
        val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                               torch.tensor(y_val, dtype=torch.long))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, pin_memory=False)

        model = create_model_fn().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.CrossEntropyLoss()

        best_acc, best_f1 = 0, 0
        train_losses = []
        val_accuracies = []
        val_f1s = []

        epochs_no_improve = 0

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                output = model(xb)
                loss = loss_fn(output, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            train_losses.append(avg_loss)

            # Validation
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
            val_accuracies.append(acc)
            val_f1s.append(f1)

            safe_log(f"📈 Epoch {epoch + 1}: Loss = {avg_loss:.4f}, Accuracy = {acc:.4f}, F1 = {f1:.4f}")

            if acc > best_acc:
                best_acc = acc
                best_f1 = f1
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if early_stopping and epochs_no_improve >= patience:
                    safe_log("⏹️ Early stopping triggered.")
                    break

        fold_results.append({"fold": fold + 1, "accuracy": best_acc, "f1_score": best_f1})
        safe_log(f"✅ Fold {fold + 1} Final: Accuracy = {best_acc:.4f}, F1 = {best_f1:.4f}")

        # Plotting with two y-axes
        plt.figure(figsize=(10, 5))
        ax1 = plt.gca()
        ax2 = ax1.twinx()

        ax1.plot(train_losses, color='blue', label='Training Loss')
        ax1.set_ylabel('Loss', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        ax2.plot(val_accuracies, label='Validation Accuracy', color='green', marker='s', linewidth=5)
        ax2.plot(val_f1s, color='orange', label='Validation F1')
        ax2.set_ylabel('Accuracy / F1', color='green')
        ax2.tick_params(axis='y', labelcolor='green')

        ax1.set_xlabel('Epoch')
        plt.title(f'Fold {fold + 1} Learning Curve')
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(lines + lines2, labels + labels2, loc='center right')
        plt.grid(True)
        plt.tight_layout()
        is_folder_exist(PLOT_PATH + get_today_date())
        file_path = os.path.join(PLOT_PATH + get_today_date() + "/", 'validarion.png')
        plt.savefig(file_path)

        plt.show()

    # Summary
    mean_acc = np.mean([f["accuracy"] for f in fold_results])
    mean_f1 = np.mean([f["f1_score"] for f in fold_results])
    safe_log(f"\n📊 Cross-Validation Summary:")
    safe_log(f"🔹 Mean Accuracy: {mean_acc:.4f}")
    safe_log(f"🔹 Mean F1 Score: {mean_f1:.4f}")

    return fold_results

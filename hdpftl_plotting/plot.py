import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from hdpftl_utility.config import PLOT_PATH


# ✅ 1. Global vs Personalized Accuracy per Client

def plot_accuracy_comparison(global_accs, personalized_accs):
    clients = list(global_accs.keys())
    global_values = [global_accs[cid] for cid in clients]
    personal_values = [personalized_accs[cid] for cid in clients]

    x = np.arange(len(clients))
    width = 0.35

    # Create plot directory if it doesn't exist
    os.makedirs(PLOT_PATH, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, global_values, width, label='Global Model')
    plt.bar(x + width / 2, personal_values, width, label='Personalized Model')

    plt.xlabel('Client ID')
    plt.ylabel('Accuracy')
    plt.title('Global vs Personalized Model Accuracy per Client')
    plt.xticks(x, [f"Client {i}" for i in clients])
    plt.ylim(0, 1.0)  # Assuming accuracy is between 0 and 1
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    file_path = os.path.join(PLOT_PATH, 'plot_accuracy_comparison_glo_personalized.png')
    plt.savefig(file_path)
    plt.show()
    print(f"✅ Accuracy comparison plot saved at: {file_path}")


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

    file_path = os.path.join(PLOT_PATH, name)
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
    file_path = os.path.join(PLOT_PATH, 'plot_accuracy_heatmap.png')
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
    file_path = os.path.join(PLOT_PATH, 'plot_fine_tuning_improvement.png')
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
    file_path = os.path.join(PLOT_PATH, 'plot_global_personalised.png')
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

    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()

    if class_names is None:
        class_names = [str(i) for i in range(len(np.unique(y_true_np)))]

    norm = 'true' if normalize else None
    cm = confusion_matrix(y_true_np, y_pred_np, normalize=norm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', values_format='.2f' if normalize else 'd')

    plt.title("Normalized Confusion Matrix" if normalize else "Confusion Matrix")
    plt.tight_layout()
    file_path = os.path.join(PLOT_PATH, 'plot_confusion_matrix.png')
    plt.savefig(file_path)
    plt.show()


def plot_client_accuracies(accs, global_acc=None, title="Per-Client Accuracy", save_path=None):
    import matplotlib.pyplot as plt

    client_ids = list(accs.keys())
    accuracies = [accs[cid] for cid in client_ids]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(client_ids, accuracies, color='skyblue', edgecolor='black')
    plt.xlabel("Client ID")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.ylim(0, 1.0)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f"{yval:.2f}", ha='center', va='bottom',
                 fontsize=8)

    # Optionally show global accuracy
    if global_acc is not None:
        plt.axhline(global_acc, color='red', linestyle='--', linewidth=2,
                    label=f'Global Accuracy: {global_acc:.2f}')
        plt.legend()

    plt.grid(axis='y', linestyle='--', alpha=0.7)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    file_path = os.path.join(PLOT_PATH, 'plot_client_accuracies.png')
    plt.savefig(file_path)
    plt.show()


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
    file_path = os.path.join(PLOT_PATH, 'plot_personalized_vs_global.png')
    plt.savefig(file_path)
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
    file_path = os.path.join(PLOT_PATH, 'plot_class_distribution_per_client.png')
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

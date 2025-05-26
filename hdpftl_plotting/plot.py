import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from hdpftl_plotting import predictions


# ✅ 1. Global vs Personalized Accuracy per Client

def plot_accuracy_comparison(global_accs, personalized_accs):
    clients = list(global_accs.keys())
    global_values = [global_accs[cid] for cid in clients]
    personal_values = [personalized_accs[cid] for cid in clients]

    x = np.arange(len(clients))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, global_values, width, label='Global Model')
    plt.bar(x + width / 2, personal_values, width, label='Personalized Model')

    plt.xlabel('Client ID')
    plt.ylabel('Accuracy')
    plt.title('Global vs Personalized Model Accuracy per Client')
    plt.xticks(x, [f"Client {i}" for i in clients])
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


# Training Loss vs Epochs (Global or Local)
"""
plot_training_loss(global_training_losses, label='Global Loss')
plot_training_loss(client_3_losses, label='Client 3 Loss')
"""


def plot_training_loss(losses, label='Loss'):
    plt.figure(figsize=(8, 5))
    plt.plot(losses, marker='o')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel(label)
    plt.grid(True)
    plt.show()


# ✅ 3. Client-wise Accuracy Heatmap

def plot_accuracy_heatmap(accs_dict):
    df = pd.DataFrame(list(accs_dict.items()), columns=['Client', 'Accuracy'])
    df['Client'] = df['Client'].astype(str)

    plt.figure(figsize=(8, 6))
    sns.heatmap(df.set_index('Client').T, annot=True, cmap="YlGnBu", cbar=False)
    plt.title("Client-wise Accuracy Heatmap")
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
    plt.show()


def plot_confusion_matrix(y_test):
    # Convert ground truth and preds to NumPy
    y_true = y_test.cpu().numpy()
    y_pred = predictions.cpu().numpy()

    # Compute and display
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
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
    plt.show()

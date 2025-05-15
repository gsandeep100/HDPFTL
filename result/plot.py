import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from result import predictions


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

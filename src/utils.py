"""
Utility functions for training metrics and visualization.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def compute_accuracy(correct, total):
    """Compute accuracy as a percentage."""
    return 100.0 * correct / total


def plot_loss_curve(train_losses, save_path):
    """Plot and save the training loss curve."""
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker="o", color="steelblue")
    plt.title("Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Loss curve saved to {save_path}")


def plot_accuracy_curve(train_accuracies, save_path):
    """Plot and save the training accuracy curve."""
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, marker="o", color="darkorange")
    plt.title("Training Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Accuracy curve saved to {save_path}")


def plot_confusion_matrix(all_labels, all_preds, save_path):
    """Generate and save a confusion matrix plot using sklearn."""
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))

    fig, ax = plt.subplots(figsize=(9, 9))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title("Confusion Matrix on Test Set")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

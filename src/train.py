"""
Training script for the MNIST CNN classifier.

Usage:
    python src/train.py
    python src/train.py --epochs 5 --batch_size 128 --lr 0.0005 --seed 42
"""

import argparse
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import MNISTNet
from utils import compute_accuracy, plot_loss_curve, plot_accuracy_curve, plot_confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Train a CNN on MNIST")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training and testing")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for Adam optimizer")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_data_loaders(batch_size):
    """Download MNIST and create train/test data loaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST(
        root="data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train the model for one epoch and return average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = compute_accuracy(correct, total)
    return avg_loss, accuracy


def evaluate(model, loader, device):
    """Evaluate the model on the test set. Return accuracy, all labels, and predictions."""
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = compute_accuracy(correct, total)
    return accuracy, all_labels, all_preds


def main():
    args = parse_args()

    # Reproducibility
    set_seed(args.seed)

    # Create outputs directory
    os.makedirs("outputs", exist_ok=True)

    # Device (CPU only)
    device = torch.device("cpu")
    print(f"Using device: {device}")
    print(f"Hyperparameters: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}, seed={args.seed}")
    print("-" * 50)

    # Data
    train_loader, test_loader = get_data_loaders(args.batch_size)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples:     {len(test_loader.dataset)}")
    print("-" * 50)

    # Model, loss, optimizer
    model = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    train_losses = []
    train_accuracies = []

    for epoch in range(1, args.epochs + 1):
        loss, acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(loss)
        train_accuracies.append(acc)
        print(f"Epoch [{epoch}/{args.epochs}]  Loss: {loss:.4f}  Train Acc: {acc:.2f}%")

    print("-" * 50)

    # Evaluation on test set
    test_acc, all_labels, all_preds = evaluate(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.2f}%")
    print("-" * 50)

    # Save metrics to JSON
    metrics = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "seed": args.seed,
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "test_accuracy": test_acc,
    }
    metrics_path = os.path.join("outputs", "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # Generate and save plots
    plot_loss_curve(train_losses, os.path.join("outputs", "loss_curve.png"))
    plot_accuracy_curve(train_accuracies, os.path.join("outputs", "acc_curve.png"))
    plot_confusion_matrix(all_labels, all_preds, os.path.join("outputs", "confusion_matrix.png"))

    print("\nDone! All outputs saved to the outputs/ folder.")


if __name__ == "__main__":
    main()

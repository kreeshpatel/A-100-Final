"""
BUG CASE 09: optimizer.step() called before loss.backward().

Drop-in replacement for src/train.py.

ORIGINAL : optimizer.zero_grad()
           outputs = model(images)
           loss = criterion(outputs, labels)
           loss.backward()
           optimizer.step()
MODIFIED : optimizer.zero_grad()
           outputs = model(images)
           loss = criterion(outputs, labels)
           optimizer.step()    # <- before backward
           loss.backward()

EXPECTED FAILURE (silent on CPU):
    No crash with default settings, but the model never learns.
    optimizer.step() consumes .grad - which has just been zeroed by
    zero_grad() and not yet repopulated by backward(). Each step is a
    no-op; the gradients backward() then computes are discarded by the
    next zero_grad(). Final accuracy stays at random ~10%.

REPRODUCE:
    cp bug_cases/case_09_step_before_backward__train.py src/train.py
    python src/train.py
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
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_data_loaders(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # BUG: order swapped. step() is called BEFORE backward(), so it
        # operates on the freshly-zeroed .grad buffers - the update is a
        # no-op, and the gradients populated by backward() are then thrown
        # away by the next iteration's zero_grad().
        optimizer.step()
        loss.backward()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, compute_accuracy(correct, total)


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    return compute_accuracy(correct, total), all_labels, all_preds


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs("outputs", exist_ok=True)
    device = torch.device("cpu")
    train_loader, test_loader = get_data_loaders(args.batch_size)
    model = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_losses, train_accuracies = [], []
    for epoch in range(1, args.epochs + 1):
        loss, acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(loss)
        train_accuracies.append(acc)
        print(f"Epoch [{epoch}/{args.epochs}]  Loss: {loss:.4f}  Train Acc: {acc:.2f}%")

    test_acc, all_labels, all_preds = evaluate(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.2f}%")

    metrics = {
        "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.lr, "seed": args.seed,
        "train_losses": train_losses, "train_accuracies": train_accuracies,
        "test_accuracy": test_acc,
    }
    with open(os.path.join("outputs", "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    plot_loss_curve(train_losses, os.path.join("outputs", "loss_curve.png"))
    plot_accuracy_curve(train_accuracies, os.path.join("outputs", "acc_curve.png"))
    plot_confusion_matrix(all_labels, all_preds, os.path.join("outputs", "confusion_matrix.png"))


if __name__ == "__main__":
    main()

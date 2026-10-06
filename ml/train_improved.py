from pathlib import Path
import csv
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import create_dataloaders
from improved_model import ImprovedCNN


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SEED = 42
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = Path("models/improved_cnn_best.pth")
HISTORY_PATH = Path("results/metrics/improved_cnn_history.csv")


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------
# Training
# --------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.float().to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = (torch.sigmoid(outputs) >= 0.5).long()
        correct += (predictions == labels.long()).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# --------------------------------------------------
# Validation
# --------------------------------------------------

def validate(model, loader, criterion):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.float().to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            predictions = (torch.sigmoid(outputs) >= 0.5).long()
            correct += (predictions == labels.long()).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# --------------------------------------------------
# Save history
# --------------------------------------------------

def save_history(history):
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    with HISTORY_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_accuracy",
                "validation_loss",
                "validation_accuracy",
            ],
        )

        writer.writeheader()
        writer.writerows(history)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    set_seed(SEED)

    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print()

    train_dataset, validation_dataset, train_loader, validation_loader = (
        create_dataloaders(
            batch_size=BATCH_SIZE,
            num_workers=0,
        )
    )

    print(f"Training samples:   {len(train_dataset):,}")
    print(f"Validation samples: {len(validation_dataset):,}")
    print()

    model = ImprovedCNN().to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_validation_accuracy = 0.0
    best_epoch = 0

    history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        validation_loss, validation_accuracy = validate(
            model,
            validation_loader,
            criterion,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {validation_loss:.4f} | "
            f"Val Acc: {validation_accuracy:.4f}"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch

            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "validation_accuracy": validation_accuracy,
                    "epoch": epoch,
                    "seed": SEED,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                },
                MODEL_PATH,
            )

            print("  -> Best model saved.")

    save_history(history)

    print()
    print("=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    print(f"Best validation accuracy: {best_validation_accuracy:.4f}")
    print(f"Best epoch:               {best_epoch}")
    print(f"Model saved to:           {MODEL_PATH}")
    print(f"History saved to:         {HISTORY_PATH}")


if __name__ == "__main__":
    main()
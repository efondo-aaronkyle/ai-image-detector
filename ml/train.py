from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam

from dataset import create_dataloaders
from model import BaselineCNN


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BATCH_SIZE = 64
EPOCHS = 2
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "baseline_cnn_best.pth"


# --------------------------------------------------
# Training
# --------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).float()

        optimizer.zero_grad()

        logits = model(images)

        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = (logits >= 0).long()

        correct += (predictions == labels.long()).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# --------------------------------------------------
# Validation
# --------------------------------------------------

def validate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).float()

            logits = model(images)

            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)

            predictions = (logits >= 0).long()

            correct += (predictions == labels.long()).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")

    train_dataset, validation_dataset, train_loader, validation_loader = (
        create_dataloaders(
            batch_size=BATCH_SIZE,
            num_workers=0,
        )
    )

    print(f"\nTraining samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(validation_dataset):,}")

    model = BaselineCNN().to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_validation_accuracy = 0.0

    print("\nStarting training...\n")

    for epoch in range(EPOCHS):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            DEVICE,
        )

        validation_loss, validation_accuracy = validate(
            model,
            validation_loader,
            criterion,
            DEVICE,
        )

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_accuracy:.4f} "
            f"Val Loss: {validation_loss:.4f} "
            f"Val Acc: {validation_accuracy:.4f}"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "validation_accuracy": validation_accuracy,
                    "epoch": epoch + 1,
                },
                BEST_MODEL_PATH,
            )

            print(f"  Saved best model -> {BEST_MODEL_PATH}")

    print("\nTraining complete.")
    print(f"Best validation accuracy: {best_validation_accuracy:.4f}")
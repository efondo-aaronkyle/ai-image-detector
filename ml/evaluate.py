from pathlib import Path
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from dataset import CIFAKEDataset, base_transform
from model import BaselineCNN


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BATCH_SIZE = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = Path("models/baseline_cnn_best.pth")

TEST_REAL_DIR = Path("data/CIFAKE/test/REAL")
TEST_FAKE_DIR = Path("data/CIFAKE/test/FAKE")

RESULTS_DIR = Path("results/metrics")
FIGURES_DIR = Path("results/figures")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

METRICS_PATH = RESULTS_DIR / "baseline_cnn_test_metrics.csv"
ROC_PATH = RESULTS_DIR / "baseline_cnn_roc.csv"
CONFUSION_MATRIX_PATH = FIGURES_DIR / "baseline_cnn_confusion_matrix.png"
ROC_FIGURE_PATH = FIGURES_DIR / "baseline_cnn_roc_curve.png"


# --------------------------------------------------
# Build test dataset
# --------------------------------------------------

def create_test_dataset():
    samples = []

    for image_path in sorted(TEST_REAL_DIR.iterdir()):
        if image_path.is_file():
            samples.append(
                {
                    "path": str(image_path),
                    "label": 0,
                }
            )

    for image_path in sorted(TEST_FAKE_DIR.iterdir()):
        if image_path.is_file():
            samples.append(
                {
                    "path": str(image_path),
                    "label": 1,
                }
            )

    return samples


# --------------------------------------------------
# Test Dataset
# --------------------------------------------------

class TestDataset(CIFAKEDataset):
    def __init__(self, samples, transform=None):
        self.samples = [
            (Path(sample["path"]), sample["label"])
            for sample in samples
        ]
        self.transform = transform


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def evaluate(model, loader, device):
    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            logits = model(images)

            probabilities = torch.sigmoid(logits)

            predictions = (probabilities >= 0.5).long()

            all_labels.extend(labels.tolist())
            all_predictions.extend(predictions.cpu().tolist())
            all_probabilities.extend(probabilities.cpu().tolist())

    return (
        all_labels,
        all_predictions,
        all_probabilities,
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_PATH}")

    # ----------------------------------------------
    # Verify model exists
    # ----------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    # ----------------------------------------------
    # Create test dataset
    # ----------------------------------------------

    test_samples = create_test_dataset()

    test_dataset = TestDataset(
        test_samples,
        transform=base_transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print(f"Test samples: {len(test_dataset):,}")

    # ----------------------------------------------
    # Load model
    # ----------------------------------------------

    model = BaselineCNN().to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    print(
        f"Best validation accuracy from checkpoint: "
        f"{checkpoint['validation_accuracy']:.4f}"
    )

    print(
        f"Checkpoint epoch: "
        f"{checkpoint['epoch']}"
    )

    # ----------------------------------------------
    # Evaluate
    # ----------------------------------------------

    print("\nEvaluating on official test set...\n")

    labels, predictions, probabilities = evaluate(
        model,
        test_loader,
        DEVICE,
    )

    # ----------------------------------------------
    # Calculate metrics
    # ----------------------------------------------

    accuracy = accuracy_score(labels, predictions)

    precision = precision_score(
        labels,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        labels,
        probabilities,
    )

    confusion = confusion_matrix(
        labels,
        predictions,
    )

    # ----------------------------------------------
    # Print results
    # ----------------------------------------------

    print("==================================================")
    print("BASELINE CNN — TEST SET RESULTS")
    print("==================================================")

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion)

    print("\nClassification Report:")
    print(
        classification_report(
            labels,
            predictions,
            target_names=["REAL", "FAKE"],
            digits=4,
            zero_division=0,
        )
    )

    # ----------------------------------------------
    # Save scalar metrics
    # ----------------------------------------------

    with METRICS_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(["metric", "value"])

        writer.writerow(["accuracy", accuracy])
        writer.writerow(["precision", precision])
        writer.writerow(["recall", recall])
        writer.writerow(["f1", f1])
        writer.writerow(["roc_auc", roc_auc])

    # ----------------------------------------------
    # Save ROC data
    # ----------------------------------------------

    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        labels,
        probabilities,
    )

    with ROC_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "false_positive_rate",
                "true_positive_rate",
                "threshold",
            ]
        )

        for fpr, tpr, threshold in zip(
            false_positive_rate,
            true_positive_rate,
            thresholds,
        ):
            writer.writerow(
                [
                    fpr,
                    tpr,
                    threshold,
                ]
            )

    # ----------------------------------------------
    # Save confusion matrix figure
    # ----------------------------------------------

    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 5))

    plt.imshow(confusion)

    plt.title("Baseline CNN - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.xticks(
        [0, 1],
        ["REAL", "FAKE"],
    )

    plt.yticks(
        [0, 1],
        ["REAL", "FAKE"],
    )

    for row in range(2):
        for column in range(2):
            plt.text(
                column,
                row,
                confusion[row, column],
                ha="center",
                va="center",
            )

    plt.tight_layout()
    plt.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=200,
    )
    plt.close()

    # ----------------------------------------------
    # Save ROC curve
    # ----------------------------------------------

    plt.figure(figsize=(6, 5))

    plt.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"ROC-AUC = {roc_auc:.4f}",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title("Baseline CNN - ROC Curve")

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        ROC_FIGURE_PATH,
        dpi=200,
    )

    plt.close()

    print("\nResults saved:")
    print(f"  Metrics:           {METRICS_PATH}")
    print(f"  ROC data:          {ROC_PATH}")
    print(f"  Confusion matrix:  {CONFUSION_MATRIX_PATH}")
    print(f"  ROC curve:         {ROC_FIGURE_PATH}")
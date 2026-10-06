from pathlib import Path
import csv

import torch
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
import matplotlib.pyplot as plt

from dataset import CIFAKEDataset, base_transform
from improved_model import ImprovedCNN


BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = Path("models/improved_cnn_best.pth")

TEST_REAL_DIR = Path("data/CIFAKE/test/REAL")
TEST_FAKE_DIR = Path("data/CIFAKE/test/FAKE")

METRICS_PATH = Path("results/metrics/improved_cnn_test_metrics.csv")
ROC_PATH = Path("results/metrics/improved_cnn_roc.csv")

CONFUSION_MATRIX_PATH = Path(
    "results/figures/improved_cnn_confusion_matrix.png"
)

ROC_CURVE_PATH = Path(
    "results/figures/improved_cnn_roc_curve.png"
)


def create_test_dataset():
    samples = []

    real_files = sorted(TEST_REAL_DIR.iterdir())
    fake_files = sorted(TEST_FAKE_DIR.iterdir())

    for path in real_files:
        if path.is_file():
            samples.append((path, 0))

    for path in fake_files:
        if path.is_file():
            samples.append((path, 1))

    dataset = CIFAKEDataset.__new__(CIFAKEDataset)

    dataset.split_file = None
    dataset.transform = base_transform
    dataset.samples = samples

    return dataset


def main():
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_PATH}")

    test_dataset = create_test_dataset()

    print(f"Test samples: {len(test_dataset):,}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = ImprovedCNN().to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(
        f"Best validation accuracy from checkpoint: "
        f"{checkpoint['validation_accuracy']:.4f}"
    )
    print(f"Checkpoint epoch: {checkpoint['epoch']}")

    all_labels = []
    all_probabilities = []

    print()
    print("Evaluating on official test set...")

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)

            outputs = model(images)
            probabilities = torch.sigmoid(outputs)

            all_labels.extend(labels.numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    predictions = [
        1 if probability >= 0.5 else 0
        for probability in all_probabilities
    ]

    accuracy = accuracy_score(all_labels, predictions)
    precision = precision_score(all_labels, predictions)
    recall = recall_score(all_labels, predictions)
    f1 = f1_score(all_labels, predictions)
    roc_auc = roc_auc_score(all_labels, all_probabilities)

    cm = confusion_matrix(all_labels, predictions)

    print()
    print("=" * 50)
    print("IMPROVED CNN — TEST SET RESULTS")
    print("=" * 50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    print()
    print("Confusion Matrix:")
    print(cm)

    print()
    print("Classification Report:")
    print(
        classification_report(
            all_labels,
            predictions,
            target_names=["REAL", "FAKE"],
        )
    )

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ROC_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFUSION_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)

    with METRICS_PATH.open("w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(["metric", "value"])
        writer.writerow(["accuracy", accuracy])
        writer.writerow(["precision", precision])
        writer.writerow(["recall", recall])
        writer.writerow(["f1_score", f1])
        writer.writerow(["roc_auc", roc_auc])

    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        all_labels,
        all_probabilities,
    )

    with ROC_PATH.open("w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(
            ["false_positive_rate", "true_positive_rate", "threshold"]
        )

        for fpr, tpr, threshold in zip(
            false_positive_rate,
            true_positive_rate,
            thresholds,
        ):
            writer.writerow([fpr, tpr, threshold])

    plt.figure(figsize=(6, 6))
    plt.imshow(cm)
    plt.title("Improved CNN Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks([0, 1], ["REAL", "FAKE"])
    plt.yticks([0, 1], ["REAL", "FAKE"])

    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
            )

    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=300)
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"ROC-AUC = {roc_auc:.4f}",
    )
    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Improved CNN ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROC_CURVE_PATH, dpi=300)
    plt.close()

    print()
    print("Results saved:")
    print(f"  Metrics:           {METRICS_PATH}")
    print(f"  ROC data:          {ROC_PATH}")
    print(f"  Confusion matrix:  {CONFUSION_MATRIX_PATH}")
    print(f"  ROC curve:         {ROC_CURVE_PATH}")


if __name__ == "__main__":
    main()
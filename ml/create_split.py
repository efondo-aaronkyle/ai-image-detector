from pathlib import Path
import random
import csv

DATASET_DIR = Path("data/CIFAKE")
OUTPUT_DIR = Path("results/experiments")

SEED = 42
VALIDATION_RATIO = 0.20

CLASSES = {
    "REAL": 0,
    "FAKE": 1,
}


def collect_images():
    samples = []

    for class_name, label in CLASSES.items():
        directory = DATASET_DIR / "train" / class_name

        for image_path in sorted(directory.iterdir()):
            if image_path.is_file():
                samples.append((image_path, label))

    return samples


def main():
    random.seed(SEED)

    samples = collect_images()

    print(f"Total official training images: {len(samples):,}")

    # Shuffle deterministically.
    random.shuffle(samples)

    validation_size = int(len(samples) * VALIDATION_RATIO)

    validation_samples = samples[:validation_size]
    training_samples = samples[validation_size:]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_file = OUTPUT_DIR / "train_split.csv"
    validation_file = OUTPUT_DIR / "validation_split.csv"

    with train_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["path", "label"])

        for path, label in training_samples:
            writer.writerow([path.as_posix(), label])

    with validation_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["path", "label"])

        for path, label in validation_samples:
            writer.writerow([path.as_posix(), label])

    print(f"Training samples:   {len(training_samples):,}")
    print(f"Validation samples: {len(validation_samples):,}")

    print("\nClass distribution:")

    for name, label in CLASSES.items():
        train_count = sum(
            1 for _, sample_label in training_samples
            if sample_label == label
        )

        validation_count = sum(
            1 for _, sample_label in validation_samples
            if sample_label == label
        )

        print(
            f"  {name}: "
            f"train={train_count:,}, "
            f"validation={validation_count:,}"
        )

    print("\nSplit files created:")
    print(f"  {train_file}")
    print(f"  {validation_file}")

    print("\nOfficial test set remains untouched:")
    print("  data/CIFAKE/test/")


if __name__ == "__main__":
    main()
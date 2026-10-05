from pathlib import Path
import csv

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CIFAKEDataset(Dataset):
    """
    PyTorch Dataset for CIFAKE split CSV files.

    Labels:
        0 = REAL
        1 = FAKE
    """

    def __init__(self, split_file, transform=None):
        self.split_file = Path(split_file)
        self.transform = transform

        self.samples = []

        with self.split_file.open("r", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            for row in reader:
                self.samples.append(
                    (
                        Path(row["path"]),
                        int(row["label"]),
                    )
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)

        return image, label


# CIFAKE images are already 32x32 RGB.
# Convert pixel values from [0, 255] to normalized tensors.
base_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ]
)


def create_dataloaders(batch_size=64, num_workers=0):
    train_dataset = CIFAKEDataset(
        "results/experiments/train_split.csv",
        transform=base_transform,
    )

    validation_dataset = CIFAKEDataset(
        "results/experiments/validation_split.csv",
        transform=base_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return (
        train_dataset,
        validation_dataset,
        train_loader,
        validation_loader,
    )


if __name__ == "__main__":
    train_dataset, validation_dataset, train_loader, validation_loader = (
        create_dataloaders()
    )

    print(f"Training samples:   {len(train_dataset):,}")
    print(f"Validation samples: {len(validation_dataset):,}")

    images, labels = next(iter(train_loader))

    print("\nFirst training batch:")
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Image dtype:  {images.dtype}")
    print(f"Label dtype:  {labels.dtype}")

    print(f"\nPixel minimum: {images.min().item():.4f}")
    print(f"Pixel maximum: {images.max().item():.4f}")

    print(f"\nLabels in batch: {labels.tolist()}")
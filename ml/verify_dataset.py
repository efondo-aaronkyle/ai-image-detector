from pathlib import Path
from PIL import Image

DATASET_DIR = Path("data/CIFAKE")
CLASSES = ["REAL", "FAKE"]
SPLITS = ["train", "test"]

total_images = 0
corrupted_images = []
dimension_counts = {}
mode_counts = {}

for split in SPLITS:
    for class_name in CLASSES:
        directory = DATASET_DIR / split / class_name

        image_files = [
            path for path in directory.iterdir()
            if path.is_file()
        ]

        print(f"{split}/{class_name}: {len(image_files):,} files")

        for image_path in image_files:
            total_images += 1

            try:
                with Image.open(image_path) as image:
                    image.verify()

                # Re-open because verify() invalidates the image object.
                with Image.open(image_path) as image:
                    dimensions = image.size
                    mode = image.mode

                dimension_counts[dimensions] = (
                    dimension_counts.get(dimensions, 0) + 1
                )

                mode_counts[mode] = (
                    mode_counts.get(mode, 0) + 1
                )

            except Exception as exc:
                corrupted_images.append(
                    (str(image_path), str(exc))
                )

print("\n" + "=" * 50)
print("DATASET VERIFICATION")
print("=" * 50)

print(f"Total images: {total_images:,}")

print("\nImage dimensions:")
for dimensions, count in sorted(dimension_counts.items()):
    print(f"  {dimensions}: {count:,}")

print("\nImage modes:")
for mode, count in sorted(mode_counts.items()):
    print(f"  {mode}: {count:,}")

print(f"\nCorrupted images: {len(corrupted_images):,}")

if corrupted_images:
    print("\nCorrupted files:")
    for path, error in corrupted_images[:20]:
        print(f"  {path}")
        print(f"    {error}")

print("\nVerification complete.")
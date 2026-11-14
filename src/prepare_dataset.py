
from pathlib import Path
from PIL import Image, ImageFile
from tqdm import tqdm
import json

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "dataset" / "raw" / "Indian currency dataset v1"
OUTPUT_DIR = BASE_DIR / "dataset" / "processed"

# Target classes
CLASSES = ["10", "20", "50", "100", "200", "500", "Background"]
SPLITS = ["training", "validation", "test"]

IMAGE_SIZE = (224, 224)
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def preprocess_image(image_path, output_path):
    try:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = image.resize(IMAGE_SIZE)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path, format="JPEG", quality=95)
        return True
    except Exception as error:
        print(f"Could not process {image_path}: {error}")
        return False


def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Dataset folder not found: {RAW_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    counts = {}

    for split in SPLITS:
        split_dir = RAW_DIR / split
        counts[split] = {}

        if not split_dir.exists():
            print(f"Missing split: {split}")
            continue

        for class_name in CLASSES:
            source_dir = split_dir / class_name
            destination_dir = OUTPUT_DIR / split / class_name

            counts[split][class_name] = {
                "processed": 0,
                "failed": 0
            }

            if not source_dir.exists():
                print(f"Class folder not found: {source_dir}")
                continue

            images = [
                path for path in source_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
            ]

            print(f"\nProcessing {split}/{class_name}: {len(images)} images")

            for index, image_path in enumerate(tqdm(images)):
                output_path = destination_dir / f"{index:05d}.jpg"

                if preprocess_image(image_path, output_path):
                    counts[split][class_name]["processed"] += 1
                else:
                    counts[split][class_name]["failed"] += 1

    report_path = OUTPUT_DIR / "preprocessing_report.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(counts, file, indent=4)

    print("\nPREPROCESSING COMPLETED!")
    print(f"Processed dataset: {OUTPUT_DIR}")
    print(f"Report: {report_path}")

    print("\nSUMMARY")
    for split, classes in counts.items():
        print(f"\n{split.upper()}")
        for class_name, result in classes.items():
            print(
                f"{class_name}: "
                f"{result['processed']} processed, "
                f"{result['failed']} failed"
            )


if __name__ == "__main__":
    main()

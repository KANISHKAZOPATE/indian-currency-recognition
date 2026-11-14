
from pathlib import Path
from PIL import Image
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_DIR = BASE_DIR / "dataset" / "raw" / "Indian currency dataset v1" / "test"
OUTPUT_DIR = BASE_DIR / "dataset" / "processed" / "test"

CLASSES = ["10", "20", "50", "100", "200", "500", "Background"]

def main():
    counts = {name: 0 for name in CLASSES}
    skipped = 0
    failed = 0

    images = [
        p for p in TEST_DIR.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    print(f"Found {len(images)} test images.")

    for index, image_path in enumerate(tqdm(images)):

        # Extract denomination from filename
        filename = image_path.stem

        if filename.startswith("Background"):
            class_name = "Background"
        elif filename.startswith("2000"):
            skipped += 1
            continue
        else:
            class_name = filename.split(".")[0].split("_")[0]

        if class_name not in CLASSES:
            skipped += 1
            continue

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image = image.resize((224, 224))

                destination = OUTPUT_DIR / class_name
                destination.mkdir(parents=True, exist_ok=True)

                output_path = destination / f"{index:05d}.jpg"
                image.save(output_path, "JPEG", quality=95)

                counts[class_name] += 1

        except Exception as error:
            print(f"Failed: {image_path.name}: {error}")
            failed += 1

    print("\nTEST PREPROCESSING SUMMARY")

    for name, count in counts.items():
        print(f"{name}: {count} processed")

    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    print(f"Total processed: {sum(counts.values())}")


if __name__ == "__main__":
    main()

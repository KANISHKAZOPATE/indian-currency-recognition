
from pathlib import Path
import json

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_DIR = BASE_DIR / "dataset" / "processed" / "test"
MODEL_PATH = BASE_DIR / "artifacts" / "models" / "efficientnet_b0_best.pth"
RESULT_DIR = BASE_DIR / "results" / "reports"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16

print("INDIAN CURRENCY MODEL EVALUATION")
print("Architecture: EfficientNet-B0")
print("Device:", DEVICE)

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load test dataset
test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# Load checkpoint
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

classes = checkpoint["classes"]

print("\nClasses:", classes)
print("Test images:", len(test_dataset))

assert test_dataset.class_to_idx == checkpoint["class_to_idx"], (
    "Test and training class mappings do not match!"
)

# Rebuild EfficientNet-B0
model = models.efficientnet_b0(weights=None)

in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, len(classes))

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()

# Predict
actual_labels = []
predicted_labels = []

print("\nEvaluating test images...")

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)

        outputs = model(images)
        predictions = outputs.argmax(dim=1).cpu()

        actual_labels.extend(labels.tolist())
        predicted_labels.extend(predictions.tolist())

# Calculate metrics
accuracy = accuracy_score(actual_labels, predicted_labels)

report = classification_report(
    actual_labels,
    predicted_labels,
    labels=list(range(len(classes))),
    target_names=classes,
    output_dict=True,
    zero_division=0
)

matrix = confusion_matrix(
    actual_labels,
    predicted_labels,
    labels=list(range(len(classes)))
)

print("\n" + "=" * 50)
print("TEST EVALUATION RESULTS")
print("=" * 50)

print(f"Test Accuracy: {accuracy * 100:.2f}%")

print("\nCLASSIFICATION REPORT")

print(classification_report(
    actual_labels,
    predicted_labels,
    labels=list(range(len(classes))),
    target_names=classes,
    zero_division=0
))

print("\nCONFUSION MATRIX")
print(pd.DataFrame(
    matrix,
    index=classes,
    columns=classes
))

# Save results
results = {
    "architecture": "EfficientNet-B0",
    "test_accuracy": accuracy,
    "test_images": len(test_dataset),
    "classes": classes,
    "classification_report": report,
    "confusion_matrix": matrix.tolist()
}

report_path = RESULT_DIR / "evaluation_report.json"

with open(report_path, "w") as file:
    json.dump(results, file, indent=4)

pd.DataFrame(
    matrix,
    index=classes,
    columns=classes
).to_csv(RESULT_DIR / "confusion_matrix.csv")

print("\nEvaluation completed!")
print("Report saved:", report_path)
print("Confusion matrix saved.")

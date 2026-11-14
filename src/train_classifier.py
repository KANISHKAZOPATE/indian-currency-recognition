
import json
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

# ---------------- CONFIGURATION ----------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset" / "processed"
MODEL_DIR = BASE_DIR / "artifacts" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "efficientnet_b0_best.pth"

CLASSES = ["10", "20", "50", "100", "200", "500", "Background"]

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 0.0001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 55)
print("INDIAN CURRENCY CLASSIFICATION")
print("Architecture: EfficientNet-B0")
print("=" * 55)

print(f"Device: {DEVICE}")
print(f"Classes: {CLASSES}")

# ---------------- DATA TRANSFORMS ----------------

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- LOAD DATASETS ----------------

train_dataset = datasets.ImageFolder(
    DATA_DIR / "training",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    DATA_DIR / "validation",
    transform=eval_transform
)

print("\nClass mapping:", train_dataset.class_to_idx)

assert train_dataset.class_to_idx == val_dataset.class_to_idx, (
    "Training and validation class mappings do not match!"
)

assert len(train_dataset) > 0
assert len(val_dataset) > 0

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")

# ---------------- LOAD EFFICIENTNET-B0 ----------------

print("\nLoading EfficientNet-B0 pretrained weights...")

try:
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    print("Pretrained ImageNet weights loaded.")
except Exception as error:
    print(f"Could not load pretrained weights: {error}")
    print("Using randomly initialized EfficientNet-B0.")
    model = models.efficientnet_b0(weights=None)

# Replace final classifier for 7 classes
in_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    in_features,
    len(CLASSES)
)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)

# ---------------- TRAINING FUNCTION ----------------

def train_one_epoch():
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(train_loader, desc="Training")

    for images, labels in progress:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        progress.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / total, correct / total


# ---------------- VALIDATION FUNCTION ----------------

def validate():
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


# ---------------- MAIN TRAINING LOOP ----------------

best_accuracy = 0.0
best_loss = float("inf")
best_weights = copy.deepcopy(model.state_dict())

history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": []
}

print("\nStarting training...\n")

for epoch in range(EPOCHS):

    print(f"\nEPOCH {epoch + 1}/{EPOCHS}")
    print("-" * 45)

    train_loss, train_accuracy = train_one_epoch()
    val_loss, val_accuracy = validate()

    scheduler.step(val_loss)

    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_accuracy)
    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_accuracy)

    print(f"\nTrain Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

    if val_accuracy > best_accuracy or (
        val_accuracy == best_accuracy and val_loss < best_loss
    ):
        best_accuracy = val_accuracy
        best_loss = val_loss
        best_weights = copy.deepcopy(model.state_dict())

        torch.save({
            "model_state_dict": best_weights,
            "classes": train_dataset.classes,
            "class_to_idx": train_dataset.class_to_idx,
            "architecture": "EfficientNet-B0",
            "image_size": IMAGE_SIZE
        }, MODEL_PATH)

        print("Best model saved!")

# ---------------- SAVE TRAINING HISTORY ----------------

model.load_state_dict(best_weights)

history_path = MODEL_DIR / "training_history.json"

with open(history_path, "w") as file:
    json.dump(history, file, indent=4)

print("\n" + "=" * 55)
print("TRAINING COMPLETED!")
print("=" * 55)

print(f"Best validation accuracy: {best_accuracy * 100:.2f}%")
print(f"Best validation loss: {best_loss:.4f}")
print(f"Best model: {MODEL_PATH}")
print(f"Training history: {history_path}")

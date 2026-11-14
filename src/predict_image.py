
from pathlib import Path
import sys

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "artifacts" / "models" / "efficientnet_b0_best.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if len(sys.argv) < 2:
    print("Usage: python src/predict_image.py path_to_image")
    sys.exit(1)

image_path = Path(sys.argv[1])

if not image_path.exists():
    print("ERROR: Image not found:", image_path)
    sys.exit(1)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
classes = checkpoint["classes"]

model = models.efficientnet_b0(weights=None)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, len(classes))

model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    outputs = model(image_tensor)
    probabilities = torch.softmax(outputs, dim=1)
    confidence, prediction = probabilities.max(dim=1)

predicted_class = classes[prediction.item()]
confidence_percent = confidence.item() * 100

print("\n===================================")
print("INDIAN CURRENCY IMAGE PREDICTION")
print("===================================")
print("Image:", image_path.name)

if predicted_class == "Background":
    print("Result: NOT A CURRENCY NOTE")
else:
    print("Detected denomination: ₹", predicted_class)

print(f"Model confidence: {confidence_percent:.2f}%")
print("===================================")

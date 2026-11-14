
import cv2
import torch
import torch.nn as nn
import subprocess
from datetime import datetime
from pathlib import Path
from PIL import Image
from torchvision import transforms, models


# ==========================================
# CONFIGURATION
# ==========================================

MODEL_PATH = "artifacts/models/efficientnet_b0_best.pth"
SAVE_DIR = Path("results/predictions")
CONFIDENCE_THRESHOLD = 70

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# WINDOWS VOICE SYSTEM
# ==========================================

def speak(message):

    print("VOICE:", message)

    # Escape single quotes for PowerShell
    safe_message = message.replace("'", "''")

    powershell_script = (
        "Add-Type -AssemblyName System.Speech; "
        "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$speaker.Volume = 100; "
        "$speaker.Rate = 0; "
        f"$speaker.Speak('{safe_message}'); "
        "$speaker.Dispose()"
    )

    try:

        subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                powershell_script
            ],
            check=True,
            capture_output=True,
            text=True
        )

        print("VOICE PLAYBACK COMPLETED")

    except subprocess.CalledProcessError as error:

        print("VOICE ERROR:")
        print(error.stderr)

    except Exception as error:

        print("VOICE ERROR:", error)


# ==========================================
# LOAD MODEL
# ==========================================

print("Loading EfficientNet-B0 model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

classes = checkpoint["classes"]
class_to_idx = checkpoint["class_to_idx"]

model = models.efficientnet_b0(weights=None)

in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, len(classes))

model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

idx_to_class = {
    index: class_name
    for class_name, index in class_to_idx.items()
}

print("Model loaded successfully!")
print("Device:", DEVICE)


# ==========================================
# IMAGE TRANSFORMATION
# ==========================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_currency(frame):

    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)

    image_tensor = transform(pil_image)
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted_index = torch.max(
            probabilities, dim=1
        )

    predicted_class = idx_to_class[predicted_index.item()]
    confidence = confidence.item() * 100

    if confidence < CONFIDENCE_THRESHOLD:

        result = "UNCERTAIN"
        voice_message = "Unable to identify the note. Please try again."

    elif predicted_class == "Background":

        result = "NOT A CURRENCY NOTE"
        voice_message = "This is not a currency note."

    else:

        result = f"DETECTED: Rs. {predicted_class}"
        voice_message = f"{predicted_class} rupees note detected."

    return result, confidence, voice_message


# ==========================================
# START WEBCAM
# ==========================================

print("\nStarting webcam...")
print("Press SPACE to capture and predict.")
print("Press Q to quit.")

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("ERROR: Could not open webcam.")
    raise SystemExit


last_result = "Waiting for capture..."
last_confidence = 0.0
last_capture_time = None


try:

    while True:

        ret, frame = camera.read()

        if not ret:

            print("Could not read camera frame.")
            break

        display_frame = frame.copy()

        cv2.putText(
            display_frame,
            "Press SPACE to capture",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            display_frame,
            last_result,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        if last_capture_time is not None:

            cv2.putText(
                display_frame,
                f"Confidence: {last_confidence:.2f}%",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

        cv2.putText(
            display_frame,
            "Q: Quit",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Indian Currency Recognition",
            display_frame
        )

        key = cv2.waitKey(1) & 0xFF

        # Quit
        if key == ord("q") or key == ord("Q"):

            break

        # Capture and predict
        elif key == 32:

            print("\nCapturing image...")

            captured_frame = frame.copy()

            result, confidence, voice_message = predict_currency(
                captured_frame
            )

            last_result = result
            last_confidence = confidence
            last_capture_time = datetime.now()

            # Save captured image
            timestamp = last_capture_time.strftime(
                "%Y%m%d_%H%M%S_%f"
            )

            image_path = SAVE_DIR / f"capture_{timestamp}.jpg"

            cv2.imwrite(
                str(image_path),
                captured_frame
            )

            # Display prediction
            print("--------------------------------")
            print("Prediction:", result)
            print(f"Confidence: {confidence:.2f}%")
            print("Saved image:", image_path)
            print("--------------------------------")

            # Speak after EVERY capture
            speak(voice_message)

            print("Ready for next capture.")

finally:

    camera.release()
    cv2.destroyAllWindows()

    print("\nWebcam closed.")

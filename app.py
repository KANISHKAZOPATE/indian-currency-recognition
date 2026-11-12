

import json
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Indian Currency Recognition",
    page_icon="🇮🇳",
    layout="centered"
)

st.title("🇮🇳 Indian Currency Recognition")
st.write("Capture or upload an Indian currency note to identify its denomination.")


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MODEL_PATH = Path("artifacts/models/efficientnet_b0_best.pth")
CONFIDENCE_THRESHOLD = 70


# --------------------------------------------------
# DEVICE
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    classes = checkpoint["classes"]

    idx_to_class = {
        index: label
        for index, label in enumerate(classes)
    }

    model = models.efficientnet_b0(weights=None)

    input_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        input_features,
        len(classes)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model, idx_to_class


# --------------------------------------------------
# IMAGE PREPROCESSING
# Keep the same preprocessing used during training
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# CURRENCY PREDICTION
# --------------------------------------------------

def predict_currency(image, model, idx_to_class):

    image = image.convert("RGB")

    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted_index = torch.max(
            probabilities,
            dim=1
        )

    predicted_class = idx_to_class[
        predicted_index.item()
    ]

    confidence_value = confidence.item() * 100

    if confidence_value < CONFIDENCE_THRESHOLD:

        result = "UNCERTAIN"

        voice_message = (
            "Unable to identify the currency note. "
            "Please try again with a clearer image."
        )

    elif predicted_class == "Background":

        result = "NOT A CURRENCY NOTE"

        voice_message = (
            "This is not an Indian currency note."
        )

    else:

        result = f"₹{predicted_class}"

        voice_message = (
            f"Indian currency note detected. "
            f"{predicted_class} rupees."
        )

    return result, confidence_value, voice_message


# --------------------------------------------------
# AUTOMATIC BROWSER VOICE ANNOUNCEMENT
# --------------------------------------------------

def speak_automatically(message):

    safe_message = json.dumps(message)

    components.html(
        f"""
        <html>
        <body>
        <script>
            const message = {safe_message};

            function speakResult() {{

                if (!("speechSynthesis" in window)) {{
                    console.log("Speech synthesis is not supported.");
                    return;
                }}

                window.speechSynthesis.cancel();

                const utterance =
                    new SpeechSynthesisUtterance(message);

                utterance.lang = "en-IN";
                utterance.rate = 0.9;
                utterance.pitch = 1.0;
                utterance.volume = 1.0;

                const voices = window.speechSynthesis.getVoices();

                const indianVoice = voices.find(
                    voice => voice.lang === "en-IN"
                );

                if (indianVoice) {{
                    utterance.voice = indianVoice;
                }}

                window.speechSynthesis.speak(utterance);
            }}

            if (window.speechSynthesis.getVoices().length > 0) {{
                speakResult();
            }} else {{
                window.speechSynthesis.onvoiceschanged = function() {{
                    speakResult();
                }};

                setTimeout(speakResult, 500);
            }}
        </script>
        </body>
        </html>
        """,
        height=0
    )


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "result" not in st.session_state:
    st.session_state.result = None

if "confidence" not in st.session_state:
    st.session_state.confidence = None

if "voice_message" not in st.session_state:
    st.session_state.voice_message = None

if "speak_now" not in st.session_state:
    st.session_state.speak_now = False


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

try:

    with st.spinner("Loading currency recognition model..."):

        model, idx_to_class = load_model()

    st.success("Currency recognition model loaded successfully.")

except Exception as error:

    st.error(f"Unable to load model: {error}")
    st.stop()


# --------------------------------------------------
# CAMERA OR IMAGE UPLOAD
# --------------------------------------------------

input_option = st.radio(
    "Select input method:",
    ["Camera", "Upload Image"],
    horizontal=True
)

image_file = None

if input_option == "Camera":

    image_file = st.camera_input(
        "Capture currency note"
    )

else:

    image_file = st.file_uploader(
        "Upload currency note image",
        type=["jpg", "jpeg", "png"]
    )


# --------------------------------------------------
# DISPLAY IMAGE AND RECOGNIZE
# --------------------------------------------------

if image_file is not None:

    image = Image.open(image_file).convert("RGB")

    st.image(
        image,
        caption="Selected Currency Image",
        use_container_width=True
    )

    if st.button(
        "🔍 Recognize Currency",
        use_container_width=True
    ):

        with st.spinner("Identifying currency denomination..."):

            result, confidence, voice_message = predict_currency(
                image,
                model,
                idx_to_class
            )

            st.session_state.result = result
            st.session_state.confidence = confidence
            st.session_state.voice_message = voice_message

            # Trigger automatic speech for this recognition
            st.session_state.speak_now = True


# --------------------------------------------------
# DISPLAY RESULT
# --------------------------------------------------

if st.session_state.result is not None:

    st.divider()

    st.subheader("Recognition Result")

    result = st.session_state.result
    confidence = st.session_state.confidence

    if result.startswith("₹"):

        st.success(
            f"Detected Denomination: {result}"
        )

    elif result == "NOT A CURRENCY NOTE":

        st.warning(result)

    else:

        st.error(result)

    st.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )

    st.progress(
        min(max(confidence / 100, 0.0), 1.0)
    )

    # Automatically speak only after a new recognition
    if st.session_state.speak_now:

        message = st.session_state.voice_message

        speak_automatically(message)

        st.session_state.speak_now = False


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Indian Currency Recognition and Voice Assistance System"
)

st.caption(
    "Powered by EfficientNet-B0 and PyTorch"
)

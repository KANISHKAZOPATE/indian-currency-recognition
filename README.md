# Indian Currency Recognition and Voice Assistance Using EfficientNet-B0

## 1. Project Overview

This project presents a deep-learning-based Indian currency recognition system that identifies Indian currency denominations using image classification. It uses the EfficientNet-B0 Convolutional Neural Network (CNN) architecture implemented with PyTorch.

The application allows users to capture an image using a camera or upload an existing image. The trained model predicts the denomination and displays the result along with its confidence score. A voice assistance feature automatically announces the prediction using browser-based speech synthesis.

The system supports six Indian currency denominations: ₹10, ₹20, ₹50, ₹100, ₹200 and ₹500. It also includes a Background class to distinguish non-currency images.

## 2. Project Objectives

- To develop an Indian currency recognition system using deep learning.
- To implement EfficientNet-B0 for image classification.
- To classify six Indian currency denominations and a Background class.
- To provide an easy-to-use interface for image upload and camera capture.
- To provide automatic voice announcements after recognition.
- To evaluate model performance using accuracy, precision, recall, F1-score and a confusion matrix.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| PyTorch | Deep learning framework |
| Torchvision | EfficientNet-B0 and image transformations |
| EfficientNet-B0 | CNN classification architecture |
| Streamlit | Web application interface |
| OpenCV | Desktop webcam capture |
| Pillow | Image processing |
| Scikit-learn | Model evaluation |
| NumPy | Numerical operations |
| Pandas | Evaluation reports |
| Matplotlib | Graph generation |
| JavaScript SpeechSynthesis | Voice assistance |

## 4. Supported Classes

The model recognizes the following seven classes:

- ₹10
- ₹20
- ₹50
- ₹100
- ₹200
- ₹500
- Background (non-currency images)

**Note:** ₹2,000 notes and counterfeit currency detection are not supported in the current implementation.

## 5. Model Architecture

The project uses EfficientNet-B0, a Convolutional Neural Network architecture designed for efficient image classification.

Implementation details:

- Architecture: EfficientNet-B0
- Input image size: 224 × 224 pixels
- Output classes: 7
- Loss function: CrossEntropyLoss
- Optimizer: AdamW
- Learning rate: 0.0001
- Weight decay: 0.01
- Batch size: 16
- Training epochs: 15
- Learning rate scheduler: ReduceLROnPlateau
- Random seed: 42

Training augmentation includes random horizontal flipping, rotation and color jitter.

Images are resized, converted into tensors and normalized using ImageNet mean and standard deviation values.

## 6. Dataset Description

The project uses the Indian Currency Note Images Dataset 2020 from Kaggle.

Dataset source:

https://www.kaggle.com/datasets/vishalmane109/indian-currency-note-images-dataset-2020

The final processed dataset contains 3,508 images.

| Dataset Split | Number of Images |
|---|---:|
| Training | 3,128 |
| Validation | 300 |
| Testing | 80 |
| **Total** | **3,508** |

The dataset contains six currency denomination classes and one Background class.

The dataset is not included in full in this repository. Users should download the dataset from Kaggle and organize it according to the preprocessing scripts.

## 7. Model Performance

The trained model achieved the following results:

| Metric | Result |
|---|---:|
| Training Accuracy | 99.14% |
| Validation Accuracy | 98.67% |
| Testing Accuracy | 98.75% |
| Correct Test Predictions | 79 |
| Incorrect Test Predictions | 1 |
| Total Test Images | 80 |

The model achieved 98.75% test accuracy on the processed test set.

The confusion matrix showed one misclassification: an actual ₹20 note was predicted as ₹200.

These results represent performance on the available dataset and do not guarantee the same accuracy under all real-world lighting, camera and background conditions.

## 8. Application Features

### Image Upload
Users can upload an image of an Indian currency note.

### Camera Capture
Users can capture an image using their device camera.

### Currency Prediction
The trained EfficientNet-B0 model predicts the denomination.

### Confidence Score
The application displays the prediction confidence.

### Confidence Threshold
A 70% confidence threshold is used. Predictions below this threshold are reported as uncertain.

### Background Detection
The Background class helps identify images that are not currency notes.

### Voice Assistance
The application automatically announces the recognition result using browser-based JavaScript SpeechSynthesis.

The desktop webcam script also supports voice announcements through Windows speech synthesis.

## 9. Project Structure

```text
Indian_Currency_Detection/
│
├── app.py
├── requirements.txt
├── README.md
│
├── artifacts/
│   └── models/
│       └── efficientnet_b0_best.pth
│
├── dataset/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── train_classifier.py
│   ├── evaluate_classifier.py
│   ├── prepare_dataset.py
│   ├── prepare_test.py
│   ├── predict_image.py
│   └── webcam_detection.py
│
├── results/
│   ├── reports/
│   └── predictions/
│
└── report_figures/
```

## 10. Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/KANISHKAZOPATE/indian-currency-recognition.git
```

### Step 2: Navigate to the Project Folder

```bash
cd indian-currency-recognition
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 4: Activate the Environment

For Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### Step 5: Install Dependencies

```bash
python -m pip install -r requirements.txt
```

## 11. Running the Application

To start the Streamlit application, execute:

```bash
streamlit run app.py
```

The application will open in your browser.

Select Camera or Upload Image, provide a currency image and click Recognize Currency.

The predicted denomination, confidence score and voice announcement will be displayed or played.

## 12. Training the Model

After downloading and arranging the dataset, run the preprocessing scripts:

```bash
python src/prepare_dataset.py
```

```bash
python src/prepare_test.py
```

Then train the EfficientNet-B0 model:

```bash
python src/train_classifier.py
```

The best model checkpoint is saved in:

```text
artifacts/models/efficientnet_b0_best.pth
```

## 13. Model Evaluation

To evaluate the trained model on the test dataset, run:

```bash
python src/evaluate_classifier.py
```

The evaluation script generates classification metrics and a confusion matrix.

Evaluation reports are saved in:

```text
results/reports/
```

## 14. Additional Prediction Scripts

### Single Image Prediction

```bash
python src/predict_image.py path/to/image.jpg
```

### Desktop Webcam Recognition

```bash
python src/webcam_detection.py
```

Press SPACE to capture and recognize a currency note. Press Q to exit the webcam window.

## 15. Deployment

The Streamlit application is deployed using Streamlit Community Cloud.

**Live Application:**

https://indian-currency-recognition-nbxsayjtxotvn2gdwgrffj.streamlit.app/

**GitHub Repository:**

https://github.com/KANISHKAZOPATE/indian-currency-recognition

## 16. Limitations

- The model supports only six denominations and the Background class.
- ₹2,000 currency recognition is not included.
- Counterfeit currency detection has not been implemented.
- Performance may vary with lighting, image quality, camera angle and background.
- The test dataset contains 80 images, so further testing with larger and more diverse datasets is necessary.
- Voice output depends on browser or device speech-synthesis support and available voices.
- The Streamlit application performs recognition after image capture or upload, not continuous live-video recognition.

## 17. Future Scope

- Extend the dataset with more real-world currency images.
- Include ₹2,000 denomination recognition if required.
- Investigate counterfeit currency detection using appropriate security features and validated datasets.
- Improve robustness under different lighting and camera conditions.
- Explore deployment on mobile and embedded devices.
- Investigate multilingual voice assistance.

## 18. Conclusion

This project demonstrates the application of deep learning for Indian currency recognition using EfficientNet-B0. The trained model achieved 98.75% accuracy on the processed test dataset containing 80 images. The Streamlit interface provides image upload and camera capture facilities, while automatic voice assistance makes the recognition result accessible through spoken output.

The project combines image preprocessing, CNN-based classification, model evaluation and application deployment into a practical currency recognition system.
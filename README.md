Face Mask Detection System

An AI/ML-based Face Mask Detection System that detects whether a
person is wearing a face mask or not. The project uses MobileNetV2
Transfer Learning, TensorFlow/Keras, and OpenCV for image
classification and real-time face detection. It also includes a Flask
web dashboard for live webcam detection and uploaded-image analysis.

Features

Detects With Mask and Without Mask

Real-time webcam face mask detection

Upload an image and detect mask status

Displays prediction confidence

Supports detection of multiple faces when the face detector finds
them

Displays total, masked, and unmasked detection counts

Green bounding box for masked faces

Red bounding box for faces without masks

Responsive Flask web dashboard

Trained model saved in Keras format for reuse

Model Performance

The model was trained on 7,553 images.

Metric                      Result

Total images                 7,553
Training images              6,043
Validation images            1,510
Validation accuracy     98.94%
Validation loss         0.0392
Input image size         224 × 224
Classification              Binary

Class mapping used during training:

0 -> with_mask
1 -> without_mask

Technologies Used

Python

TensorFlow / Keras

MobileNetV2

Transfer Learning

OpenCV

NumPy

Matplotlib

Scikit-learn

Pillow

Flask

HTML5

CSS3

JavaScript

Project Structure

face_mask_detection/
│
├── data/
│   ├── with_mask/
│   └── without_mask/
│
├── models/
│   └── mask_detector.keras
│
├── src/
│   ├── train_model.py
│   ├── predict_image.py
│   └── webcam_detection.py
│
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── app.js
│   └── uploads/
│
├── templates/
│   └── index.html
│
├── test_images/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore

How the System Works

Image / Webcam
      |
      v
Face Detection using OpenCV
      |
      v
Face Region Extraction
      |
      v
Resize to 224 x 224
      |
      v
MobileNetV2-based Classifier
      |
      v
Mask / No Mask Prediction
      |
      v
Confidence + Bounding Box

The system first detects a face using OpenCV. The detected face is
cropped and resized to 224 × 224. The trained MobileNetV2-based model
then performs binary classification. A prediction near 0 represents
with mask, while a prediction near 1 represents without mask.

For uploaded images, the application can classify the full image as a
fallback when the face detector does not locate a face. This is useful
for already-cropped face images.

Installation

1. Clone the repository

git clone <https://github.com/rajesh2408-tech/Face_Mask_Detection>
cd face_mask_detection

2. Create a virtual environment

On macOS/Linux:

python3 -m venv .venv
source .venv/bin/activate

On Windows:

python -m venv .venv
.venv\Scripts\activate

3. Install dependencies

pip install --upgrade pip
pip install -r requirements.txt

The project uses OpenCV 4.12 because the face-detection implementation
depends on CascadeClassifier.

Example requirements.txt:

tensorflow
opencv-python==4.12.0.88
numpy
matplotlib
scikit-learn
pillow
flask

Training the Model

Make sure the dataset follows this structure:

data/
├── with_mask/
│   ├── image1.jpg
│   └── ...
│
└── without_mask/
    ├── image1.jpg
    └── ...

Train the model with:

python src/train_model.py

The trained model is saved as:

models/mask_detector.keras

Training Technique

The project uses MobileNetV2 Transfer Learning. The pretrained
MobileNetV2 convolutional base is initially frozen and used as a feature
extractor. A custom binary classification layer is trained for face-mask
classification.

Data augmentation is applied during training, including:

Horizontal flipping

Random rotation

Random zoom

Random contrast

These techniques help the model generalize to variations in pose,
lighting, and image conditions.

Test a Single Image

Place a test image inside test_images/ and run:

python src/predict_image.py test_images/person.jpg

Example output:

Prediction Result
--------------------------
Class      : WITH MASK
Confidence : 99.62%
Raw output : 0.0038

Real-Time Webcam Detection

Run:

python src/webcam_detection.py

The application will:

Open the webcam.

Detect faces in each frame.

Extract each detected face.

Pass it to the trained model.

Display MASK or NO MASK with confidence.

Draw the corresponding bounding box.

Press Q to close the webcam window.

On macOS, make sure Terminal or VS Code has camera permission under
System Settings → Privacy & Security → Camera.

Run the Web Application

Activate the virtual environment:

source .venv/bin/activate

Start Flask:

python app.py

Then open the local address displayed by Flask, normally:

http://127.0.0.1:5000

The dashboard provides:

Live camera detection

Image upload detection

Mask/no-mask confidence

Total detection count

With-mask count

Without-mask count

Model information

AI/ML Concepts Used

Convolutional Neural Networks

CNN-based feature extraction enables the system to learn visual patterns
associated with masked and unmasked faces.

Transfer Learning

Instead of training a large CNN entirely from scratch, the project uses
a pretrained MobileNetV2 model. This reduces training requirements
and provides strong image features learned from ImageNet.

Data Augmentation

Random transformations create variations of training images and improve
model generalization.

Binary Classification

The final sigmoid layer outputs a value between 0 and 1:

Prediction < 0.5  -> WITH MASK
Prediction >= 0.5 -> WITHOUT MASK

Computer Vision

OpenCV handles webcam capture, face localization, image manipulation,
bounding boxes, and real-time frame processing.

Key Files

File                                Purpose

src/train_model.py                Loads the dataset and trains the
MobileNetV2 model

src/predict_image.py              Tests the trained model on a single
image

src/webcam_detection.py           Performs real-time webcam detection

models/mask_detector.keras        Saved trained model

app.py                            Flask backend and AI inference

templates/index.html              Web dashboard

static/css/style.css              Dashboard styling

Future Enhancements

Replace Haar Cascade with a more robust modern face detector

Add persistent detection history

Store detections in a database

Add real-time dashboard charts

Add date and time for detection events

Export detection reports

Improve multi-person tracking and duplicate-count prevention

Deploy the application to a cloud platform

Add role-based admin monitoring

Optimize the model for edge/mobile deployment

Applications

This type of computer-vision system can be adapted for controlled
environments such as:

Hospitals and healthcare facilities

Laboratories

Manufacturing facilities

Offices

Educational institutions

Public-entry monitoring systems

Note: Predictions are produced by a machine-learning model and can
be affected by lighting, face angle, occlusion, image quality, and
dataset limitations. The system should not be treated as a substitute
for safety-critical human judgment.

Author

Rajesh Cheruku

License

This project is intended for educational and portfolio purposes. Add a
specific open-source license (such as MIT) before distributing it under
that license.
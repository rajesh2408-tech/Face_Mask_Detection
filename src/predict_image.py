import os
import sys

import numpy as np
import tensorflow as tf
from PIL import Image


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "mask_detector.keras")

IMG_SIZE = (224, 224)

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

if len(sys.argv) < 2:
    print("Usage:")
    print("python src/predict_image.py path/to/image.jpg")
    sys.exit()

image_path = sys.argv[1]

if not os.path.exists(image_path):
    print("Image not found:", image_path)
    sys.exit()

image = Image.open(image_path).convert("RGB")
image = image.resize(IMG_SIZE)

image_array = np.array(image, dtype=np.float32)
image_array = np.expand_dims(image_array, axis=0)

prediction = model.predict(image_array, verbose=0)[0][0]

if prediction < 0.5:
    label = "WITH MASK"
    confidence = (1 - prediction) * 100
else:
    label = "WITHOUT MASK"
    confidence = prediction * 100

print("\nPrediction Result")
print("--------------------------")
print("Class      :", label)
print(f"Confidence : {confidence:.2f}%")
print(f"Raw output : {prediction:.4f}")
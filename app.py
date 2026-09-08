import os
import uuid

import cv2
import numpy as np
import tensorflow as tf

from flask import (
    Flask,
    render_template,
    Response,
    request,
    jsonify,
    send_from_directory
)
from werkzeug.utils import secure_filename


# ==========================================================
# CONFIGURATION
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "mask_detector.keras"
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg"
}

IMG_SIZE = (224, 224)
MASK_THRESHOLD = 0.5

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ==========================================================
# FLASK APP
# ==========================================================

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# ==========================================================
# LOAD AI MODEL
# ==========================================================

print("Loading Face Mask Detection model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ==========================================================
# LOAD FACE DETECTOR
# ==========================================================

cascade_path = os.path.join(
    cv2.data.haarcascades,
    "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    raise RuntimeError(
        "Could not load OpenCV face detector."
    )

print("Face detector loaded successfully.")


# ==========================================================
# GLOBAL CAMERA
# ==========================================================

camera = None


# ==========================================================
# UTILITY
# ==========================================================

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ==========================================================
# PREDICT ONE FACE
# ==========================================================

def predict_face(face):

    face_rgb = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2RGB
    )

    face_rgb = cv2.resize(
        face_rgb,
        IMG_SIZE
    )

    face_array = np.asarray(
        face_rgb,
        dtype=np.float32
    )

    face_array = np.expand_dims(
        face_array,
        axis=0
    )

    prediction = float(
        model.predict(
            face_array,
            verbose=0
        )[0][0]
    )

    # Class mapping:
    # 0 = with_mask
    # 1 = without_mask

    if prediction < MASK_THRESHOLD:

        label = "MASK"

        confidence = (
            1 - prediction
        ) * 100

        color = (0, 180, 0)

    else:

        label = "NO MASK"

        confidence = (
            prediction
        ) * 100

        color = (0, 0, 255)

    return (
        label,
        confidence,
        color
    )


# ==========================================================
# PROCESS IMAGE
# ==========================================================

def process_frame(frame, classify_full_image_if_no_face=False):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.equalizeHist(gray)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(30, 30)
    )

    mask_count = 0
    no_mask_count = 0
    results = []

    # -----------------------------------------------------
    # FALLBACK FOR UPLOADED FACE IMAGES
    # -----------------------------------------------------

    if len(faces) == 0 and classify_full_image_if_no_face:

        label, confidence, color = predict_face(frame)

        if label == "MASK":
            mask_count += 1
        else:
            no_mask_count += 1

        results.append({
            "label": label,
            "confidence": round(confidence, 2)
        })

        height, width = frame.shape[:2]

        text = f"{label} {confidence:.1f}%"

        cv2.rectangle(
            frame,
            (5, 5),
            (width - 5, height - 5),
            color,
            3
        )

        cv2.rectangle(
            frame,
            (5, 5),
            (min(width - 5, 300), 45),
            color,
            -1
        )

        cv2.putText(
            frame,
            text,
            (15, 33),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

        return frame, {
            "total": 1,
            "mask": mask_count,
            "no_mask": no_mask_count,
            "results": results
        }

    # -----------------------------------------------------
    # PROCESS DETECTED FACES
    # -----------------------------------------------------

    for (x, y, w, h) in faces:

        # Slight padding around face
        padding_x = int(w * 0.10)
        padding_y = int(h * 0.10)

        x1 = max(0, x - padding_x)
        y1 = max(0, y - padding_y)

        x2 = min(
            frame.shape[1],
            x + w + padding_x
        )

        y2 = min(
            frame.shape[0],
            y + h + padding_y
        )

        face = frame[
            y1:y2,
            x1:x2
        ]

        if face.size == 0:
            continue

        label, confidence, color = predict_face(face)

        if label == "MASK":
            mask_count += 1
        else:
            no_mask_count += 1

        results.append({
            "label": label,
            "confidence": round(
                confidence,
                2
            )
        })

        text = f"{label} {confidence:.1f}%"

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            3
        )

        text_y = max(
            y1 - 10,
            25
        )

        cv2.putText(
            frame,
            text,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    statistics = {
        "total": len(results),
        "mask": mask_count,
        "no_mask": no_mask_count,
        "results": results
    }

    return frame, statistics


# ==========================================================
# HOME
# ==========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ==========================================================
# START CAMERA
# ==========================================================

@app.route("/start-camera", methods=["POST"])
def start_camera():

    global camera

    if camera is None or not camera.isOpened():

        camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        return jsonify({
            "success": False,
            "message":
                "Unable to access camera."
        }), 500

    return jsonify({
        "success": True
    })


# ==========================================================
# STOP CAMERA
# ==========================================================

@app.route("/stop-camera", methods=["POST"])
def stop_camera():

    global camera

    if camera is not None:

        camera.release()
        camera = None

    return jsonify({
        "success": True
    })


# ==========================================================
# VIDEO GENERATOR
# ==========================================================

def generate_frames():

    global camera

    while (
        camera is not None
        and camera.isOpened()
    ):

        success, frame = camera.read()

        if not success:
            break

        frame = cv2.flip(
            frame,
            1
        )

        frame, _ = process_frame(
            frame
        )

        success, buffer = cv2.imencode(
            ".jpg",
            frame
        )

        if not success:
            continue

        frame_bytes = (
            buffer.tobytes()
        )

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


# ==========================================================
# VIDEO FEED
# ==========================================================

@app.route("/video-feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype=(
            "multipart/x-mixed-replace;"
            " boundary=frame"
        )
    )


# ==========================================================
# IMAGE UPLOAD
# ==========================================================

@app.route(
    "/predict-image",
    methods=["POST"]
)
def predict_image():

    if "image" not in request.files:

        return jsonify({
            "success": False,
            "message": "No image uploaded."
        }), 400

    file = request.files["image"]

    if file.filename == "":

        return jsonify({
            "success": False,
            "message":
                "Please select an image."
        }), 400

    if not allowed_file(file.filename):

        return jsonify({
            "success": False,
            "message":
                "Only JPG, JPEG and PNG are allowed."
        }), 400

    original_name = secure_filename(
        file.filename
    )

    extension = (
        original_name
        .rsplit(".", 1)[1]
        .lower()
    )

    unique_name = (
        f"{uuid.uuid4().hex}.{extension}"
    )

    original_path = os.path.join(
        UPLOAD_FOLDER,
        unique_name
    )

    file.save(original_path)

    image = cv2.imread(
        original_path
    )

    if image is None:

        return jsonify({
            "success": False,
            "message":
                "Could not read image."
        }), 400

    processed_image, statistics = (
        process_frame(
            image,
            classify_full_image_if_no_face=True
        )
    )

    result_name = (
        f"result_{unique_name}"
    )

    result_path = os.path.join(
        UPLOAD_FOLDER,
        result_name
    )

    cv2.imwrite(
        result_path,
        processed_image
    )

    return jsonify({
        "success": True,

        "image_url":
            f"/uploads/{result_name}",

        "total":
            statistics["total"],

        "mask":
            statistics["mask"],

        "no_mask":
            statistics["no_mask"],

        "results":
            statistics["results"]
    })


# ==========================================================
# SERVE UPLOAD
# ==========================================================

@app.route(
    "/uploads/<filename>"
)
def uploaded_file(filename):

    return send_from_directory(
        UPLOAD_FOLDER,
        filename
    )


# ==========================================================
# RUN SERVER
# ==========================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False
    )
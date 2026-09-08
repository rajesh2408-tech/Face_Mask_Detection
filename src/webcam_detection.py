import os
import cv2
import numpy as np
import tensorflow as tf


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "mask_detector.keras"
)


# =========================================================
# CONFIGURATION
# =========================================================

IMG_SIZE = (224, 224)

MASK_THRESHOLD = 0.5


# =========================================================
# LOAD MASK DETECTION MODEL
# =========================================================

print("Loading mask detection model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Mask detection model loaded successfully.")


# =========================================================
# LOAD OPENCV FACE DETECTOR
# =========================================================

face_cascade_path = os.path.join(
    cv2.data.haarcascades,
    "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(face_cascade_path)

if face_detector.empty():
    print("Error: Could not load face detector.")
    exit()

print("Face detector loaded successfully.")


# =========================================================
# OPEN WEBCAM
# =========================================================

print("Opening webcam...")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open webcam.")
    print("Check macOS camera permissions.")
    exit()

print("Webcam started successfully.")
print("Press Q to quit.")


# =========================================================
# REAL-TIME DETECTION
# =========================================================

while True:

    success, frame = camera.read()

    if not success:
        print("Error: Could not read frame.")
        break

    # Flip image so webcam behaves like a mirror
    frame = cv2.flip(frame, 1)

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Detect faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    # =====================================================
    # PROCESS EACH DETECTED FACE
    # =====================================================

    for (x, y, w, h) in faces:

        # Extract face
        face = frame[
            y:y + h,
            x:x + w
        ]

        if face.size == 0:
            continue

        # OpenCV uses BGR
        # Training images were loaded as RGB
        face_rgb = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB
        )

        # Resize to model input size
        face_rgb = cv2.resize(
            face_rgb,
            IMG_SIZE
        )

        # Convert to float
        face_array = np.array(
            face_rgb,
            dtype=np.float32
        )

        # Add batch dimension
        face_array = np.expand_dims(
            face_array,
            axis=0
        )

        # =================================================
        # PREDICTION
        # =================================================

        prediction = model.predict(
            face_array,
            verbose=0
        )[0][0]

        # Our class mapping:
        #
        # 0 -> with_mask
        # 1 -> without_mask

        if prediction < MASK_THRESHOLD:

            label = "MASK"

            confidence = (
                1 - prediction
            ) * 100

            # Green
            color = (0, 255, 0)

        else:

            label = "NO MASK"

            confidence = prediction * 100

            # Red
            color = (0, 0, 255)

        # =================================================
        # LABEL
        # =================================================

        display_text = (
            f"{label} {confidence:.1f}%"
        )

        # =================================================
        # DRAW FACE BOUNDING BOX
        # =================================================

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            color,
            2
        )

        # =================================================
        # DRAW LABEL BACKGROUND
        # =================================================

        (text_width, text_height), _ = (
            cv2.getTextSize(
                display_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                2
            )
        )

        label_y = max(
            y - 10,
            text_height + 10
        )

        cv2.rectangle(
            frame,
            (x, label_y - text_height - 10),
            (x + text_width + 10, label_y + 5),
            color,
            -1
        )

        # =================================================
        # DISPLAY TEXT
        # =================================================

        cv2.putText(
            frame,
            display_text,
            (x + 5, label_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    # =====================================================
    # SHOW NUMBER OF FACES
    # =====================================================

    cv2.putText(
        frame,
        f"Faces: {len(faces)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # =====================================================
    # INSTRUCTION
    # =====================================================

    cv2.putText(
        frame,
        "Press Q to quit",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # =====================================================
    # DISPLAY WINDOW
    # =====================================================

    cv2.imshow(
        "AI Face Mask Detection",
        frame
    )

    # Q key exits application
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================================================
# CLEANUP
# =========================================================

camera.release()

cv2.destroyAllWindows()

print("Face Mask Detection stopped.")
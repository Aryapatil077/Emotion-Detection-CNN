import cv2
import numpy as np
import tensorflow as tf


# =========================================
# LOAD TRAINED MODEL
# =========================================

model = tf.keras.models.load_model("emotion_model.h5")


# =========================================
# EMOTION LABELS
# =========================================

emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]


# =========================================
# FACE DETECTOR
# =========================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# =========================================
# START WEBCAM
# =========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open webcam.")
    exit()

print("Webcam started.")
print("Press Q to quit.")


# =========================================
# LIVE EMOTION DETECTION
# =========================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Cannot read webcam.")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    # Process every detected face
    for (x, y, w, h) in faces:

        # Draw rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

        # Crop face
        face = gray[
            y:y + h,
            x:x + w
        ]

        # Resize to 48x48
        face = cv2.resize(
            face,
            (48, 48)
        )

        # Normalize
        face = face.astype(
            "float32"
        ) / 255.0

        # Reshape for CNN
        face = np.reshape(
            face,
            (1, 48, 48, 1)
        )

        # Predict emotion
        prediction = model.predict(
            face,
            verbose=0
        )

        # Get highest probability
        emotion_index = np.argmax(
            prediction
        )

        emotion = emotion_labels[
            emotion_index
        ]

        confidence = (
            np.max(prediction) * 100
        )

        # Display emotion
        text = (
            f"{emotion} "
            f"{confidence:.1f}%"
        )

        cv2.putText(
            frame,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # Show webcam
    cv2.imshow(
        "Live Emotion Detection",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================================
# RELEASE CAMERA
# =========================================

cap.release()
cv2.destroyAllWindows()

print("Emotion detection stopped.")
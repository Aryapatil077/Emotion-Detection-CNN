import cv2
import numpy as np
import pygame
import random
import tensorflow as tf


# ==============================
# LOAD MODEL
# ==============================

model = tf.keras.models.load_model(
    "emotion_model.h5"
)


# ==============================
# EMOTION LABELS
# ==============================

emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]


# ==============================
# FACE DETECTOR
# ==============================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# ==============================
# PYGAME SETUP
# ==============================

pygame.init()

WIDTH = 800
HEIGHT = 600

screen = pygame.display.set_mode(
    (WIDTH, HEIGHT)
)

pygame.display.set_caption(
    "Emotion Detection Emoji Game"
)


font = pygame.font.SysFont(
    "Arial",
    35
)

small_font = pygame.font.SysFont(
    "Arial",
    25
)

clock = pygame.time.Clock()


# ==============================
# GAME VARIABLES
# ==============================

score = 0

target_emotion = random.choice(
    emotion_labels
)

emoji_x = random.randint(
    100,
    650
)

emoji_y = 100

speed = 3

detected_emotion = "Neutral"


# ==============================
# CAMERA
# ==============================

cap = cv2.VideoCapture(0)


# ==============================
# MAIN LOOP
# ==============================

running = True

while running:

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False


    # --------------------------
    # CAMERA FRAME
    # --------------------------

    ret, frame = cap.read()

    if ret:

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_cascade.detectMultiScale(
            gray,
            1.3,
            5
        )


        for (x, y, w, h) in faces:

            face = gray[
                y:y+h,
                x:x+w
            ]

            face = cv2.resize(
                face,
                (48,48)
            )

            face = face / 255.0

            face = np.reshape(
                face,
                (1,48,48,1)
            )


            prediction = model.predict(
                face,
                verbose=0
            )


            emotion_index = np.argmax(
                prediction
            )

            detected_emotion = emotion_labels[
                emotion_index
            ]


    # --------------------------
    # MOVE EMOTION OBJECT
    # --------------------------

    emoji_y += speed


    if emoji_y > HEIGHT:

        emoji_y = 100

        emoji_x = random.randint(
            100,
            650
        )

        target_emotion = random.choice(
            emotion_labels
        )


    # --------------------------
    # CHECK MATCH
    # --------------------------

    if detected_emotion == target_emotion:

        score += 1

        emoji_y = 100

        emoji_x = random.randint(
            100,
            650
        )

        target_emotion = random.choice(
            emotion_labels
        )


    # --------------------------
    # DISPLAY
    # --------------------------

    screen.fill(
        (255,255,255)
    )


    title = font.render(
        "Emotion Detection Game",
        True,
        (0,0,0)
    )

    screen.blit(
        title,
        (220,30)
    )


    detected = small_font.render(
        "Detected: " + detected_emotion,
        True,
        (0,150,0)
    )

    screen.blit(
        detected,
        (50,120)
    )


    target = small_font.render(
        "Match: " + target_emotion,
        True,
        (200,0,0)
    )

    screen.blit(
        target,
        (50,170)
    )


    score_text = small_font.render(
        "Score: " + str(score),
        True,
        (0,0,200)
    )

    screen.blit(
        score_text,
        (650,120)
    )


    emoji = font.render(
        target_emotion,
        True,
        (0,0,0)
    )

    screen.blit(
        emoji,
        (emoji_x, emoji_y)
    )


    pygame.display.update()


    clock.tick(30)


# ==============================
# CLOSE
# ==============================

cap.release()

pygame.quit()
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


# =========================================
# DATASET PATHS
# =========================================

train_dir = "train"
test_dir = "test"


# =========================================
# SETTINGS
# =========================================

IMG_SIZE = 48
BATCH_SIZE = 64
EPOCHS = 20


# =========================================
# DATA PREPROCESSING
# =========================================

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0
)

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0
)


# =========================================
# LOAD TRAINING DATA
# =========================================

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)


# =========================================
# LOAD TESTING DATA
# =========================================

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)


# =========================================
# DISPLAY DATASET INFORMATION
# =========================================

print("\nEmotion Classes:")
print(train_data.class_indices)

print("\nTraining Images:", train_data.samples)
print("Testing Images:", test_data.samples)


# =========================================
# CREATE CNN MODEL
# =========================================

model = Sequential([

    # Convolution Layer 1
    Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(48, 48, 1)
    ),

    MaxPooling2D(
        pool_size=(2, 2)
    ),


    # Convolution Layer 2
    Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    MaxPooling2D(
        pool_size=(2, 2)
    ),


    # Convolution Layer 3
    Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    MaxPooling2D(
        pool_size=(2, 2)
    ),


    # Convert feature maps into vector
    Flatten(),


    # Fully Connected Layer
    Dense(
        128,
        activation="relu"
    ),


    # Dropout
    Dropout(0.5),


    # Output Layer
    Dense(
        7,
        activation="softmax"
    )
])


# =========================================
# MODEL SUMMARY
# =========================================

print("\nCNN MODEL:")
model.summary()


# =========================================
# COMPILE MODEL
# =========================================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# =========================================
# TRAIN MODEL
# =========================================

print("\n========================================")
print("        TRAINING STARTED")
print("========================================\n")


history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=EPOCHS
)


# =========================================
# EVALUATE MODEL
# =========================================

print("\n========================================")
print("        MODEL EVALUATION")
print("========================================\n")


test_loss, test_accuracy = model.evaluate(
    test_data
)


print(
    f"\nTest Accuracy: {test_accuracy * 100:.2f}%"
)

print(
    f"Test Loss: {test_loss:.4f}"
)


# =========================================
# SAVE TRAINED MODEL
# =========================================

model.save(
    "emotion_model.h5"
)


print("\n========================================")
print("     MODEL SAVED SUCCESSFULLY!")
print("========================================")

print("\nSaved file: emotion_model.h5")
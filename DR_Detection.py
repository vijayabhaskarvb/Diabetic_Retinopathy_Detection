import os
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QFileDialog, QVBoxLayout
from PyQt5.QtGui import QPixmap

# Paths
DATASET_DIR = "Diagnosis of Diabetic Retinopathy/train"
MODEL_PATH = "diabetic_retinopathy_model.h5"
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 10  # Reduce for quick testing, increase for better results

# 🔹 Check dataset exists before training
if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(f"Dataset directory '{DATASET_DIR}' not found!")

# 🔹 Load dataset
datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
)

val_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
)

# 🔹 Debugging: Class mapping
print("Class Indices:", train_generator.class_indices)  # {'DR': 0, 'No_DR': 1}

# 🔹 Load or Train Model
if not os.path.exists(MODEL_PATH):
    print("Training new model...")

    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    history = model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS)
    model.save(MODEL_PATH)
    print("Model saved successfully!")

else:
    print("Loading pre-trained model...")
    model = load_model(MODEL_PATH, compile=False)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    print("Model loaded successfully!")

# 🔹 GUI for Image Prediction
class DRDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.model = model

    def initUI(self):
        self.setWindowTitle("Diabetic Retinopathy Detection")
        self.setGeometry(100, 100, 400, 300)
        self.layout = QVBoxLayout()

        self.label = QLabel("Upload an Image", self)
        self.layout.addWidget(self.label)

        self.upload_btn = QPushButton("Upload Image", self)
        self.upload_btn.clicked.connect(self.loadImage)
        self.layout.addWidget(self.upload_btn)

        self.result_label = QLabel("", self)
        self.layout.addWidget(self.result_label)

        self.setLayout(self.layout)

    def loadImage(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg)")
        if fname:
            pixmap = QPixmap(fname)
            self.label.setPixmap(pixmap.scaled(200, 200))
            self.predict(fname)

    def predict(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, IMG_SIZE)
        img = img / 255.0  # Normalize
        img = np.expand_dims(img, axis=0)
        prediction = self.model.predict(img)[0][0]

        # 🔥 FIXED: Adjust Label Mapping
        result = "Diabetic Retinopathy" if prediction < 0.5 else "No Diabetic Retinopathy"
        self.result_label.setText(f"Prediction: {result}")
        print(f"Raw Prediction Value: {prediction} → {result}")

# 🔹 Run GUI
if __name__ == "__main__":
    app = QApplication([])
    window = DRDetectionApp()
    window.show()
    app.exec_()

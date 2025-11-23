import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

DATA_DIR = "dataset"
IMG_SIZE = 150

labels = ["correct_mask", "incorrect_mask"]
data = []
targets = []

print("Loading images...")

for idx, label in enumerate(labels):
    folder = os.path.join(DATA_DIR, label)
    for img in os.listdir(folder):
        path = os.path.join(folder, img)
        image = cv2.imread(path)
        try:
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            data.append(image)
            targets.append(idx)
        except:
            pass

data = np.array(data) / 255.0
targets = to_categorical(targets, num_classes=4)

x_train, x_test, y_train, y_test = train_test_split(data, targets, test_size=0.2)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(4, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Training model...")
model.fit(x_train, y_train, epochs=10, validation_split=0.2)

model.save("model/mask_detector.h5")
print("Model saved as mask_detector.h5")

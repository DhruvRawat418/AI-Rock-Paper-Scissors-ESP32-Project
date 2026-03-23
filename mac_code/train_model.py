import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# --- 1. Load and Prepare the Dataset ---
# Note: We are using alphabetical order here to keep the numbering simple
CLASSES = ['paper', 'rock', 'scissors']
DATASET_DIR = 'dataset'

X = []
y = []

print("Loading images...")
for index, gesture in enumerate(CLASSES):
    folder_path = os.path.join(DATASET_DIR, gesture)
    if not os.path.exists(folder_path):
        print(f"Warning: Folder {folder_path} not found!")
        continue
        
    for filename in os.listdir(folder_path):
        if filename.endswith('.bmp'):
            img_path = os.path.join(folder_path, filename)
            # Read as grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                X.append(img)
                y.append(index)

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Normalize pixel values to be between 0 and 1 (Crucial for neural networks!)
X = X / 255.0

# Reshape to tell TensorFlow it's a 1-channel (grayscale) image: (Batch, 32, 32, 1)
X = X.reshape(-1, 32, 32, 1)

print(f"Total images loaded: {len(X)}")

# Split into training (80%) and testing (20%) data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. Build the TinyML CNN ---
# This architecture is intentionally tiny so it fits in the ESP32's limited RAM
# --- 2. Build the TinyML CNN (TinyMaix Compatible) ---
# We use strides=(2, 2) to shrink the image instead of MaxPooling
model = models.Sequential([
    # First Conv layer, downsamples from 32x32 to 15x15
    layers.Conv2D(8, (3, 3), strides=(2, 2), activation='relu', input_shape=(32, 32, 1)),
    
    # Second Conv layer, downsamples from 15x15 to 7x7
    layers.Conv2D(16, (3, 3), strides=(2, 2), activation='relu'),
    
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(3, activation='softmax') # 3 outputs for Paper, Rock, Scissors
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- 3. Train the Model ---
print("\nStarting training...")
# 20 epochs is usually plenty for a dataset this simple
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# --- 4. Evaluate and Save ---
test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
print(f"\nFinal Test Accuracy: {test_acc * 100:.2f}%")

# Save the model in the Keras format (.h5) so we can convert it for the ESP32 later
model.save('prs_model.h5')
print("\nModel saved successfully as 'prs_model.h5'")
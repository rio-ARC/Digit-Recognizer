"""
CNN Handwritten Digit Recognizer - Training Script (v2)
Trains a CNN on the MNIST dataset with data augmentation for better
generalization to real-world hand-drawn digits.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# ── 1. Load MNIST Dataset ──
print("Loading MNIST dataset...")
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# ── 2. Preprocess ──
X_train = X_train / 255.0
X_test = X_test / 255.0
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
print(f"Train reshaped: {X_train.shape}")

# ── 3. Visualize Sample Digits ──
print("Saving sample digits visualization...")
plt.figure(figsize=(8, 8))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_train[i].reshape(28, 28), cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
plt.tight_layout()
plt.savefig('sample_digits.png', dpi=150)
plt.close()
print("  -> Saved: sample_digits.png")

# ── 4. Data Augmentation ──
# Simulates real-world drawing variations (off-center, rotated, scaled)
# NO horizontal_flip — flipping digits would break labels
print("\nSetting up data augmentation...")
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    shear_range=8,
    fill_mode='constant',
    cval=0.0  # fill with black (background)
)
datagen.fit(X_train)

# ── 5. Build CNN Model (improved with BatchNorm) ──
print("\nBuilding CNN model...")
model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ── 6. Callbacks ──
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3,
    min_lr=1e-6, verbose=1
)

# ── 7. Train with Augmentation ──
EPOCHS = 25
BATCH_SIZE = 64

# Split training data for validation
val_split = 0.1
val_size = int(len(X_train) * val_split)
X_val = X_train[:val_size]
y_val = y_train[:val_size]
X_train_aug = X_train[val_size:]
y_train_aug = y_train[val_size:]

print(f"\nTraining model ({EPOCHS} epochs with augmentation)...")
history = model.fit(
    datagen.flow(X_train_aug, y_train_aug, batch_size=BATCH_SIZE),
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    callbacks=[reduce_lr],
    steps_per_epoch=len(X_train_aug) // BATCH_SIZE
)

# ── 8. Evaluate ──
print("\nEvaluating on test set...")
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# ── 9. Plot Training History ──
print("\nSaving training history plot...")
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
plt.close()
print("  -> Saved: training_history.png")

# ── 10. Confusion Matrix & Classification Report ──
print("\nGenerating confusion matrix...")
y_pred = model.predict(X_test).argmax(axis=1)
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.close()
print("  -> Saved: confusion_matrix.png")

# ── 11. Save Model ──
model_path = 'mnist_digit_model.keras'
model.save(model_path)
print(f"\nModel saved to: {model_path}")
print("\nDone! Restart the Flask server to load the new model.")

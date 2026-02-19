import sys
import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODEL_PATH = 'mnist_digit_model.keras'


def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28), Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32)

    if np.mean(img_array) > 127:
        img_array = 255.0 - img_array

    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array


def predict_digit(model, image_path):
    img_array = preprocess_image(image_path)

    predictions = model.predict(img_array, verbose=0)
    predicted_digit = np.argmax(predictions[0])
    confidence = predictions[0][predicted_digit] * 100

    print(f"\n{'='*40}")
    print(f"  Predicted Digit: {predicted_digit}")
    print(f"  Confidence:      {confidence:.2f}%")
    print(f"{'='*40}")
    print(f"\n  All class probabilities:")
    for i in range(10):
        bar = '█' * int(predictions[0][i] * 30)
        print(f"    {i}: {predictions[0][i]*100:6.2f}%  {bar}")

    output_path = 'prediction_result.png'
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(img_array.reshape(28, 28), cmap='gray')
    axes[0].set_title(f'Preprocessed Input')
    axes[0].axis('off')

    colors = ['#2196F3'] * 10
    colors[predicted_digit] = '#4CAF50'
    axes[1].barh(range(10), predictions[0] * 100, color=colors)
    axes[1].set_yticks(range(10))
    axes[1].set_xlabel('Confidence (%)')
    axes[1].set_title(f'Prediction: {predicted_digit} ({confidence:.1f}%)')
    axes[1].set_xlim(0, 105)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\n  Visualization saved to: {output_path}")

    return predicted_digit


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        print("Example: python predict.py test_digit.png")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found: {MODEL_PATH}")
        print("Run train_model.py first to train and save the model.")
        sys.exit(1)

    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print(f"Processing image: {image_path}")
    predict_digit(model, image_path)


if __name__ == '__main__':
    main()

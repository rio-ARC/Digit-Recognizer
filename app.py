"""
Flask backend for Handwritten Digit Classifier.
Serves the frontend and handles prediction requests via the trained CNN model.
Includes MNIST-style preprocessing: centers digit with padding before prediction.
"""

import os
import io
import base64
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image, ImageOps
import tensorflow as tf

app = Flask(__name__, static_folder='static', template_folder='static')

# Load model once at startup
MODEL_PATH = 'mnist_digit_model.keras'
print(f"Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")


def preprocess_canvas_image(image_bytes):
    """
    Preprocess a canvas-drawn image to match MNIST format:
    1. Convert to grayscale
    2. Find the bounding box of the drawn digit
    3. Crop to the digit with padding
    4. Make it square (preserve aspect ratio)
    5. Resize to 20x20 (digit area in MNIST)
    6. Center in a 28x28 image (MNIST format)
    7. Normalize to [0, 1]
    """
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    img_array = np.array(img, dtype=np.float32)

    # Invert if background is light (MNIST = white digit on black bg)
    if np.mean(img_array) > 127:
        img_array = 255.0 - img_array

    # Threshold to clean up noise
    img_array[img_array < 30] = 0

    # Find bounding box of the digit (non-zero pixels)
    rows = np.any(img_array > 0, axis=1)
    cols = np.any(img_array > 0, axis=0)

    if not np.any(rows) or not np.any(cols):
        # Empty canvas — return blank image
        return np.zeros((1, 28, 28, 1), dtype=np.float32)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Crop to bounding box
    digit = img_array[rmin:rmax + 1, cmin:cmax + 1]

    # Make it square by padding the shorter dimension
    h, w = digit.shape
    if h > w:
        pad = (h - w) // 2
        digit = np.pad(digit, ((0, 0), (pad, h - w - pad)), mode='constant')
    elif w > h:
        pad = (w - h) // 2
        digit = np.pad(digit, ((pad, w - h - pad), (0, 0)), mode='constant')

    # Convert back to PIL for high-quality resize
    digit_img = Image.fromarray(digit.astype(np.uint8))

    # Resize to 20x20 (MNIST digits occupy ~20x20 area within 28x28)
    digit_img = digit_img.resize((20, 20), Image.LANCZOS)

    # Center in 28x28 canvas (4px padding on each side, like MNIST)
    result = Image.new('L', (28, 28), 0)
    result.paste(digit_img, (4, 4))

    # Normalize
    result_array = np.array(result, dtype=np.float32) / 255.0
    result_array = result_array.reshape(1, 28, 28, 1)

    return result_array


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data = data['image']

        # Remove the data:image/png;base64, prefix
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)

        # Preprocess with MNIST-style centering
        img_array = preprocess_canvas_image(image_bytes)

        # Predict
        predictions = model.predict(img_array, verbose=0)
        predicted_digit = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_digit] * 100)
        probabilities = [round(float(p * 100), 1) for p in predictions[0]]

        return jsonify({
            'digit': predicted_digit,
            'confidence': round(confidence, 1),
            'probabilities': probabilities
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

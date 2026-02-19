
import os
import io
import base64
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image, ImageOps
import tensorflow as tf

app = Flask(__name__, static_folder='static', template_folder='static')

MODEL_PATH = 'mnist_digit_model.keras'
print(f"Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")


def preprocess_canvas_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    img_array = np.array(img, dtype=np.float32)

    if np.mean(img_array) > 127:
        img_array = 255.0 - img_array

    img_array[img_array < 30] = 0

    rows = np.any(img_array > 0, axis=1)
    cols = np.any(img_array > 0, axis=0)

    if not np.any(rows) or not np.any(cols):
        return np.zeros((1, 28, 28, 1), dtype=np.float32)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    digit = img_array[rmin:rmax + 1, cmin:cmax + 1]

    h, w = digit.shape
    if h > w:
        pad = (h - w) // 2
        digit = np.pad(digit, ((0, 0), (pad, h - w - pad)), mode='constant')
    elif w > h:
        pad = (w - h) // 2
        digit = np.pad(digit, ((pad, w - h - pad), (0, 0)), mode='constant')

    digit_img = Image.fromarray(digit.astype(np.uint8))

    digit_img = digit_img.resize((20, 20), Image.LANCZOS)

    result = Image.new('L', (28, 28), 0)
    result.paste(digit_img, (4, 4))

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

        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)

        img_array = preprocess_canvas_image(image_bytes)

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

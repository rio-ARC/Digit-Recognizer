---
title: DigitVision
emoji: ✍️
colorFrom: gray
colorTo: indigo
sdk: docker
pinned: false
---

# DigitVision — Handwritten Digit Classifier

A CNN-based handwritten digit recognizer with a web interface. Draw any digit (0–9) on the canvas and get real-time predictions with confidence scores.

### [🔴 Live Demo](https://rio-arc-digitvision.hf.space/)

![App Preview](view.png)

## Tech Stack

- **Model:** TensorFlow/Keras CNN (99.5% accuracy on MNIST)
- **Backend:** Flask + Gunicorn
- **Frontend:** HTML5 Canvas, Vanilla JS, CSS
- **Deployment:** Docker on Hugging Face Spaces

## Features

- Interactive drawing canvas with touch support
- Real-time digit prediction with confidence bar chart
- MNIST-style preprocessing (bounding box crop, centering, 28×28 normalization)
- Data augmentation trained model (rotation, shift, zoom, shear)
- Dark-themed responsive UI

## Model Architecture

```
Input (28×28×1)
→ Conv2D(32) + BatchNorm + Conv2D(32) + MaxPool + Dropout(0.25)
→ Conv2D(64) + BatchNorm + Conv2D(64) + MaxPool + Dropout(0.25)
→ Dense(256) + BatchNorm + Dropout(0.5)
→ Dense(10, softmax)
```

**Training:** 25 epochs with augmentation, ReduceLROnPlateau callback

## Run Locally

```bash
pip install -r requirements.txt
python train_model.py
python app.py
```

Then open `http://localhost:5000`

## Project Structure

```
├── app.py                  # Flask server + prediction API
├── train_model.py          # Model training with augmentation
├── predict.py              # CLI prediction tool
├── mnist_digit_model.keras # Pre-trained model
├── Dockerfile              # HF Spaces deployment
├── requirements.txt        # Dependencies
└── static/
    ├── index.html          # Frontend
    ├── style.css           # Dark theme + animations
    └── script.js           # Canvas drawing + API calls
```

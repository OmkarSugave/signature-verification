import os
import numpy as np
import cv2
import tensorflow as tf
import urllib.request
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

app = Flask(__name__)

IMG_SIZE = 105

# ✅ Correct base path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "siamese_model.keras")

# 🔥 IMPORTANT: Add your model download link here
MODEL_URL = "https://drive.google.com/file/d/1gm52FWe69BqTL3nLOts92vMa2XIJvmG6/view?usp=sharing"


# 🔥 Download model if not present
if not os.path.exists(MODEL_PATH):
    os.makedirs(os.path.join(BASE_DIR, "model"), exist_ok=True)
    print("Downloading model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

print("Loading model from:", MODEL_PATH)
model = load_model(MODEL_PATH, compile=False)


# 🔥 Preprocess image
def preprocess(file):
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return None

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
    img = img / 255.0

    return img.reshape(1, IMG_SIZE, IMG_SIZE, 1)


# 🔥 Main route
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    score = None

    if request.method == "POST":
        file1 = request.files.get("file1")
        file2 = request.files.get("file2")

        if not file1 or not file2:
            result = "Please upload both images"
            return render_template("index.html", result=result)

        img1 = preprocess(file1)
        img2 = preprocess(file2)

        if img1 is None or img2 is None:
            result = "Error reading images"
            return render_template("index.html", result=result)

        score = float(model.predict([img1, img2])[0][0])

        print("Similarity Score:", score)

        if score > 0.6:
            result = "Genuine Signature ✅"
        elif score > 0.45:
            result = "Suspicious ⚠️"
        else:
            result = "Forged Signature ❌"

    return render_template("index.html", result=result, score=score)


if __name__ == "__main__":
    app.run(debug=True)
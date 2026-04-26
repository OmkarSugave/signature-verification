import os
import numpy as np
import cv2
import tensorflow as tf
import requests
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

app = Flask(__name__)

IMG_SIZE = 105

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "siamese_model.keras")

# ✅ YOUR GOOGLE DRIVE FILE ID
FILE_ID = "1gm52FWe69BqTL3nLOts92vMa2XIJvmG6"

# Google Drive download URL
MODEL_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"


# 🔥 Create model folder
os.makedirs(MODEL_DIR, exist_ok=True)


# 🔥 Handle Google Drive large file download
def download_model():
    print("Downloading model from Google Drive...")

    session = requests.Session()
    response = session.get(MODEL_URL, stream=True)

    # Handle Google Drive confirmation token
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            params = {"id": FILE_ID, "confirm": value}
            response = session.get("https://drive.google.com/uc", params=params, stream=True)
            break

    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(8192):
            if chunk:
                f.write(chunk)

    print("Download complete")


# 🔥 Ensure valid model exists
def ensure_model():
    if not os.path.exists(MODEL_PATH):
        download_model()

    # Validate file size (avoid HTML file issue)
    if os.path.getsize(MODEL_PATH) < 1_000_000:
        print("Downloaded file invalid. Re-downloading...")
        os.remove(MODEL_PATH)
        download_model()


ensure_model()

print("Loading model...")
model = load_model(MODEL_PATH, compile=False)


# 🔥 Preprocessing
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

        if score > 0.6:
            result = "Genuine Signature ✅"
        elif score > 0.45:
            result = "Suspicious ⚠️"
        else:
            result = "Forged Signature ❌"

    return render_template("index.html", result=result, score=score)


if __name__ == "__main__":
    app.run(debug=True)
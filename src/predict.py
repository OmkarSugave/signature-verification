import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = 105

def preprocess(path):
    img = cv2.imread(path, 0)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img.reshape(1, IMG_SIZE, IMG_SIZE, 1)

model = load_model("model/siamese_model.h5", compile=False)

img1 = preprocess("sig1.png")
img2 = preprocess("sig2.png")

score = model.predict([img1, img2])[0][0]

print("Similarity:", score)

if score > 0.5:
    print("Same Person")
else:
    print("Forged") 
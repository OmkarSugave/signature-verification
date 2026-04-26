import os
import cv2
import numpy as np

BASE = "dataset/train"
IMG_SIZE = 105

def create_signature(seed):
    img = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.uint8) * 255
    rng = np.random.RandomState(seed)

    # draw random curves (simulate signature strokes)
    for _ in range(6):
        points = rng.randint(0, IMG_SIZE, (5, 2))
        for i in range(len(points) - 1):
            cv2.line(img, tuple(points[i]), tuple(points[i+1]), (0), 2)

    return img

# Create dataset
for person_id in range(1, 6):  # 5 persons
    for type_ in ["genuine", "forged"]:
        path = os.path.join(BASE, f"person_{person_id:03d}", type_)
        os.makedirs(path, exist_ok=True)

        for i in range(12):  # 12 images per type
            if type_ == "genuine":
                img = create_signature(person_id)
            else:
                img = create_signature(person_id + i + 100)

            cv2.imwrite(os.path.join(path, f"{i}.png"), img)

print("✅ Dummy dataset created successfully")
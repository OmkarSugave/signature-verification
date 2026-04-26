import numpy as np
import cv2
import os

IMG_SIZE = 105

def preprocess(path):
    img = cv2.imread(path, 0)

    if img is None:
        return None

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img

def generate_pairs(data_path):
    pairs = []
    labels = []

    users = os.listdir(data_path)

    for user in users:
        g_path = os.path.join(data_path, user, "genuine")
        f_path = os.path.join(data_path, user, "forged")

        # Skip if folders missing
        if not os.path.exists(g_path) or not os.path.exists(f_path):
            continue

        g_imgs = os.listdir(g_path)
        f_imgs = os.listdir(f_path)

        # Need minimum images
        if len(g_imgs) < 2 or len(f_imgs) < 1:
            continue

        # ✅ Positive pairs (ALL combinations)
        for i in range(len(g_imgs)):
            for j in range(i + 1, len(g_imgs)):
                img1 = preprocess(os.path.join(g_path, g_imgs[i]))
                img2 = preprocess(os.path.join(g_path, g_imgs[j]))

                if img1 is not None and img2 is not None:
                    pairs.append([img1, img2])
                    labels.append(1)

        # ✅ Negative pairs (ALL combinations)
        for i in range(len(g_imgs)):
            for j in range(len(f_imgs)):
                img1 = preprocess(os.path.join(g_path, g_imgs[i]))
                img2 = preprocess(os.path.join(f_path, f_imgs[j]))

                if img1 is not None and img2 is not None:
                    pairs.append([img1, img2])
                    labels.append(0)

    return np.array(pairs), np.array(labels)

# Generate and save
train_pairs, train_labels = generate_pairs("dataset/train")
test_pairs, test_labels = generate_pairs("dataset/test")

# Debug prints (IMPORTANT)
print("Train pairs:", len(train_pairs))
print("Test pairs:", len(test_pairs))

if len(train_pairs) == 0:
    raise ValueError("❌ No training data found. Fix dataset structure.")

os.makedirs("pairs", exist_ok=True)

np.save("pairs/train_pairs.npy", train_pairs)
np.save("pairs/train_labels.npy", train_labels)

np.save("pairs/test_pairs.npy", test_pairs)
np.save("pairs/test_labels.npy", test_labels)

print("✅ Pairs generated successfully")
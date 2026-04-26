import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

IMG_SIZE = 105

# load model safely
model = load_model("model/siamese_model.keras", compile=False)

# load test data
pairs = np.load("pairs/test_pairs.npy")
labels = np.load("pairs/test_labels.npy")

pairs = pairs.reshape(-1, 2, IMG_SIZE, IMG_SIZE, 1)

x1 = pairs[:, 0]
x2 = pairs[:, 1]

print("Test samples:", len(x1))

if len(x1) == 0:
    raise ValueError("❌ No test data found")

# predictions
pred = model.predict([x1, x2])
pred_labels = (pred > 0.5).astype(int)

# metrics
print("\n===== RESULTS =====")

print("\nConfusion Matrix:")
print(confusion_matrix(labels, pred_labels))

print("\nAccuracy:", accuracy_score(labels, pred_labels))
print("Precision:", precision_score(labels, pred_labels))
print("Recall:", recall_score(labels, pred_labels))
print("F1 Score:", f1_score(labels, pred_labels))

print("\nDetailed Report:")
print(classification_report(labels, pred_labels))
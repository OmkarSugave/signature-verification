import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

IMG_SIZE = 105

def build_model():
    inp = layers.Input((IMG_SIZE, IMG_SIZE, 1))

    x = layers.Conv2D(32, (3,3), activation='relu')(inp)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)

    return Model(inp, x)

# load data
pairs = np.load("pairs/train_pairs.npy")
labels = np.load("pairs/train_labels.npy")

pairs = pairs.reshape(-1, 2, IMG_SIZE, IMG_SIZE, 1)

x1 = pairs[:, 0]
x2 = pairs[:, 1]

print("Training samples:", len(x1))

# build model
base = build_model()

input_a = layers.Input((IMG_SIZE, IMG_SIZE, 1))
input_b = layers.Input((IMG_SIZE, IMG_SIZE, 1))

feat_a = base(input_a)
feat_b = base(input_b)

# 🔥 SAFE difference (NO lambda)
diff = layers.Subtract()([feat_a, feat_b])
diff = layers.Activation("relu")(diff)

# classification head
x = layers.Dense(64, activation="relu")(diff)
x = layers.Dense(32, activation="relu")(x)
out = layers.Dense(1, activation="sigmoid")(x)

model = Model([input_a, input_b], out)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit([x1, x2], labels, epochs=10, batch_size=16)

model.save("model/siamese_model.keras")

print("✅ Model retrained and saved")
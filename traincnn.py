import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Reshape

heatmaps = np.load("yyz_heatmaps_augmented.npy")

print("Dataset loaded")
print("Dataset shape:", heatmaps.shape)



X = heatmaps[..., np.newaxis]

X = X[:-1]
Y = heatmaps[1:]

Y = Y[..., np.newaxis]

print("Training samples:", X.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

model = Sequential([

    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,1)),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Flatten(),

    Dense(512, activation='relu'),

    Dense(64*64),
    Reshape((64,64,1))
])

model.summary()


model.compile(
    optimizer='adam',
    loss='mse'
)

model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# -----------------------------
# Test prediction
# -----------------------------

prediction = model.predict(X_test[:1])

# -----------------------------
# Visualization
# -----------------------------

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Current Traffic")
plt.imshow(X_test[0].squeeze(), cmap="hot")

plt.subplot(1,3,2)
plt.title("Actual Next Traffic")
plt.imshow(y_test[0].squeeze(), cmap="hot")

plt.subplot(1,3,3)
plt.title("Predicted Next Traffic")
plt.imshow(prediction[0].squeeze(), cmap="hot")

plt.show()
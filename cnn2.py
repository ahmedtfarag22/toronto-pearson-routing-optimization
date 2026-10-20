import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D

# -----------------------------
# load dataset
# -----------------------------

X = np.load("clean_current.npy")
Y = np.load("clean_next.npy")

print("Dataset loaded")
print("X shape:", X.shape)
print("Y shape:", Y.shape)

# -----------------------------
# train test split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42
)

print("Train:", X_train.shape)
print("Test:", X_test.shape)

# -----------------------------
# CNN model
# -----------------------------

model = Sequential([

    Conv2D(32,(3,3),activation='relu',padding='same',input_shape=(64,64,1)),
    Conv2D(32,(3,3),activation='relu',padding='same'),
    Conv2D(64,(3,3),activation='relu',padding='same'),
    Conv2D(1,(3,3),activation='relu',padding='same')

])

model.compile(
    optimizer='adam',
    loss='mse'
)

model.summary()



model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test,y_test)
)


prediction = model.predict(X_test[:1])


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
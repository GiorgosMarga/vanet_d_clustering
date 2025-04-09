import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
# from google.colab import drive
import sys

def load_weights(model):

    if not model.built:
        dummy_input = np.zeros((10, 1, 1))  # Example input shape, adjust as needed
        model.predict(dummy_input)

    for i, layer in enumerate(model.layers):
        expected_shapes = [w.shape for w in layer.get_weights()]
        loaded_weights = []
        for j, expected_shape in enumerate(expected_shapes):
            filepath = f"weights/layer_{i}_weight_{j}.txt"
            weight = np.loadtxt(filepath)

            # Reshape if needed
            if weight.shape != expected_shape:
                try:
                    weight = weight.reshape(expected_shape)
                except Exception as e:
                    raise ValueError(f"Error reshaping {filepath} from {weight.shape} to {expected_shape}: {e}")

            loaded_weights.append(weight)

        # Set weights to the layer
        if loaded_weights:
            layer.set_weights(loaded_weights)
    print(loaded_weights)

def save_weights(model):
    for i, layer in enumerate(model.layers):
        weights = layer.get_weights()
        for j, weight_array in enumerate(weights):
            np.savetxt(f'weights/layer_{i}_weight_{j}.txt', weight_array)

# Prepare the dataset
def create_dataset(data, time_steps=10):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps])
    return np.array(X), np.array(y)

df = pd.read_csv("merged_dataset.csv")
df = df.dropna(subset=["TEMP"])  # Removes rows with NaN values

temp_data = df["TEMP"].values

print(len(temp_data) // 100)
temp_data = temp_data[:len(temp_data) // 100]
print(temp_data[:5])

# Normalize the data
scaler = MinMaxScaler()
temp_data = scaler.fit_transform(temp_data.reshape(-1, 1)).flatten()

# Set time step
time_steps = 10
X, y = create_dataset(temp_data, time_steps)
X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape for GRU input

# Split into training and testing sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = Sequential([
    GRU(16, activation='tanh', return_sequences=False),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
# Compile the model

# decide whether to use weights from file
if sys.argv[1] != "start":
    load_weights(model)
given_epochs = int(sys.argv[2])


early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
model.fit(X_train, y_train, epochs=given_epochs, batch_size=16, validation_data=(X_test, y_test), callbacks=[early_stopping])

# save weights to file for next round
save_weights(model)

#### Evaluation ####
y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
y_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

# # Plot results
# plt.figure(figsize=(10, 5))
# plt.plot(y_actual, label='Actual Temperature')
# plt.plot(y_pred, label='Predicted Temperature')
# plt.legend()
# plt.xlabel("Time")
# plt.ylabel("Temperature")
# plt.title("Temperature Prediction using GRU")
# plt.show()

# R^2 evaluation
r2 = r2_score(y_actual, y_pred)
print("R² Score:", r2)

X_test[0]

X_new_test = np.array([[x] for x in range(10)])
X_new_test

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # remove warning because they are annoying 

import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Parameters
hidden_size = 4  # GRU hidden state size
output_size = 1   # Output dimension
batch_size = 4   # Number of sequences in a batch
train_size = 0.8

class GRU():
    def __init__(self,X,Y, id):
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        print(f"Start {id}")
        self.id = id
        X = np.asarray(X)
        self.input_shape = X[0].shape
        Y = np.asarray(Y)
        x_shape = X.shape
        X = self.scaler_x.fit_transform(X.reshape(-1, 1)).reshape(x_shape)
        Y = self.scaler_y.fit_transform(Y.reshape(-1, 1)).reshape(Y.shape)
        self.X = X
        self.Y = Y
        self.x_train = X[:int(len(X) * train_size)]
        self.y_train = Y[:int(len(Y) * train_size)]
        self.x_test = X[int(len(X) * train_size):]
        self.y_test = Y[int(len(Y) * train_size):]
        self.model = tf.keras.Sequential([
            tf.keras.layers.GRU(hidden_size, return_sequences=False),
            tf.keras.layers.Dense(output_size)
        ])
        self.model.build(input_shape=(batch_size, x_shape[1], x_shape[2]))

    def get_weights(self):
        print(f"{self.id}: Get Weights")
        weights = []
        for weight in self.model.get_weights():
          if len(weight.shape) == 1:
            weight = weight.reshape(-1,1) 
          weights.append(weight.tolist())
        return weights

    def set_weights(self, weights):
        print(f"{self.id}: Set Weights")

        model_weights = []
        for weight in weights:
            model_weight = np.array(weight)
            if model_weight.shape == (1,1):
                model_weight = model_weight.reshape(1,)
            model_weights.append(model_weight)
        self.model.set_weights(model_weights)

    def train(self,epochs=10,batch_size=10):
        print(f"{self.id}: Train")

        self.model.compile(optimizer='adam', loss='mse')

        self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    def predict(self, X):
        print(f"{self.id}: Predict")

        X = np.array(X)
        n, m, k = X.shape
        X = self.scaler_x.transform(X.reshape(-1, 1)).reshape(n,k,m)
        output = self.model.predict(X)
        return self.scaler_y.inverse_transform(output)

    def evaluate(self):
        print(f"{self.id}: Evaluate")
        result = self.model.evaluate(self.x_test, self.y_test)
        return result



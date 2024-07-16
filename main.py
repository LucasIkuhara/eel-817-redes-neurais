# %%
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras import Input
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from keras.api.models import Sequential
from keras.api.layers import Dense
from keras import activations
from keras.losses import MeanSquaredError
from tensorflow import math
import pandas as pd

# %%

housing = fetch_california_housing()
data_x = housing.data
data_y = housing.target

x_train, x_test, y_train, y_test = train_test_split(
    data_x, data_y, test_size=0.33, random_state=42)

# %%
# Model creation with parametrized activations
def make_model(activation: callable) -> Sequential:
    model = Sequential([
        Dense(8, activation=activation),
        Dense(8, activation=activation),
        Dense(1)
    ])

    model.compile("adam", loss=MeanSquaredError)
    return model

# %%
# Custom activation functions
def capped_relu(x: tf.Tensor):
    return math.maximum(math.minimum(x, 1.0), 0.0)

def quadratic_relu(x: tf.Tensor):
    return math.maximum(math.minimum(x**2, 1.0), 0.0)

def quadratic_relu(x: tf.Tensor):
    return math.maximum(math.minimum(x**2, 1.0), 0.0)

# %%
# Compare activations
results = []

for epoch in np.linspace(100, 1000, 3, dtype=np.int32):

    # # ReLu
    # relu = make_model(activations.relu)
    # relu.fit(x, y, batch_size=200, epochs=300)
    # score = relu.evaluate(x_test, y_test)
    # results.append(["relu", epoch, score])

    # # tanh
    # tanh = make_model(activations.tanh)
    # tanh.fit(data_x, data_y, batch_size=200, epochs=epoch)
    # score = tanh.evaluate(x_test, y_test)
    # results.append(["tanh", epoch, score])

    # # sigmoid
    # sigmoid = make_model(activations.sigmoid)
    # sigmoid.fit(data_x, data_y, batch_size=200, epochs=epoch)
    # score = sigmoid.evaluate(x_test, y_test)
    # results.append(["sigmoid", epoch, score])

    # capped relu
    # reExp = make_model(capped_relu)
    # reExp.fit(data_x, data_y, batch_size=200, epochs=300)

    # rectified Exponential (reExp)
    quadratic = make_model(quadratic_relu)
    quadratic.fit(data_x, data_y, batch_size=200, epochs=epoch)
    score = quadratic.evaluate(x_test, y_test)
    results.append(["quadratic", epoch, score])
    # break

# %%
# Create results df
score_df = pd.DataFrame(results, columns=("Activation", "Epoch", "RMSE"))
score_df.head()

# %%
score_df.plot.line(x="Epoch", y="RMSE")

# %%

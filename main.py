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
from math import e
import pandas as pd
import plotly.express as px

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
IN_LIMIT = 5
OUT_LIMIT = 20

def limit(x):
    return math.maximum(math.minimum(x, IN_LIMIT), -IN_LIMIT)

def capped_relu(x: tf.Tensor):
    return math.maximum(math.minimum(x, 1.0), 0.0)

def quadratic_relu(x: tf.Tensor):
    return math.minimum((math.maximum(limit(x), 0.0)/x)*x**2.0, OUT_LIMIT)

def exp_relu(x: tf.Tensor):
    return math.minimum(math.exp(limit(x) - 1.0), OUT_LIMIT)

# %%
# Compare activations
results = []
MAX_EPOCHS = 3

for epoch in range(1, MAX_EPOCHS + 1):

    # ReLu
    relu = make_model(activations.relu)
    relu.fit(data_x, data_y, batch_size=200, epochs=epoch)
    score = relu.evaluate(x_test, y_test)
    results.append(["relu", epoch, score])

    # tanh
    tanh = make_model(activations.tanh)
    tanh.fit(data_x, data_y, batch_size=200, epochs=epoch)
    score = tanh.evaluate(x_test, y_test)
    results.append(["tanh", epoch, score])

    # sigmoid
    sigmoid = make_model(activations.sigmoid)
    sigmoid.fit(data_x, data_y, batch_size=200, epochs=epoch)
    score = sigmoid.evaluate(x_test, y_test)
    results.append(["sigmoid", epoch, score])

    # capped relu
    capped = make_model(capped_relu)
    capped.fit(data_x, data_y, batch_size=200, epochs=epoch)
    score = capped.evaluate(x_test, y_test)
    results.append(["capped", epoch, score])

    # rectified quadratic
    quadratic = make_model(quadratic_relu)
    quadratic.fit(data_x, data_y, batch_size=200, epochs=epoch)
    score = quadratic.evaluate(x_test, y_test)
    results.append(["quadratic", epoch, score])

    # rectified Exponential (reExp)
    exp = make_model(exp_relu)
    exp.fit(data_x, data_y, batch_size=200, epochs=epoch)
    score = exp.evaluate(x_test, y_test)
    results.append(["exponential", epoch, score])

# %%
# Create results df
score_df = pd.DataFrame(results, columns=("Activation", "Epoch", "RMSE"))
score_df.to_csv("activations.csv")
score_df.head(6)

# %%
fig = px.line(score_df, x="Epoch", y="RMSE", color="Activation")

# %%

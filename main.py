# %%
import tensorflow as tf
from tensorflow import keras
from keras import Input
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from keras.api.models import Sequential
from keras.api.layers import Dense
from keras import activations
from keras.losses import MeanSquaredError
from tensorflow import math

# %%

housing = fetch_california_housing()
x = housing.data
y = housing.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42)

# %%
# Model creation with parametrized activations
def make_model(activation: callable) -> Sequential:
    model = Sequential([
        # Input(shape=(None, 8)),
        Dense(8, activation=activation),
        Dense(8, activation=activation),
        Dense(1)
    ])

    model.compile("adam", loss=MeanSquaredError)
    return model

# %%
# Custom activation functions
def reExp(x: tf.float32):
    x
    return tf.maximum(0, x)

# %%
# Compare activations

# ReLu
# relu = make_model(activations.relu)
# relu.fit(x, y, batch_size=200, epochs=300)

# rectified Exponential (reExp)
reExp = make_model(reExp)
reExp.fit(x, y, batch_size=200, epochs=300)

# %%

# %%
import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
from keras import Input
from sklearn.datasets import fetch_california_housing
from keras.api.models import Sequential
from keras.api.layers import Dense
from tensorflow import math

# %%

housing = fetch_california_housing()
x = data

# %%

ds_train = tf.keras.preprocessing.timeseries_dataset_from_array(
     data=df_train[['High', 'Low', 'Volume', 'selic', 'oil_price', 'exchange_rate', 'ipca', 'bovespa']],
     targets=df_train['Open'],
     sequence_length=10)

ds_test = tf.keras.preprocessing.timeseries_dataset_from_array(
     data=df_test[['High', 'Low', 'Volume', 'selic', 'oil_price', 'exchange_rate', 'ipca', 'bovespa']],
     targets=df_test['Open'],
     sequence_length=10)

# Model
def activatea(x):
    return math.maximum(x, 0)

model = Sequential([
    Input(shape=(10, 8)),
    Dense(24, activation=activatea),
    Dense(1)
])

model.compile(loss='mean_squared_error', optimizer='adam')


# %%
print(ds_train)

model.fit(ds_train, epochs=1000)
predictions = model.predict(ds_test)

print

# %%

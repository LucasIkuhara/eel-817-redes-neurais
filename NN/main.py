import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Input
from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense, Dropout
from keras.api.callbacks import EarlyStopping

df = pd.read_csv('./data/petr_data.csv', sep=',', usecols=['Open', 'High', 'Low', 'Volume', 'selic', 'oil_price', 'exchange_rate', 'ipca', 'bovespa'])

#print(df[0:10])

df_train = df[:670]
df_test = df[670:]

ds_train = tf.keras.preprocessing.timeseries_dataset_from_array(
     data=df_train[['High', 'Low', 'Volume', 'selic', 'oil_price', 'exchange_rate', 'ipca', 'bovespa']],
     targets=df_train['Open'],
     sequence_length=10)

ds_test = tf.keras.preprocessing.timeseries_dataset_from_array(
     data=df_test[['High', 'Low', 'Volume', 'selic', 'oil_price', 'exchange_rate', 'ipca', 'bovespa']],
     targets=df_test['Open'],
     sequence_length=10)

# Model

model = Sequential([
    Input(shape=(10, 8)),
    LSTM(256, dropout=0.2, recurrent_dropout=0.2,),
    Dense(1)
])
model.compile(loss='mean_squared_error', optimizer='adam')
monitor=EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')

print(ds_train)

model.fit(ds_train, epochs=1000)
predictions = model.predict(ds_test)

print

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 11:04:35 2020

@author: rodrigo
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, LSTM

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# Data
series = np.sin((0.1*np.arange(400))**2)

plt.plot(series)
plt.show()

# Prepare the Data
# Use T past values to predict the next value
T = 10
D = 1
X = []
Y = []

for t in range(len(series)-T):
    x = series[t:t+T]
    X.append(x)
    y = series[t+T]
    Y.append(y)

# Data Shape N x T x D
X = np.array(X).reshape(-1, T, 1)
Y = np.array(Y)
N = len(X)

# make the RNN
i = Input(shape=(T, D))
x = LSTM(10)(i)
x = Dense(1)(x)
model = Model(i, x)
model.compile(loss='mse',
              optimizer=Adam(lr=0.05),
              )

# train the RNN
r = model.fit(X[:-N//2],
              Y[:-N//2],
              batch_size=32,
              epochs=200,
              validation_data=(X[-N//2:], Y[-N//2:]),
              )

# Plot loss per iteration
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend(loc='best')


# Forecast future values
# Use only self-predictionsfor making future predictions
validation_target = Y[-N//2:]
validation_predictions = []
# first validation input
last_x = X[-N//2]
while len(validation_predictions) < len(validation_target):
    p = model.predict(last_x.reshape(1, T, 1))[0, 0]
    # update the predictions list
    validation_predictions.append(p)
    # make the new input
    last_x = np.roll(last_x, -1)
    last_x[-1] = p


plt.plot(validation_target, label='forecast target')
plt.plot(validation_predictions, label='forecast prediction')
plt.legend(loc='best')
plt.show()

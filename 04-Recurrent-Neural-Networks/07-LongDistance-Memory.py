# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 11:40:23 2020

@author: rodrigo
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (Input,
                                     SimpleRNN,
                                     LSTM,
                                     GRU,
                                     GlobalAveragePooling1D,
                                     Dense)

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# ## build the dataset ## #
# This is a nonlinear AND
# long-distance/short-distance dataset


def get_label(x, i1, i2, i3):
    # x = sequence
    if x[i1] < 0 and x[i2] < 0 and x[i3] < 0:
        return 1
    if x[i1] < 0 and x[i2] > 0 and x[i3] > 0:
        return 1
    if x[i1] > 0 and x[i2] < 0 and x[i3] > 0:
        return 1
    if x[i1] > 0 and x[i2] > 0 and x[i3] < 0:
        return 1
    return 0


def generate_series(T, i1, i2, i3):
    X = []
    Y = []
    for t in range(5000):
        x = np.random.randn(T)
        X.append(x)
        y = get_label(x, i1, i2, i3)
        Y.append(y)

    X = np.array(X)
    Y = np.array(Y)
    N = len(X)

    return N, X, Y


def rnn_model_fit(
    T, D, X, method,
    h, optimizer=Adam(lr=0.01), loss='binary_crossentropy', epochs=200
):
    inputs = np.expand_dims(X, -1)
    i = Input(shape=(T, D))

    if method == 'simple':
        x = SimpleRNN(h)(i)
    elif method == 'gru':
        x = GRU(h)(i)
    elif method == 'lstm':
        x = LSTM(h)(i)
    elif method == 'lstm_max_pooling':
        '''
        return_sequences = False
            * Input: T x D
            * After RNN Unit h(T):M
            * Output: K

        return_sequences = True
            * Input: T x D
            * After RNN Unit h(1), h(2), ..., h(T): T x M
            * After global max pooling max{h(1), h(2), ..., h(T)}: M
            * Output: K
        '''
        x = LSTM(units=h, return_sequences=True)(i)
        x = GlobalAveragePooling1D()(x)
    else:
        print("Method not found")

    x = Dense(1, activation='sigmoid')(x)
    model = Model(i, x)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'],
                  )
    callback = EarlyStopping(monitor='val_loss', patience=10)

    r = model.fit(inputs,
                  Y,
                  epochs=epochs,
                  validation_split=0.5,
                  callbacks=[callback])
    return r


def plot_history(r):
    fig, axs = plt.subplots(2, sharex=True)
    # Plot the loss
    axs[0].set_title('Loss')
    axs[0].plot(r.history['loss'], label='loss')
    axs[0].plot(r.history['val_loss'], label='val_loss')
    axs[0].legend()
    # Plot the accuracy
    axs[1].set_title('Accuracy')
    axs[1].plot(r.history['accuracy'], label='acc')
    axs[1].plot(r.history['val_accuracy'], label='val_acc')
    axs[1].legend()


# Try a simple RNN with short distance memory
T = 10
D = 1
h = 5  # hidden layers
N, X, Y = generate_series(T, -1, -2, -3)  # short distance
r = rnn_model_fit(T, D, X, 'simple', h)
plot_history(r)


# Try a simple RNN with long distance memory
T = 10
D = 1
h = 5  # hidden layers
N, X, Y = generate_series(T, 0, 1, 2)  # long distance
r = rnn_model_fit(T, D, X, 'simple', h)
plot_history(r)


# Test LSTM with long distance memory
T = 10
D = 1
h = 5  # hidden layers
N, X, Y = generate_series(T, 0, 1, 2)  # long distance
r = rnn_model_fit(T, D, X, 'lstm', h)
plot_history(r)


# Make the problem harder by making T larger  T = 20

# Try a simple RNN with long distance memory
T = 20
D = 1
h = 5  # hidden layers
N, X, Y = generate_series(T, 0, 1, 2)  # long distance
r = rnn_model_fit(T, D, X, 'simple', h)
plot_history(r)

# Test LSTM with long distance memory
T = 20
D = 1
h = 5  # hidden layers
N, X, Y = generate_series(T, 0, 1, 2)  # long distance
r = rnn_model_fit(T, D, X, 'lstm', h)
plot_history(r)

# Test GRU with long distance memory
T = 20
D = 1
h = 5  # hidden layers
N, X, Y = generate_series(T, 0, 1, 2)  # long distance
r = rnn_model_fit(T, D, X, 'gru', h, epochs=400)
plot_history(r)

# Make the problem harder by making T larger  T = 30

# Test LSTM with long distance memory
T = 30
D = 1
h = 15  # hidden layers
N, X, Y = generate_series(T, 0, 1, 2)  # long distance
r = rnn_model_fit(T, D, X, 'lstm', h, epochs=400)
plot_history(r)

# Test LSTM with Global Max Pooling
T = 30
D = 1
h = 15  # hidden layers
N, X, Y = generate_series(T, 0, 1, 2)  # long distance
r = rnn_model_fit(T, D, X, 'lstm_max_pooling', h, epochs=200)
plot_history(r)

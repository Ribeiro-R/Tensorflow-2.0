# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 10:50:02 2020

@author: rodrigo
"""


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, SimpleRNN, Dense


gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


# N = number of samples
# T = sequence length
# D = number of input features
# M = number of hidden units
# K = number of output units

# Data Shape N x T x D
# Make some data
N = 3
T = 10
D = 3
K = 2
X = np.random.randn(N, T, D)

# Make an RNN
M = 5  # number of hidden units
i = Input(shape=(T, D))
x = SimpleRNN(M, activation='tanh')(i)
x = Dense(K)(x)

model = Model(i, x)

# Get the output
Yhat = model.predict(X)
print(Yhat)

# See if we can replicate this output
# Get the weights first
model.summary()

# See what's returned
model.layers[1].get_weights()

# Check their shapes
# Should make sense
# First output is input > hidden
# Second output is hidden > hidden
# Third output is bias term (vector of length M)
a, b, c = model.layers[1].get_weights()
print(a.shape, b.shape, c.shape)

Wx, Wh, bh = model.layers[1].get_weights()
Wo, bo = model.layers[2].get_weights()


h_last = np.zeros(M)  # initial hidden state
x = X  # sample
Yhats = []  # where we store the outputs

for i in range(len(x)):
    for t in range(T):
        h = np.tanh(x[i][t].dot(Wx) + h_last.dot(Wh) + bh)
        y = h.dot(Wo) + bo
        # important: assign h to h_last
        h_last = h
    Yhats.append(y)
    h_last = np.zeros(M)

# print the final output
print(Yhat)
print(Yhats)

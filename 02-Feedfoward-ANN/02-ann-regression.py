#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 13:22:59 2020

@author: rodrigo
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


# Model Function
def ann():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, input_shape=(2,), activation='relu'),
        tf.keras.layers.Dense(1)])
    return model


# Make the dataset
N = 1000
X = np.random.random((N, 2))*6 - 3
y = np.cos(2*X[:, 0]) + np.tanh(3*X[:, 1])

# Plot of the data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y)
plt.show()

# Instantiate model
model = ann()

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss='mse')

# Train the model
r = model.fit(X, y, epochs=100)

# Plot loss per iteration
plt.plot(r.history['loss'], label='loss')
plt.legend(loc='best')
plt.show()

# Plot the prediction surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y)
# surface plot
line = np.linspace(-3, 3, 50)
xx, yy = np.meshgrid(line, line)
Xgrid = np.vstack((xx.flatten(), yy.flatten())).T
Yhat = model.predict(Xgrid).flatten()
ax.plot_trisurf(Xgrid[:, 0], Xgrid[:, 1], Yhat,
                linewidth=0.2, antialiased=True)
plt.show()

# Can it extrapolate?
# Plot the prediction surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y)
# surface plot
line = np.linspace(-5, 5, 50)
xx, yy = np.meshgrid(line, line)
Xgrid = np.vstack((xx.flatten(), yy.flatten())).T
Yhat = model.predict(Xgrid).flatten()
ax.plot_trisurf(Xgrid[:, 0], Xgrid[:, 1], Yhat,
                linewidth=0.2, antialiased=True)
plt.show()

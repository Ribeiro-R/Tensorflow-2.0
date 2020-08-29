# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 21:18:27 2020

@author: rodrigo
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


# Learning rate scheduler
def schedule(epoch, lr):
    if epoch >= 50:
        return 0.0001
    return 0.001


# Load the data
data = pd.read_csv('./data/moore.csv', header=None)
data = data.rename(columns={0: "year", 1: "n_transistors"})

# Make 2D array of size NxD where D=1
X = data['year'].values.reshape(-1, 1)
y = data['n_transistors'].values

# Plot the points
plt.scatter(X, y)
plt.show()

# Take the log
y = np.log(y)

# Plot the points
plt.scatter(X, y)
plt.show()

# Center de X data so the values are not too large
X = X-X.mean()

# Create Tensorflow model
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(1)
    ])

model.compile(optimizer=tf.keras.optimizers.SGD(0.001, 0.9),
              loss='mse')

scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)

# Train the model
r = model.fit(X, y, epochs=200, callbacks=[scheduler])

# Plot the loss
plt.plot(r.history['loss'], label='Loss')
plt.legend()
plt.plot()

# Get the slope of the line
# The slope is relates to the doubling rate of transistor count
# Note: there is only 1 layer, the "Input" layer does not count
print(model.layers)
print(model.layers[0].get_weights())

# The slope of the line
a = model.layers[0].get_weights()[0][0, 0]
print("TIme to double: {}".format(np.log(2)/a))

# Make sure the line fits our data
yhat = model.predict(X).flatten()
plt.scater(X, y)
plt.plot(X, yhat)

# Manual Calculation
# Get the Weights
w, b = model.layers[0].get_weights()
# Reshape X
X = X.reshape(-1, 1)
# (Nx1)x(1x1)+(1)-->(Nx1)
yhat2 = (X.dot(w)+b).flatten()
print(np.allclose(yhat, ythat2))

# Save model to a file
model.save("./01-Machine-Learning-and-Neurons/models/linearregression.hs")

# Load the model
model = tf.keras.models.load_model("./01-Machine-Learning-and-Neurons\
                                   /models/linearregression.hs")
print(model.layers)
print(model.evaluate(X_test, y_test))

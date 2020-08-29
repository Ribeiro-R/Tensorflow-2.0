# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 17:10:02 2020

@author: rodrigo
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Load the data
data = load_breast_cancer()

# Check data
print(type(data))
print(data.keys())
print(data.data.shape)
print(data.target.shape)
print(data.target_names)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data.data,
                                                    data.target,
                                                    test_size=0.33,
                                                    random_state=42)
N, D = X_train.shape
print("N = {}".format(N))
print("D = {}".format(D))

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(D,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
r = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100)

# Evaluete the model - evaluete() returns loss and accuracy
print("Train score: {}".format(model.evaluate(X_train, y_train)))
print("Test score: {}".format(model.evaluate(X_test, y_test)))

# Plot loss and val_loss from model.fit()
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend(loc='best')
plt.show()

# Plot the accuracy
plt.plot(r.history['accuracy'], label='accuracy')
plt.plot(r.history['val_accuracy'], label='val_accuracy')
plt.legend(loc='best')
plt.show()

# Make predictions
# They are outputs of the sigmoid, interpreted as propabilities p(y=1|x)
P = model.predict(X_test)
print("{}".format(P))

# To get the predictions: P has to be flattened since the targets are size
# (N,) while the predictions are size (N,1)
P = np.round(P).flatten()

# Calculate the accuracy, compare it to evaluete() output
print("Manually calculated accuracy: {}".format(np.mean(P == y_test)))
print("Evaluate output: {}".format(model.evaluate(X_test, y_test)))

# Save model to a file
model.save("./01-Machine-Learning-and-Neurons/models/linearclassifier.hs")

# Load the model
model = tf.keras.models.load_model("./01-Machine-Learning-and-Neurons\
                                   /models/linearclassifier.hs")
print(model.layers)
print(model.evaluate(X_test, y_test))

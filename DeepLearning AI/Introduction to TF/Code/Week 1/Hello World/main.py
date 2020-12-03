# --------------------------------------------------------------------------- #
# Importing required libraries including tensorflow, keras and numpy
import tensorflow as tf
from tensorflow import keras
import numpy as np
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Define and compile the network
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
# Create layers of neurons 
model.compile(optimizer='sgd', loss='mean_squared_error')
# This measures how bad or how good the model performs
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Providing the data
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
# Data for x and data for y
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Training the data

model.fit(xs, ys, epochs=500)
# Testing hte model with 500 epocs (500 times)

print(model.predict([10.0]))
# predict the model with x of 10
# --------------------------------------------------------------------------- #


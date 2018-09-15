#!/usr/bin/env python3

"""
This is an annotated (and slightly modified) version of the
MNIST example from https://www.tensorflow.org/tutorials/.
"""

import tensorflow as tf
import numpy as np

# Load a local copy of the MNIST data set. 
# You can download this from 
# https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
# An npz file is a saved set of numpy arrays.
# This file has four named files (arrays): x_train, y_train, x_test, y_test
mnist_data = np.load('./mnist.npz')

# x_train is the set of training inputs.
# This array consists of 60,000 images. 
# Each image is 28x28 two-dimensional arrays of pixel data.
# Pixel data is represented as a value between 0 and 254 inclusive.
x_train = mnist_data['x_train']
# y_train is the set of training outputs.
# This array consists of 60,000 integers.
# Each integer is a value between 0 and 9, inclusive.
# The integer should represent the digit depicted in the input image.
y_train = mnist_data['y_train']
# x_test is the set of testing inputs.
# This array contains 10,000 images in the same format as x_train.
x_test = mnist_data['x_test']
# y_test is the set of testing outputs.
# This array contains 10,000 integers in the same format as y_train.
y_test = mnist_data['y_test']

# Convert our input pixel data to values between 0.0 and 1.0.
x_train = x_train / 255.0
x_test = x_test / 255.0

# A Sequential model is a simple stack of NN layers.
# Models are mutable objects which contain all the NN values.
# Later, we'll train this model on our data.
model = tf.keras.models.Sequential([
  # Flattens the input.
  # Specifically, this means we take our two-dimensional inputs 
  # (28x28 pixel grids) and flatten them into a one-dimensional vector 
  # of 784 pixel values.
  tf.keras.layers.Flatten(),
  # Add a densly connected layer of 512 neurons with a ReLU 
  # (rectified linear unit) activation.
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  # Add a dropout layer that randomly drops 20% of inputs during training.
  # Dropouts like this help prevent overfitting.
  tf.keras.layers.Dropout(0.2),
  # Add our output layer of 10 densly-connected nodes.
  # Each output node represents a possible output digit (0 - 9)
  # Softmax is used to ensure all output values add up to 1.0.
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Configure the model for training.
# Use an Adam optimizer - an extension to stochastic gradient decent
# that is often prefered for computer vision and NLP processing.
# For the loss function, we use 'sparse_categorical_crossentropy'.
# The sparse is important here as our outputs are integers (0-9).
# If our outputs were "one-hot encodings" instead (e.g., [0, 0, 1, ..., 0]),
# we would want to just use 'categorical_crossentropy' (sans 'sparse_').
# Finally, we instruct the model to evaluate the accuracy metric during 
# training. 
# This metric will be printed out to the console at the end of each epoch. 
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Train the model.
# At this stage, we simply need to provide the training inputs and outputs.
# We also specify that training should last for five epochs.
# When training completes, the model will hold the established weights.
model.fit(x_train, y_train, epochs=5)
# Finally, test the model.
# Here we supply the distinct test dataset for evaluation.
# This returns an array of values - the test loss and any metrics recorded
# in the model ('accuracy', in our case).
result = model.evaluate(x_test, y_test)
print(result)


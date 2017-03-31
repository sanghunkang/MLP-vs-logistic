#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Author		: ...
# Contact		: ...
# Started on	: 20170310(yyyymmdd)
# Last modified : 20170321(yyyymmdd)
# Project		: ...
#############################################################################
# Import packages
import numpy as np
import tensorflow as tf

from numpy import genfromtxt

# BUILDING THE COMPUTATIONAL GRAPH
# Import data
data_raw = genfromtxt(".\\data\\health_test_selected.csv", delimiter=',')
data = data_raw[~np.isnan(data_raw).any(axis=1)]
print(len(data))

# Shuffle the data, divide into training and test sets
np.random.shuffle(data)
data_training = data[:int(len(data)/2)]
data_test = data[int(len(data)/2):]

data_training_input = data_training[:,0:23]
data_training_output = data_training[:,23]

len_vec_input = data_training_input.shape[1]
len_vec_output = 2
# len_vec_input = 784
# len_vec_output = 10

# Model parameters
params = {
	"W_h1": tf.Variable(tf.random_normal([len_vec_input, 100])),
	"W_h2": tf.Variable(tf.random_normal([100, 100])),
	"W_out": tf.Variable(tf.random_normal([100, len_vec_output])),
	"b_h1": tf.Variable(tf.random_normal([100])),
	"b_h2": tf.Variable(tf.random_normal([100])),
	"b_out": tf.Variable(tf.random_normal([len_vec_output])),
}
# Define the model
def model(x, params):
	# Hidden layer 1
	layer_h1 = tf.add(tf.matmul(x, params["W_h1"]), params["b_h1"])
	layer_h1 = tf.nn.sigmoid(layer_h1)

	# Hidden layer 2
	layer_h2 = tf.add(tf.matmul(layer_h1, params["W_h2"]), params["b_h2"])
	layer_h2 = tf.nn.sigmoid(layer_h2)

	# Output layer
	layer_output = tf.add(tf.matmul(layer_h2, params["W_out"]), params["b_out"])
	return layer_output

# Input, output, model
x = tf.placeholder(tf.float32, [None, len_vec_input])
y_true = tf.placeholder(tf.float32, [None, len_vec_output])
y_pred = model(x, params)

# Cost function
costs = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)
cost = tf.reduce_mean(costs)

# Optimiser
optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(cost) # the final node which inherits all nodes


# RUNNING THE COMPUTATIONAL GRAPH


# Training parameters
batch_size = 1000

# Training loop
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(50):
	# batch_x, batch_y = mnist.train.next_batch(100)
	batch_x = data_training_input[i*batch_size:(i+1)*batch_size,]
	batch_y = data_training_output[i*batch_size:(i+1)*batch_size,]
	batch_y = np.c_[batch_y, 1 - batch_y]
	_, c = sess.run([train, cost], feed_dict = {x: batch_x, y_true: batch_y})
	print(c)


# Write the summary
logs_path = ".\\tensorboard"
merged = tf.summary.merge_all()
tf.summary.FileWriter(logs_path, sess.graph)


# Evaluate training accuracy
correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(y_true,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict = {x: batch_x, y_true: batch_y}))




#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Author        : ...
# Contact       : ...
# Started on    : 20170310(yyyymmdd)
# Last modified : 20170410(yyyymmdd)
# Project       : ...
#############################################################################
import os

# Import packages
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from numpy import genfromtxt

# Import custom modules
from customhelpers.customhelpers import encode_onehot
from model_body import model_mlp
from params import params

# BUILDING THE COMPUTATIONAL GRAPH
# Import data
path_data = os.path.dirname(os.getcwd()) + "\\data\\KUMC\\health_test_selected.csv"
data_raw = genfromtxt(path_data, delimiter=',')
data = data_raw[~np.isnan(data_raw).any(axis=1)]
print(data.shape)
data_X = data[:,0:23]
data = np.c_[data_X, encode_onehot(data[:,-1], 2, zero_columns=False)]

# Shuffle the data, divide into training and test sets
np.random.shuffle(data)
data_training = data[:9*int(len(data)/10)]
data_test = data[9*int(len(data)/10):]
len_X = 23

# Learning Parameters
learning_rate = 0.0005
batch_size = 2000
display_step = 1
training_epochs = 100

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 23 # MNIST data input (img shape: 28*28)
n_classes = 2 # MNIST total classes (0-9 digits)

# Model parameters, which will be saved
params = {
	'W_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
	'W_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
	'W_out': tf.Variable(tf.random_normal([n_hidden_2, n_classes])),
	'b_h1': tf.Variable(tf.random_normal([n_hidden_1])),
	'b_h2': tf.Variable(tf.random_normal([n_hidden_2])),
	'b_out': tf.Variable(tf.random_normal([n_classes]))
}

data_saved = {'var_epoch_saved': tf.Variable(0)}

# Set model
def model_mlp(x, params):
	# Hidden layer with RELU activation
	layer_1 = tf.add(tf.matmul(x, params['W_h1']), params['b_h1'])
	layer_1 = tf.nn.sigmoid(layer_1)
	# Hidden layer with RELU activation
	layer_2 = tf.add(tf.matmul(layer_1, params['W_h2']), params['b_h2'])
	layer_2 = tf.nn.sigmoid(layer_2)
	# Output layer with linear activation
	out_layer = tf.add(tf.matmul(layer_2, params['W_out']), params['b_out'])
	return out_layer

# Input, output, prediction from the model
x = tf.placeholder("float", [None, n_input])
y_true = tf.placeholder("float", [None, n_classes])
y_pred = model_mlp(x, params)

# Define loss and optimiser
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)
cost = tf.reduce_mean(entropy)
train = tf.train.AdamOptimizer(learning_rate=learning_rate)
optimizer = train.minimize(cost)

# RUNNING THE COMPUTATIONAL GRAPH
# Define saver 
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
	# Initialise the variables and run
	
	with tf.device("/cpu:0"):
		init = tf.global_variables_initializer()
		sess.run(init)
		# Restore saved model if any
		try:
			saver.restore(sess, ".\\model\\model.ckpt")
			print("Model restored")
			epoch_saved = data_saved['var_epoch_saved'].eval()
		except tf.errors.NotFoundError:
			print("No saved model found")
			epoch_saved = 0
		except tf.errors.InvalidArgumentError:
			print("Model structure has change. Rebuild model")
			epoch_saved = 0

		# Training cycle
		
		epoch_saved = data_saved['var_epoch_saved'].eval()
		print(epoch_saved)
		for epoch in range(epoch_saved, epoch_saved+training_epochs):
			avg_cost = 0.
			total_batch = int(len(data_training)/batch_size)
			
			# Loop over all batches
			for i in range(total_batch):				
				batch = data_training[np.random.choice(data_training.shape[0], size=batch_size,  replace=True)]
				batch_x = batch[:, :len_X]
				batch_y = batch[:, len_X:]

				# Run optimization op (backprop) and cost op (to get loss value)
				_, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y_true: batch_y})
				# Compute average loss
				avg_cost += c / total_batch
			
			# Display logs per epoch step
			if epoch % display_step == 0:
				print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
		print("Optimization Finished!")

		# Test model
		correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
		# Calculate accuracy
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		print("Accuracy:", accuracy.eval({x: data_test[:,:len_X], y_true: data_test[:,len_X:]}))

		# Save the variables
		sess.run(data_saved['var_epoch_saved'].assign(epoch_saved + training_epochs))

		save_path = saver.save(sess, ".\\model\\model.ckpt")
		print("Model saved in file: %s" % save_path)

input("Press any key to quit")
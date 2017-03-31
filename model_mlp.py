#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Author        : ...
# Contact       : ...
# Started on    : 20170310(yyyymmdd)
# Last modified : 20170321(yyyymmdd)
# Project       : ...
#############################################################################
# Import packages
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from numpy import genfromtxt

# BUILDING THE COMPUTATIONAL GRAPH
# Import data
data_raw = genfromtxt(".\\data\\health_test_selected.csv", delimiter=',')
# data = np.nan_to_num(data_raw)
data = data_raw[~np.isnan(data_raw).any(axis=1)]
print(len(data))

# Shuffle the data, divide into training and test sets
np.random.shuffle(data)
data_training = data[:int(len(data)/2)]
data_test = data[int(len(data)/2):]

data_training_input = data_training[:,0:23]
data_training_output = data_training[:,23]

data_test_input = data_test[:,0:23]
data_test_output = data_test[:,23]
len_vec_x =23

# Learning Parameters
learning_rate = 0.001
batch_size = 1000
display_step = 1
training_epochs = 500

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 23 # MNIST data input (img shape: 28*28)
n_classes = 2 # MNIST total classes (0-9 digits)

# Model parameters, which will be saved
params = {
    'var_epoch_saved': tf.Variable(0),
    'W_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'W_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'W_out': tf.Variable(tf.random_normal([n_hidden_2, n_classes])),
    'b_h1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b_h2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b_out': tf.Variable(tf.random_normal([n_classes]))
}

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
    init = tf.global_variables_initializer()
    sess.run(init)
    
    # Restore saved model if any
    try:
        saver.restore(sess, ".\\model\\model.ckpt")
        print("Model restored")
    except tf.errors.NotFoundError:
        print("No saved model found")

    # Training cycle
    epoch_saved = params['var_epoch_saved'].eval()
    print(epoch_saved)
    for epoch in range(epoch_saved, epoch_saved+training_epochs):
        avg_cost = 0.
        total_batch = int(len(data_training)/batch_size)
        
        # Loop over all batches
        for i in range(total_batch):
            batch_x = data_training[i*batch_size:(i+1)*batch_size, 0:len_vec_x]
            batch_y = data_training[i*batch_size:(i+1)*batch_size, 23]
            batch_y = np.c_[batch_y, 1-batch_y]
            
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y_true: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: data_training_input, y_true: np.c_[data_training_output, 1-data_training_output]}))

    # Save the variables
    sess.run(params['var_epoch_saved'].assign(epoch_saved + training_epochs))
    print(params['var_epoch_saved'].eval())
    save_path = saver.save(sess, ".\\model\\model.ckpt")
    print("Model saved in file: %s" % save_path)
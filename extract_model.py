#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Author        : ...
# Contact       : ...
# Started on    : 20170322(yyyymmdd)
# Last modified : 20170327(yyyymmdd)
# Project       : ...
#############################################################################
# Import packages
import _pickle as cPickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# from customhelpers import encode_onehot
from numpy import genfromtxt

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


	# Save the variables
    # sess.run(params['var_epoch_saved'].assign(epoch_saved + training_epochs))
    print(params['W_h1'].eval())
    print(params['W_h2'].eval())
    save_path = saver.save(sess, ".\\model\\model.ckpt")
    print("Model saved in file: %s" % save_path)
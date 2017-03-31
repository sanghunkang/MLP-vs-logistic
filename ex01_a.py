'''
A Multilayer Perceptron implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function
from numpy import genfromtxt

# Import MNIST data
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

data_raw = genfromtxt(".\\data\\health_test_selected.csv", delimiter=',')
data_filtered = data_raw[~np.isnan(data_raw).any(axis=1)]

data = data_filtered

# Shuffling the dataset, in case target values are highly related to the order
# target value의 분포가 고르지 않을 수 있으므로 전체 데이터 순서를 섞어준다.
np.random.shuffle(data)

# Spliting the dataset into training and test datasets
# 자료를 training set, test set의 둘로 나눈다.
data_training = data[:int(len(data)/2)]
data_test = data[int(len(data)/2):]

# Parameters
learning_rate = 0.001
training_epochs = 50
batch_size = 1000
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 23 # MNIST data input (img shape: 28*28)
n_classes = 2 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        # total_batch = int(mnist.train.num_examples/batch_size)
        total_batch = int(len(data_training)/batch_size)

        # Loop over all batches
        for i in range(total_batch):
            # batch_x, batch_y = mnist.train.next_batch(batch_size)
            # 아래의 코드를 대신 사용하면 순서대로, 중복없이 batch를 입력할 수 있다.
            batch_x = data_training[i*batch_size:(i+1)*batch_size, 0:23]
            batch_y = data_training[i*batch_size:(i+1)*batch_size, 23]
            batch_y = np.c_[batch_y, 1-batch_y]
            
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    data_training_x = data_training[:, 0:23]
    data_training_y = data_training[:, 23]
    data_training_y = np.c_[data_training_y, 1-data_training_y]

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy(Training):", accuracy.eval({x: data_training_x, y: data_training_y}))
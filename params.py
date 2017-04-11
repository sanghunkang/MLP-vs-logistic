import tensorflow as tf

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
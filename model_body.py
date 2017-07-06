import tensorflow as tf

# Set model
def model_mlp(x, params):
	# Hidden layer with RELU activation
	layer_1 = tf.add(tf.matmul(x, params['W_h1']), params['b_h1'])
	layer_1 = tf.nn.relu(layer_1)
	# Hidden layer with RELU activation
	layer_2 = tf.add(tf.matmul(layer_1, params['W_h2']), params['b_h2'])
	layer_2 = tf.nn.relu(layer_2)
	# Output layer with linear activation
	out_layer = tf.add(tf.matmul(layer_2, params['W_out']), params['b_out'])
	return out_layer
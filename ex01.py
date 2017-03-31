#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Author		: ...
# Contact		: ...
# Started on	: 20170310(yyyymmdd)
# Last modified : 20170321(yyyymmdd)
# Project		: ...
#############################################################################
from __future__ import print_function
from numpy import genfromtxt

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Reading the data
# csv파일을 읽어옴
data_raw = genfromtxt(".\\data\\health_test_selected.csv", delimiter=',')
# 읽어온 자료에서 NA가 등장하는 모든 행을 제외
data_filtered = data_raw[~np.isnan(data_raw).any(axis=1)]

def scale_linear_bycolumn(rawpoints, high=100.0, low=0.0):
    mins = np.min(rawpoints, axis=0)
    maxs = np.max(rawpoints, axis=0)
    rng = maxs - mins
    return high - (((high - low) * (maxs - rawpoints)) / rng)

data_filtered = scale_linear_bycolumn(data_filtered, high=1.0)
# Undersampling
# data_sanitised_t1_0 = data_sanitised[np.logical_or.reduce([data_sanitised[:,23] == x for x in [0]])]
# data_sanitised_t1_1 = data_sanitised[np.logical_or.reduce([data_sanitised[:,23] == x for x in [1]])]
# max_samplesize = min(len(data_sanitised_t1_0), len(data_sanitised_t1_1))
# data_sanitised_t1_0 = data_sanitised_t1_0[np.random.choice(data_sanitised_t1_0.shape[0], size=max_samplesize, replace=False), :]
# data_sanitised_t1_1 = data_sanitised_t1_1[np.random.choice(data_sanitised_t1_1.shape[0], size=max_samplesize, replace=False), :]
# data = np.concatenate((data_sanitised_t1_0, data_sanitised_t1_1))

# Powering(2) of elemetns
data_p1 = data_filtered[:,0:23]
data_dscrt = data_filtered[:,0:8]

data_cont = data_filtered[:,8:23]
data_cont_p5 = np.power(data_cont, 10)[:,0:23]

# data_cont_normed = (data_cont - data_cont.min(0)) / data_cont.max(0)
# data_cont_normed_p5 = np.power(data_cont_normed, 10)[:,0:23]

# data_cont_std = (data_cont - np.mean(data_cont)) / np.std(data_cont)

# Logarithmisation of elements 
data_1log = np.log(data_filtered+1)[:,0:23]

# Concatenation of modified data
data_input = np.c_[data_dscrt, data_cont_p5]
len_vec_x = data_input.shape[1]

print(len_vec_x)

# Target data
data_input = data_filtered[:,0:23]
data_target = data_filtered[:,23]
# data_target = data[:,24]

# Merging input and target data
data = np.c_[data_input, data_target]

# Shuffling the dataset, in case target values are highly related to the order
# target value의 분포가 고르지 않을 수 있으므로 전체 데이터 순서를 섞음
np.random.shuffle(data)

# Spliting the dataset into training and test datasets
# 자료를 training set, test set의 둘로 나눔
data_training = data[:int(len(data)/2)]
data_test = data[int(len(data)/2):]

# Learning parameters
learning_rate = 0.001
# training_epochs = int(input("Insert training epochs:"))
training_epochs = 200
batch_size = 1000
display_step = 1

# tf Graph input
n_input = len_vec_x # ... data input (23 input variables)
n_classes = 2 # ... total classes (0 or 1)

# 아래에서 eval 또는 run메소드를 실행을 시킬 때 feed 내용을 입력받는 자리가 되는 tensor들이다.
# 구체적인 값은 정해져있지 않고, feed값을 받을 수 있는 칸만을 제공한다.
# 입력은 23개의 변수로 이루어져 있으므로 input tensor(x)는 shpae가 [None, 23]이어야 한다.
# 분류가 두가지로 되므로 output tensor(y)는 shape가 [None, 2]이어야 한다.
x = tf.placeholder(tf.float32, shape=[None, n_input])
y = tf.placeholder(tf.float32, shape=[None, n_classes])

# Model Parameters
# 각 hidden layer의 노드수를 설정
n_hidden_1 = 100 # 1st layer number of features
n_hidden_2 = 100 # 2nd layer number of features

# 시행에 따라 값을 재설정할 수 있는 weight값의 행렬들로 된 dict를 정의
params = {
    'h_1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h_2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h_out': tf.Variable(tf.random_normal([n_hidden_2, n_classes])),
    'b_1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b_2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b_out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
def multilayer_perceptron(x, params):
    # Hidden layer_1 with sigmoid activation
    # 행렬곱 연산을 하고 bias값을 더함
    layer_1 = tf.add(tf.matmul(x, params['h_1']), params['b_1'])
    # sigmoid 활성함수를 사용
    layer_1 = tf.sigmoid(layer_1)

    # Hidden layer_2 with sigmoid activation
    layer_2 = tf.add(tf.matmul(layer_1, params['h_2']), params['b_2'])
    layer_2 = tf.sigmoid(layer_2)

    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, params['h_out']) + params['b_out']
    return out_layer

# predictor를 정의. x는 밑에서 eval메소드를 통해서 여러 입력값들이 한번에 입력된다.
predictor = multilayer_perceptron(x, params)

# Define loss and optimizer
# cross entropy를 비용함수로 정의
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=predictor, labels=y)
# cross entropy들의 평균을 계산하는 함수를 정의
cost = tf.reduce_mean(cross_entropy)
# gradient decent 알고리즘을 이용하여 cost의 값을 최소화하도록 weights를 재설정하는 함수를 정의
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Launching the graph
with tf.Session() as sess:
	# Selecting the device
	# 연산하는 장치를 선택
	# with tf.device("/gpu:1"):
	# with tf.device("/cpu:0"):
	# Initializing the variables
	tf.global_variables_initializer().run()	

	# Training cycle
	# 각 epoch의 cost를 plot하기 위하여 먼저 빈 list변수를 정의
	seq_epoch = []
	seq_cost = []

	# 위에서 정의한 epoch횟수만큼 동작을 반복
	for epoch in range(training_epochs):
		avg_cost = 0.
		total_batch = int(len(data_training)/batch_size)
		
		# Loop over batches
		# batch 입력, cost계산, 역전파 및 weights재설정
		for i in range(total_batch):
			# 아래의 코드를 사용하면 batch_size만큼 무작위추출을 하여 batch로 입력한다.
			# batch_x = data_training[np.random.choice(data_training.shape[0], size=batch_size, replace=False), 0:len_vec_x]
			# batch_y = data_training[np.random.choice(data_training.shape[0], size=batch_size, replace=False), len_vec_x]
			# batch_y = np.c_[batch_y, 1-batch_y]

			# 아래의 코드를 대신 사용하면 순서대로, 중복없이 batch를 입력할 수 있다.
			batch_x = data_training[i*batch_size:(i+1)*batch_size, 0:len_vec_x]
			batch_y = data_training[i*batch_size:(i+1)*batch_size, len_vec_x]
			batch_y = np.c_[batch_y, 1-batch_y]

			# Run optimization op (backprop) and cost op (to get loss value)
			# 앞에서 골라낸 batch를 input으로 cost를 계산하고, 역전파를 통해 weights 재설정
			_, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})

			# Compute average loss
			# 평균 error cost를 계산
			avg_cost += c / total_batch				

		# Display logs per epoch step
		# epoch마다 성능보고를 출력
		if epoch % display_step == 0:
			print("Epoch:", "%04d" % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

		# 각 epoch의 cost를 plot하기 위하여 epoch와 그 epoch의 error cost를 list에 추가
		seq_epoch.append(epoch)
		seq_cost.append(avg_cost)
	
	print("Optimization Finished!")

	# Ploting error cost over epoch
	# 가로축은 epoch, 세로축은 error cost인 그래프를 작성
	plt.clf()
	plt.plot(seq_epoch, seq_cost)
	plt.show()
	
	# Function to check if a prediction is correct
	# predictor에서 값이 가장 큰 index, y에서 값이 가장 큰 원소의 index가 같은지를 판단하는 함수를 정의
	correct_prediction = tf.equal(tf.argmax(predictor, 1), tf.argmax(y, 1))
	
	# Function to calculate accuracy
	# prediction이 성공한 비율을 계산하는 함수를 정의
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	
	# Reporting performance for training data
	# 모든 training data를 입력하여 성능보고를 출력
	data_training_x = data_training[:, 0:len_vec_x]
	data_training_y = data_training[:, len_vec_x]
	data_training_y = np.c_[data_training_y, 1-data_training_y]
	print("Accuracy(Training):", accuracy.eval({x: data_training_x, y: data_training_y}))
	
	# Reporting performance for test data
	# test data를 입력하여 성능보고를 출력
	data_test_x = data_test[:, 0:len_vec_x]
	data_test_y = data_test[:, len_vec_x]
	data_test_y = np.c_[data_test_y, 1-data_test_y]
	print("Accuracy(Test):", accuracy.eval({x: data_test_x, y: data_test_y}))

	# Printing the values of model parameters("h_1" matrix)
	# model의 weight값들을 출력("h_1"행렬)
	# print(params["h_1"].eval())


	# Plotting an ROC curve
	# ROC커브를 그린다.
	# prediction = tf.argmax(predictor,1)
	# prediction = prediction.eval(feed_dict={x: data_test_x})
	# prediction = np.c_[prediction, 1-prediction]
	# speci = tf.contrib.metrics.streaming_sensitivity_at_specificity(predictions=prediction, labels=data_test_y, specificity=0.9)		
	# print(speci[0].eval())
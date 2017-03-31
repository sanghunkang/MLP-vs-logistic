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
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf



# Reading the data
# csv파일을 읽어옴
data_raw = genfromtxt(".\\data\\health_test_selected.csv", delimiter=',')
# 읽어온 자료에서 NA가 등장하는 모든 행을 제외
data_filtered = data_raw[~np.isnan(data_raw).any(axis=1)]
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

print(data_input[0])

print(len_vec_x)

# Target data
data_target = data_filtered[:,23]
# data_target = data[:,24]

# Merging input and target data
data = np.c_[data_input, data_target]

# Shuffling the dataset, in case target values are highly related to the order
np.random.shuffle(data)

# Spliting the dataset into training and test datasets
data_training = data[:int(len(data)/2)]
data_test = data[int(len(data)/2):]

XX_tr = data_training[:, 0:len_vec_x]
yy_tr = data_training[:,len_vec_x]

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(1000,1000.), random_state=1)

# clf = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
#        beta_1=0.9, beta_2=0.999, early_stopping=False,
#        epsilon=1e-08, hidden_layer_sizes=((23,23),(23,23)), learning_rate='constant',
#        learning_rate_init=0.001, max_iter=200, momentum=0.9,
#        nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
#        solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
#        warm_start=False)
clf.fit(XX_tr, yy_tr)
print(yy_tr)

# XX_test = data_test[:, 0:len_vec_x]
# yy_test = data_test[:,len_vec_x]
yy_tr_pred = []
for X in XX_tr:
	y_tr_pred = clf.predict([X])[0]
	yy_tr_pred.append(y_tr_pred)
print(yy_tr_pred)

from sklearn.metrics import accuracy_score
accu = accuracy_score(yy_tr, yy_tr_pred)
print(accu)

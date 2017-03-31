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
import statsmodels.api as sm

from numpy import genfromtxt
from sklearn.metrics import accuracy_score

# Import data
data_raw = genfromtxt(".\\data\\health_test_selected.csv", delimiter=',')
data = data_raw[~np.isnan(data_raw).any(axis=1)]
print(len(data))

def scale_linear_bycolumn(rawpoints, high=100.0, low=0.0):
    mins = np.min(rawpoints, axis=0)
    maxs = np.max(rawpoints, axis=0)
    rng = maxs - mins
    return high - (((high - low) * (maxs - rawpoints)) / rng)

data = scale_linear_bycolumn(data, high=1.0)
print(data[0])

# Shuffle the data, divide into training and test sets
np.random.shuffle(data)
data_training = data[:int(len(data)/2)]
data_test = data[int(len(data)/2):]

data_training_input = data_training[:,0:23]
data_training_output = data_training[:,23]



# logst = linear_model.LogisticRegression()

# logst.fit(data_training_input, data_training_output)

data_test_input = data_test[:,0:23]
data_test_output = data_test[:,23]

logit = sm.Logit(data_test_output, data_test_input)
logit_res = logit.fit()
print(logit_res.summary())

def pred(X, cutoff = 0.5):
	prob = logit.predict(logit_res.params, exog=X)
	if prob > cutoff:
		ret = 1
	else:
		ret = 0
	return ret

for c in [0.1*x for x in range(10)]:
	yy_tr_pred = [pred(X, cutoff=c) for X in data_test_input]
	accu = accuracy_score(data_test_output, yy_tr_pred)
	print(accu)
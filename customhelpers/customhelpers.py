#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Author        : Kang, Sanghun
# Contact       : sanghunkang.dev@gmail.com
# Started on    : 20170322(yyyymmdd)
# Last modified : 20170331(yyyymmdd)
# Project       : 20170322
#############################################################################
# Import packages
import matplotlib.pyplot as plt
import numpy as np

# Functions for messy computations
def encode_onehot(seq, n_classes, zero_columns=True):
    # Determine shape of the output one-hot matrix
    seq = seq.astype(np.int32, copy=False)
    n_rec = len(seq)
    n_classes = n_classes

    # Create the output one-hot matrix 
    mat = np.zeros((n_rec, n_classes))
    mat[np.arange(n_rec), seq] = 1

    # Omit all-zero columns if specified
    if zero_columns == False:
        ret = mat[:, np.apply_along_axis(np.count_nonzero, 0, mat) > 0]
    
    return ret
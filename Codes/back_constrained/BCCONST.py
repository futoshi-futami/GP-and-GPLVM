# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 13:14:32 2017

@author: Futami
"""
import numpy as np

def init_weights(shape, name, sample = "xavier"):
    if sample == "uniform":
        values = np.random.uniform(-0.08, 0.08, shape)
    elif sample == "xavier":
        values = np.random.uniform(-np.sqrt(6. / (shape[0] + shape[1])), np.sqrt(6. / (shape[0] + shape[1])), shape)
    elif sample == "ortho":
        W = np.random.randn(shape[0], shape[0])
        u, s, v = np.linalg.svd(W)
        values = u
    else:
        raise ValueError("Unsupported initialization scheme: %s" % sample)
    
    return values, name

def init_bias(size, name):
    return np.zeros((size,)), name


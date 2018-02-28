# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 09:53:07 2017

@author: Futami
"""
from __future__ import print_function

__docformat__ = 'restructedtext en'

import numpy as np
from mlp import HiddenLayer

import theano
import theano.tensor as T
rng = np.random.RandomState(1234)

class back_constrained_model(object):
    def __init__(self,D,Q):
        self.X=T.matrix('X')
        self.y = T.matrix('y')
        N=self.X.shape[0]
                
        self.hiddenLayer_x = HiddenLayer(rng=rng,input=self.X,n_in=D,n_out=20,activation=T.nnet.relu,number='_x')
        self.hiddenLayer_m = HiddenLayer(rng=rng,input=self.hiddenLayer_x.output,n_in=20,n_out=Q,activation=T.nnet.relu,number='_m')
        self.hiddenLayer_S = HiddenLayer(rng=rng,input=self.hiddenLayer_x.output,n_in=20,n_out=Q,activation=T.nnet.relu,number='_S')
        
        self.params= []
        self.params.extend(self.hiddenLayer_x.params)
        self.params.extend(self.hiddenLayer_m.params)
        #self.params.extend(self.hiddenLayer_S.params)
        
        self.L2_sqr = (self.hiddenLayer_x.W ** 2+self.hiddenLayer_m.W ** 2).sum()
        
        self.error = 0.5*T.sum((self.hiddenLayer_m.output-self.y)**2)/N

    def pretraining_mlp(self,test_set_x,test_set_y,learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001,batch_size=20):
        
        index = T.lscalar()
        cost = self.error + L2_reg * self.L2_sqr
        
        gparams = [T.grad(cost, param) for param in self.params]
        updates = [(param, param - learning_rate * gparam) for param, gparam in zip(self.params, gparams)]

        self.training_model = theano.function(
                inputs=[index],
                       outputs=cost,
                       updates=updates,
                       givens={
                               self.X: test_set_x[index * batch_size:(index + 1) * batch_size],
                               self.y: test_set_y[index * batch_size:(index + 1) * batch_size]
                               }
                       )
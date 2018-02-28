# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 15:36:57 2017

@author: Futami
"""

v = T.vector()
A = T.matrix()
results, updates = theano.scan(lambda i,A: rv_u[i]+A, sequences=[T.arange(rv_u.shape[0])],non_sequences=A)

 re = theano.function([A], results)
 rv_u = srng.normal((1000,10,10))
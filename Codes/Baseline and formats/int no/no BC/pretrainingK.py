# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 17:20:36 2017

@author: Futami
"""

import numpy as np

eps = 1e-40
class kernel(object):
    def __init__(self,sf2,sigma):
        self.sf2,self.l = sf2, sigma
    def RBF(self,X1, X2 = None):
        _X2 = X1 if X2 is None else X2
        dist = ((X1 / self.l)**2).sum(1)[:, None] + ((_X2 / self.l)**2).sum(1)[None, :] - 2*(X1 / self.l).dot((_X2 / self.l).T)
        RBF = self.sf2 * np.exp(-dist / 2.0)
        return (RBF + eps * np.eye(X1.shape[0])) if X2 is None else RBF

class pretraining(object):
    def __init__(self,beta,sf2,sigma,X,Z,Y):
        self.ker=kernel(sf2,sigma)
        self.Kmm=self.ker.RBF(Z)
        self.Kmn=self.ker.RBF(Z,X)
        Ainv=np.linalg.inv(np.sum(self.Kmn[:,None,:]*self.Kmn[None,:,:],-1)*beta+self.Kmm)
        self.Sigma=np.dot(self.Kmm,np.dot(Ainv,self.Kmm))
        self.Mu=beta*np.dot(self.Kmm,np.dot(Ainv,np.sum(self.Kmn[:,None,:]*Y.T[None,:,:],-1)))
    def preU(self):
        return self.Mu,self.Sigma
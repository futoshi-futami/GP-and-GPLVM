# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 14:01:29 2017

@author: Futami
"""

__docformat__ = 'restructedtext en'

import numpy as np

import theano
import theano.tensor as T
import theano.sandbox.linalg as sT
rng = np.random.RandomState(1234)

eps = 1e-4
class kernel(object):
    
    def __init__(self, Q,number="1"):
        lhyp_values = np.zeros(Q+1,dtype=theano.config.floatX)
        self.lhyp = theano.shared(value=lhyp_values, name='lhyp'+number, borrow=True)
        self.params = [self.lhyp]
        
        self.sf2,self.l = T.exp(self.lhyp[0]), T.exp(self.lhyp[1:1+Q])
        
    def RBF(self,X1, X2 = None):
        _X2 = X1 if X2 is None else X2
        dist = ((X1 / self.l)**2).sum(1)[:, None] + ((_X2 / self.l)**2).sum(1)[None, :] - 2*(X1 / self.l).dot((_X2 / self.l).T)
        RBF = self.sf2 * T.exp(-dist / 2.0)
        return (RBF + eps * T.eye(X1.shape[0])) if X2 is None else RBF
    def RBFnn(self, sf2, l, X):
        return sf2 + eps
    def LIN(self, sl2, X1, X2 = None):
        _X2 = X1 if X2 is None else X2
        LIN = sl2 * (X1.dot(_X2.T) + 1)
        return (LIN + eps * T.eye(X1.shape[0])) if X2 is None else LIN
    def LINnn(self, sl2, X):
        return sl2 * (T.sum(X**2, 1) + 1) + eps

class MMDpenalty(object):
    def __init__(self, rng,input_m,input_S, n_in, n_out,inducing_number,Domain_number=None,
                 liklihood="Gaussian",Domain_consideration=True,number="1",kernel_name='X'):

        m=input_m
        S_0=input_S
        
        self.N=m.shape[0]
        D=n_out
        Q=n_in
        M=inducing_number
        
        #set_initial_value
        ker=kernel(Q,kernel_name)
        mu_value = np.random.randn(M,D)* 1e-2
        Sigma_b_value = np.zeros((M,M))
        Z_value = np.random.randn(M,Q)
        if Domain_consideration:
            ls_value=np.zeros(Domain_number)+np.log(0.1)
        else:
            ls_value=np.zeros(1)+np.log(0.1)
        
        self.mu = theano.shared(value=mu_value, name='mu'+number, borrow=True)
        self.Sigma_b = theano.shared(value=Sigma_b_value, name='Sigma_b'+number, borrow=True)
        self.Z = theano.shared(value=Z_value, name='Z'+number, borrow=True)
        self.ls = theano.shared(value=ls_value, name='ls'+number, borrow=True)
        
        self.params = [self.mu,self.Sigma_b,self.Z,self.ls]
        
        
        self.params.extend(ker.params)
        
        self.hyp_params_list=[self.mu,self.Sigma_b,self.ls]
        self.Z_params_list=[self.Z]        
        self.global_params_list=self.params
        
        S_1=T.exp(S_0)
        S=T.sqrt(S_1)
        
        from theano.tensor.shared_randomstreams import RandomStreams
        srng = RandomStreams(seed=234)
        eps_NQ = srng.normal((self.N,Q))
        eps_M = srng.normal((M,D))#平均と分散で違う乱数を使う必要があるので別々に銘銘
        eps_ND = srng.normal((self.N,D))
                          
        self.beta = T.exp(self.ls)
        #uについては対角でないのでコレスキー分解するとかして三角行列を作る必要がある

        Sigma = T.tril(self.Sigma_b - T.diag(T.diag(self.Sigma_b)) + T.diag(T.exp(T.diag(self.Sigma_b))))
        
        #スケール変換
        mu_scaled, Sigma_scaled = ker.sf2**0.5 * self.mu, ker.sf2**0.5 * Sigma
        
        Xtilda = m + S * eps_NQ
        self.U = mu_scaled+Sigma_scaled.dot(eps_M)
        
        Kmm = ker.RBF(self.Z)
        KmmInv = sT.matrix_inverse(Kmm) 
        
        Kmn = ker.RBF(self.Z,Xtilda)
        
        Knn = ker.RBF(Xtilda)        
        Ktilda=Knn-T.dot(Kmn.T,T.dot(KmmInv,Kmn))
        
        
        #F = T.dot(Kmn.T,T.dot(KmmInv,self.U)) + T.dot(T.maximum(Ktilda, 1e-16)**0.5,eps_ND)
        
        Kinterval=T.dot(KmmInv,Kmn)
        A=Kinterval.T      
        Sigma_tilda=Ktilda+T.dot(A,T.dot(Sigma_scaled,A.T))
        mean_tilda=T.dot(A,mu_scaled)
        #mean_U=F
        #mean_U=T.dot(Kinterval.T,self.U)
        self.mean_U=mean_tilda + T.dot(T.maximum(Sigma_tilda, 1e-16)**0.5,eps_ND)

        
        self.output=self.mean_U
        self.KL_X = -self.KLD_X(m,S)
        self.KL_U = -self.KLD_U(mu_scaled , Sigma_scaled , Kmm,KmmInv)
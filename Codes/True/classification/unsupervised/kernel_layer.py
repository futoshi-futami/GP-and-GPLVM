# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 10:36:53 2017

@author: Futami
"""
"""
kernel_layer for the deep bayesian model
"""

#from __future__ import print_function

__docformat__ = 'restructedtext en'

import numpy as np

import theano
import theano.tensor as T
import theano.sandbox.linalg as sT
rng = np.random.RandomState(1234)

from logistic_sgd import LogisticRegression

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


######################################################################        
class KernelLayer(object):
    def __init__(self, rng, target,input_m,input_S, n_in, n_out,inducing_number,Domain_number,Xlabel,
                 liklihood="Gaussian",Domain_consideration=True,number="1"):

        m=input_m
        S_0=input_S
        
        N=m.shape[0]
        D=n_out
        Q=n_in
        M=inducing_number
        
        #set_initial_value
        ker=kernel(Q)
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
        eps_NQ = srng.normal((N,Q))
        eps_M = srng.normal((M,D))#平均と分散で違う乱数を使う必要があるので別々に銘銘
        eps_ND = srng.normal((N,D))
                          
        beta = T.exp(self.ls)
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
        mean_U=mean_tilda + T.dot(T.maximum(Sigma_tilda, 1e-16)**0.5,eps_ND)
        betaI=T.diag(T.dot(Xlabel,beta))
        Covariance = betaI       
        
        self.output=mean_U
        
        self.LL = self.log_mvn(target, mean_U, Covariance)/N# - 0.5*T.sum(T.dot(betaI,Ktilda))       
        self.KL_X = -self.KLD_X(m,S)
        self.KL_U = -self.KLD_U(mu_scaled , Sigma_scaled , Kmm,KmmInv)
        
    
    def log_mvn(self, y, mean,beta):#対角ノイズ、YはＮ×Ｄのデータ,それの正規分布の対数尤度
        N = y.shape[0]
        D = y.shape[1]
        return -0.5 *  D * T.sum(T.log(2 * np.pi*(1/T.diag(beta)))) - 0.5 * T.sum(T.dot(beta,(y - mean)**2))
    
    def KLD_X(self,m,S):
        N = m.shape[0]
        Q = m.shape[1]
        
        KL_X = T.sum(m*m)+T.sum(S-T.log(S)) - Q*N
        
        return 0.5*KL_X
    
    def KLD_U(self, m, L_scaled, Kmm,KmmInv):#N(u|m,S)とN(u|0,Kmm) S=L*L.T(コレスキー分解したのを突っ込みましょう)
        M = m.shape[0]
        D = m.shape[1]
        #KmmInv = sT.matrix_inverse(Kmm)
        
        KL_U = D * (T.sum(KmmInv.T * L_scaled.dot(L_scaled.T)) - M - 2.0*T.sum(T.log(T.diag(L_scaled))) + 2.0*T.sum(T.log(T.diag(sT.cholesky(Kmm)))))
        KL_U += T.sum(T.dot(KmmInv,m)*m) 
        
        return 0.5*KL_U
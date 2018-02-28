# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 11:31:46 2017

@author: Futami
"""

from __future__ import print_function

__docformat__ = 'restructedtext en'

import numpy as np

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

# start-snippet-1
class RFFLayer(object):
    def __init__(self, rng, input, n_in, n_out, num_MC,num_FF,Domain_number=None,number="1",Domain_consideration=True):
        #inputも100＊N*Dで入ってくるようにする.
        #DATA=input
        #N=DATA.shape[1]
        #n_in_D=DATA.shape[2]
        srng = RandomStreams(seed=234)
        
        #Define hyperparameters
        lhyp_values = np.zeros(n_in+1,dtype=theano.config.floatX)+np.log(0.1,dtype=theano.config.floatX)
        self.lhyp = theano.shared(value=lhyp_values, name='lhyp'+number, borrow=True)
        self.sf2,self.l = T.exp(self.lhyp[0]), T.exp(self.lhyp[1:1+n_in])
        
        if Domain_consideration:
            ls_value=np.zeros(Domain_number,dtype=theano.config.floatX)+np.log(0.1,dtype=theano.config.floatX)
        else:
            ls_value=np.zeros(1,dtype=theano.config.floatX)+np.log(0.1,dtype=theano.config.floatX)
        
        self.ls = theano.shared(value=ls_value, name='ls'+number, borrow=True)
        
        
        
        #Define prior omega
        #prior_mean_Omega.append(tf.zeros([self.d_in[i],1]))
        log_prior_var_Omega=T.tile(1/(self.l)**0.5,(num_FF,1)).T
        
        #Define posterior omega
        
        #get samples from  omega
        sample_value = np.random.randn(1,n_in,num_FF)
        sample_Omega_epsilon_0 = theano.shared(value=sample_value, name='sample_Omega'+number)
        #sample_Omega_epsilon_0 = srng.normal((1,n_in,num_FF))
        Omega_sample=sample_Omega_epsilon_0*log_prior_var_Omega[None,:,:]
        Omega_samples=T.tile(Omega_sample,(num_MC,1,1))
        
        
        #Define prior W
        prior_mean_W = T.zeros(2*num_FF)
        log_prior_var_W = T.ones(2*num_FF)        
        
        #Define posterior W
        mean_mu_value = np.random.randn(2*num_FF,n_out)* 1e-2 
        self.mean_mu = theano.shared(value=mean_mu_value, name='mean_mu'+number, borrow=True)
        
        log_var_value = np.zeros((2*num_FF,n_out))
        self.log_var_W = theano.shared(value=log_var_value, name='q_W'+number, borrow=True)
        
        #get samples from W
        sample_Omega_epsilon = srng.normal((num_MC,2*num_FF,n_out))
        W_samples = sample_Omega_epsilon * (T.exp(self.log_var_W)**0.5)[None,:,:] + self.mean_mu[None,:,:]
                
        # calculate lyaer N_MC*N*D_out
        F_next, updates = theano.scan(fn=lambda a,b,c: self.passage(a,b,c,num_FF),
                              sequences=[input,Omega_samples,W_samples])
        
        #output
        self.output = F_next
        
        #KL-divergence
         #Omega
         
         
         #W
        self.KL_W=self.DKL_gaussian(self.mean_mu, self.log_var_W, prior_mean_W, log_prior_var_W)
        
        #parameter_setting
        self.all_params=[self.lhyp,self.ls,self.mean_mu,self.log_var_W]
        self.hyp_params=[self.lhyp,self.ls]
        self.variational_params=[self.mean_mu,self.log_var_W]
        #self.no_update=[sample_Omega_epsilon_0]
        #self.check=self.variational_params
    #################################################################################
    
    def passage(self,F,omega,W,num_FF):
        F_times_Omega = T.dot(F, omega)#minibatch_size*n_rff
        Phi = (self.sf2**0.5 /num_FF**0.5 ) * T.concatenate([T.cos(F_times_Omega), T.sin(F_times_Omega)],1)
        F_next=T.dot(Phi,W)
        
        return F_next    
    
    def KLD_X(self,m,S):
        N = m.shape[0]
        Q = m.shape[1]
        
        KL_X = T.sum(m*m)+T.sum(S-T.log(S)) - Q*N
        
        return 0.5*KL_X
    
    ## Kullback-Leibler divergence between multivariate Gaussian distributions q and p with diagonal covariance matrices
    def DKL_gaussian(self,mq, log_vq, mp, log_vp):
        """
        KL[q || p]
        :param mq: vector of means for q
        :param log_vq: vector of log-variances for q
        :param mp: vector of means for p
        :param log_vp: vector of log-variances for p
        :return: KL divergence between q and p
        """
        
        N,D=log_vq.shape
        
        log_vp_trans=T.tile(log_vp,(D,1)).T
        mp_trans=T.tile(mp,(D,1)).T
                           
        return 0.5 * T.sum(log_vp - log_vq + ((mq - mp_trans)**2 / T.exp(log_vp_trans)) + T.exp(log_vq - log_vp_trans) - 1)
    
    ################################################################################
    #calculate log gaussian    
    def log_mvn(self, y, mean,beta):#対角ノイズ、YはＮ×Ｄのデータ,それの正規分布の対数尤度
        N = y.shape[0]
        D = y.shape[1]
        
        LL, updates = theano.scan(fn=lambda a: -0.5 *  D * T.sum(T.log(2 * np.pi*(1/T.diag(beta)))) - 0.5 * T.sum(T.dot(beta,(y - a)**2)),
                              sequences=[mean])
        return T.mean(LL)
    
    def log_mvns(self, y, mean,beta):#対角ノイズ、YはＮ×Ｄのデータ,それの正規分布の対数尤度
        N = y.shape[0]
        D = y.shape[1]
        
        LL, updates = theano.scan(fn=lambda a: -0.5 *  D * T.sum(T.log(2 * np.pi*(1/beta))) - 0.5 * T.sum(beta*T.sum((y - a)**2)),
                              sequences=[mean])
        
        return T.mean(LL)
    
    def likelihood_domain(self,target,Xlabel):
        self.beta = T.exp(self.ls)
        betaI=T.diag(T.dot(Xlabel,self.beta))
        Covariance = betaI       
        LL = self.log_mvn(target, self.output, Covariance)# - 0.5*T.sum(T.dot(betaI,Ktilda))      
        
        return LL
    
    def liklihood_nodomain(self,target):            
        Covariance = self.beta
        LL = self.log_mvns(target, self.output, Covariance)# - 0.5*T.sum(T.dot(betaI,Ktilda)))   
    
        return LL
    
    def error_RMSE(self,target):            
        pred=T.mean(self.output,0)
        error_rate = T.sqrt(T.mean((target - pred)**2))
    
        return error_rate
    
    ###################################################################################
    #classification parts
    
    def softmax_class(self):
        output, updates = theano.scan(fn=lambda a: T.nnet.softmax(a),
                              sequences=[self.output])
        return T.mean(output,0)
    
    def error_classification(self,target):
        output, updates = theano.scan(fn=lambda a: T.nnet.softmax(a),
                              sequences=[self.output])
        y=T.mean(output,0)
        self.y_pred = T.argmax(y, axis=1)
        label=T.argmax(target, axis=1)
        return T.mean(T.neq(self.y_pred, label))
    
    def classification_liklihood(self,target):
        output, updates = theano.scan(fn=lambda a: T.nnet.softmax(a),
                              sequences=[self.output])
        y=T.mean(output,0)
        self.LLY=T.sum(T.log(T.maximum(T.sum(target * y, 1), 1e-16)))
        
        return self.LLY
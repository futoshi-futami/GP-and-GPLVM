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
        self.DATA=input
        #N=DATA.shape[1]
        #n_in_D=DATA.shape[2]
        srng = RandomStreams(seed=234)
        self.num_rff=num_FF
        
        #Define hyperparameters
        #lhyp_values = np.zeros(n_in+1,dtype=theano.config.floatX)+np.log(0.1,dtype=theano.config.floatX)
        lhyp_values = np.zeros(n_in+1,dtype=theano.config.floatX)+np.log(1.,dtype=theano.config.floatX)
        self.lhyp = theano.shared(value=lhyp_values, name='lhyp'+number, borrow=True)
        self.sf2,self.l = T.exp(self.lhyp[0]), T.exp(self.lhyp[1:1+n_in])
        
        if Domain_consideration:#先行研究は0.1でうまくいった
            ls_value=np.zeros(Domain_number,dtype=theano.config.floatX)+np.log(1.,dtype=theano.config.floatX)
        else:
            ls_value=np.zeros(1,dtype=theano.config.floatX)+np.log(1.,dtype=theano.config.floatX)
        
        self.ls = theano.shared(value=ls_value, name='ls'+number, borrow=True)
        
        
        
        #Define prior omega
        #prior_mean_Omega.append(tf.zeros([self.d_in[i],1]))
        self.log_prior_var_Omega=T.tile(1/(self.l)**0.5,(num_FF,1)).T
        
        #Define posterior omega
        
        #get samples from  omega
        sample_value = np.random.randn(1,n_in,num_FF)
        self.sample_Omega_epsilon_0 = theano.shared(value=sample_value, name='sample_Omega'+number)
        #self.sample_Omega_epsilon_0 = srng.normal((1,n_in,num_FF))
        Omega_sample=self.sample_Omega_epsilon_0*self.log_prior_var_Omega[None,:,:]
        Omega_samples=T.tile(Omega_sample,(num_MC,1,1))
        
        self.samples=Omega_samples
        #Define prior W
        prior_mean_W = T.zeros(2*num_FF)
        log_prior_var_W = T.ones(2*num_FF)        
        
        #Define posterior W
        mean_mu_value = np.random.randn(2*num_FF,n_out)#* 1e-2 
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
        
        LL, updates = theano.scan(fn=lambda a: -0.5 *  D *N* T.sum(T.log(2 * np.pi*(1/beta))) - 0.5 * T.sum(beta*T.sum((y - a)**2)),
                              sequences=[mean])
        
        return T.mean(LL)
    
    def likelihood_domain(self,target,Xlabel):
        self.beta = T.exp(self.ls)
        betaI=T.diag(T.dot(Xlabel,self.beta))
        Covariance = betaI       
        LL = self.log_mvn(target, self.output, Covariance)# - 0.5*T.sum(T.dot(betaI,Ktilda))      
        
        return LL
    
    def liklihood_nodomain(self,target):            
        self.beta = T.exp(self.ls)
        Covariance = self.beta
        LL = self.log_mvns(target, self.output, Covariance)# - 0.5*T.sum(T.dot(betaI,Ktilda)))   
    
        return LL
    
    def error_RMSE(self,target):            
        pred=T.mean(self.output,0)
        #mu=T.mean(target,0)
        error_rate = (T.mean((target - pred)**2,0))**0.5
    
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
    
    def MMD_central_penalty(self,Xlabel):

        MMD_sample, updates = theano.scan(fn=lambda a,b: self.For_MMD_Sub(a,b,self.num_rff,Xlabel),
                              sequences=[self.DATA,self.samples])
        
        return T.mean(MMD_sample)
        
    def MMD_class_penalty(self,target,Xlabel):
        #10個のリスト。順番に各クラスの数が入っている
        Num_c=T.sum(target,0)
        D_num=Xlabel.shape[1]
        #C*Domain_numの行列。書くますに例えばクラスｃにはドメイン１，２，３はそれぞれ何個ずついるか計算している
        Number_label=T.sum(target.T[:,None,:]*Xlabel.T[None,:,:],2)
        
        K_base=self.kern.RBF(self.cal,self.cal)
        
        #10*N*N　クラスごとの全てのドメインを無視したグラム行列
        K_class, updates = theano.scan(fn=lambda a: ((K_base*a).T*a),
                              sequences=[target.T])
        #グラム行列の和をとっている
        K_allsum=T.sum(T.sum(K_class,-1),-1)
        #それぞれのクラスの数の2乗で割る必要があるが万が一クラスの数が０だと割っては無限になってしまうので、if文でチェックを入れている
        K_sum_tot, updates = theano.scan(fn=lambda a,b: T.switch(T.gt(b,0), a/b**2, 0),
                              sequences=[K_allsum,Num_c])
        
        #10*3*N*N(クラス、ドメイン、ごとグラム行列)ただし全てのドメインとのクラスではないのでフィルターであるxlabelを両方からかけている。またあるクラスについて、その中のドメインを順番に見るので、scan文の2重ループ
        K_class_domain_cross,updates = theano.scan(fn=lambda c:
                                        theano.scan(fn=lambda a: ((c*a).T*a),
                                                    sequences=[Xlabel.T])
                                        ,sequences=[K_class])
        #domainごとにクラスごとになっている今はC*D グラム行列の和をとっている
        K_allsum=T.sum(T.sum(K_class_domain_cross,-1),-1)
        #割り方だが、あるクラスのあるドメインに属しているのが誰もいなかったらC*D_numのグラム行列和の成分が0になっているはず。それと同じ分母の数を入れている行列も0になっているはず。なのでもし０なら分母には１を変わりに入れる。しかし結局分子で０になるので問題ない
        Number_label2=T.where(T.eq(Number_label,0),1,Number_label)
        K_class_sum=T.sum(K_allsum/(Number_label2**2))
        
        
        #あるクラスのあるドメインとあるクラスの全てのドメインのクロス。そのためフィルターは横方向からかけているだけ。
        K_class_domain_center_cross,updates = theano.scan(fn=lambda c:
                                        theano.scan(fn=lambda a: (c*a),
                                                    sequences=[Xlabel.T])
                                        ,sequences=[K_class])
        #上のドメインごとのものと同じ処理を繰り返す
        K_sum_cross=T.sum(T.sum(K_class_domain_center_cross,-1),-1)
        Number_label2=T.where(T.eq(Number_label,0),1,Number_label)
        K_domain_cross_sum=T.sum(K_sum_cross/(Number_label2*Num_c[:,None]))
        #z_switch = T.switch(T.lt(a, b), T.mean(x), T.mean(y))
        
        MMD_class=K_class_sum+T.sum(K_sum_tot)*D_num-2*K_domain_cross_sum
        
        return MMD_class
    
    def For_MMD_Sub(self,data,omega,num_FF,Xlabel):
        
        Num=T.sum(Xlabel,0)
        D_num=Xlabel.shape[1]
        N=data.shape[0]
        
        F_times_Omega = T.dot(data, omega)#minibatch_size*n_rff
        Phi = (self.sf2**0.5 /num_FF**0.5 ) * T.concatenate([T.cos(F_times_Omega), T.sin(F_times_Omega)],1)
        
        #各RFFは２N_rffのたてベクトル
        Phi_total=T.sum(Phi.T,-1)/N
        
        #Domain_number*2N_rffの行列
        Phi_each_domain, updates = theano.scan(fn=lambda a,b: T.switch(T.neq(b,0), Phi.T*a/b, 0),
                              sequences=[Xlabel.T,Num])
        each_Phi=T.sum(Phi_each_domain,-1)
        #まず自分自身との内積 結果はD次元のベクトル
        each_domain_sum=T.sum(each_Phi*each_Phi,-1)
        
        #全体の内積
        tot_sum=T.dot(Phi_total,Phi_total)
        
        #全体とドメインのクロス内積
        tot_domain_sum, updates=theano.scan(fn=lambda a: a*Phi_total,
                              sequences=[each_Phi])
        
        #MMDの計算
        MMD_central=T.sum(each_domain_sum)+D_num*tot_sum-2*T.sum(tot_domain_sum)
        
        return MMD_central
    
    def For_MMD_Sub2(self,data,omega,num_FF,Xlabel):
        
        Num=T.sum(Xlabel,0)
        D_num=Xlabel.shape[1]
        N=data.shape[0]
        
        F_times_Omega = T.dot(data, omega)#minibatch_size*n_rff
        Phi = (self.sf2**0.5 /num_FF**0.5 ) * T.concatenate([T.cos(F_times_Omega), T.sin(F_times_Omega)],1)
        
        #各RFFは２N_rffのたてベクトル
        Phi_total=T.sum(Phi.T,-1)/N
        
        #Domain_number*2N_rffの行列
        T.sum(Xlabel.T[:,None,:]*Phi.T[None,:,:],-1)
        Phi_each_domain, updates = theano.scan(fn=lambda a,b: T.switch(T.neq(b,0), Phi.T*a/b, 0),
                              sequences=[Xlabel.T,Num])
        each_Phi=T.sum(Phi_each_domain,-1)
        #まず自分自身との内積 結果はD次元のベクトル
        each_domain_sum=T.sum(each_Phi*each_Phi,-1)
        
        #全体の内積
        tot_sum=T.dot(Phi_total,Phi_total)
        
        #全体とドメインのクロス内積
        tot_domain_sum, updates=theano.scan(fn=lambda a: a*Phi_total,
                              sequences=[each_Phi])
        
        #MMDの計算
        MMD_central=T.sum(each_domain_sum)+D_num*tot_sum-2*T.sum(tot_domain_sum)
        
        return MMD_central
    
    
    #よる作成のとのこと
    def For_MMD_Sub_class(self,target,data,omega,num_FF,Xlabel):
        
        Num=T.sum(Xlabel,0)
        D_num=Xlabel.shape[1]
        N=data.shape[0]
        
        F_times_Omega = T.dot(data, omega)#minibatch_size*n_rff
        Phi = (self.sf2**0.5 /num_FF**0.5 ) * T.concatenate([T.cos(F_times_Omega), T.sin(F_times_Omega)],1)
        
        #各RFFは２N_rffのたてベクトル
        Phi_total=T.sum(Phi.T,-1)/N
        
        #Domain_number*2N_rffの行列
        Phi_each_domain, updates = theano.scan(fn=lambda a,b: T.switch(T.neq(b,0), Phi.T*a/b, 0),
                              sequences=[Xlabel.T,Num])
        each_Phi=T.sum(Phi_each_domain,-1)
        #まず自分自身との内積 結果はD次元のベクトル
        each_domain_sum=T.sum(each_Phi*each_Phi,-1)
        
        #全体の内積
        tot_sum=T.dot(Phi_total,Phi_total)
        
        #全体とドメインのクロス内積
        tot_domain_sum, updates=theano.scan(fn=lambda a: a*Phi_total,
                              sequences=[each_Phi])
        
        #MMDの計算
        MMD_central=T.sum(each_domain_sum)+D_num*tot_sum-2*T.sum(tot_domain_sum)
        
        return MMD_central     
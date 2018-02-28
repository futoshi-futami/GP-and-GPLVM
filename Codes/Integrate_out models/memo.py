# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 11:43:34 2017

@author: Futami
"""


DD1=np.tile(Label.T, (10, 1,1))
tttt=DD1[:,:,:,None]*DD1.transpose((1,0,2))[:,:,None,:]
np.where(np.arange(6).reshape(2,3)==0,1,np.arange(6).reshape(2,3))
np.arange(6).reshape(2,3).cumprod(0)


 
 tt=KERNEL(X_training_0,X_training_0)
 t=tttt*tt[None,None,:,:]
 h=np.sum(np.sum(t,-1),-1)

 T.extra_ops.repeat(h,10,0)

 
 np.tile(np.diag(h),(10,1))
 
 
 np.tile(np.diag(h),(10,1)).T-np.tile(np.diag(h),(10,1))-2*h
        
        new=np.exp(-(np.tile(np.diag(h),(10,1)).T+np.tile(np.diag(h),(10,1))-2*h)/(2*3**2))
        #taskkernel

KK=tttt*new[:,:,None,None]
KK1=np.where(KK==0,1,KK)
KK2=np.prod(np.prod(KK1,0),0)
#これでＮ×Ｎになった。あとはＲＢＦカーネルと要素席を取る

f=theano.function([x],T.diag(x))
 
f=theano.function([x],T.extra_ops.repeat(x,5,0))
 

#必要なのは

DD1=T.tile(Label.T, (10, 1,1))
tttt=DD1[:,:,:,None]*DD1.transpose((1,0,2))[:,:,None,:]

Hh=T.sum(T.sum(tttt*x[None,None,:,:],-1),-1)
kong=T.diag(Hh)
godzira=T.extra_ops.repeat(Hh,10,0)

GH=T.extra_ops.repeat(T.diag(Hh),10,0)


new=T.exp(-(GH.T+GH-2*Hh)/(2*gamma**2))*alpha
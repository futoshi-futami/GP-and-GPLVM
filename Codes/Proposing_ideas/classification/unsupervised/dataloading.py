# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 18:21:53 2017

@author: Futami
"""
#matファイルの呼び出しの仕方

import scipy.io
import numpy as np

def one_hot_maker(label,class_number):
    label_case=[]
    for i in range(label.shape[0]):
        g=np.zeros(class_number)
        g[label[i]-1]=1
        label_case.append(g)
    return np.array(label_case)

#########################################################
mat = scipy.io.loadmat('amazon_SURF_L10.mat')
matim = scipy.io.loadmat('amazon_SURF_L10_imgs.mat')
#web上のやつは辞書形式になっているのが多いので
ama=mat['fts']#N*D
ama_l=mat['labels']#N個のデータセット

label_ama=one_hot_maker(ama_l,10)
########################################################
         
mat2 = scipy.io.loadmat('Caltech10_SURF_L10.mat')

#web上のやつは辞書形式になっているのが多いので
cal=mat2['fts']#N*D
cal_l=mat2['labels']#N

label_cal=one_hot_maker(cal_l,10)

#################################################################
mat3 = scipy.io.loadmat('webcam_SURF_L10.mat')
web=mat3['fts']#N*D
web_l=mat3['labels']#N

label_web=one_hot_maker(web_l,10)

############################################################

mat4 = scipy.io.loadmat('dslr_SURF_L10.mat')
dslr=mat4['fts']#N*D
dslr_l=mat4['labels']#N

label_dslr=one_hot_maker(dslr_l,10)

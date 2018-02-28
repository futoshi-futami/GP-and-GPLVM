import sys; sys.path.append("./dataset")

from DGPLVM_opt_cl import DGGPLVM_opt
import pickle
import numpy as np

import time
from numpy.random import randn, rand
from pandas import Series, DataFrame
import pandas as pd
from scipy.spatial.distance import pdist,squareform
from scipy import exp
from scipy.linalg import eigh



#以上データの読み込み--------------------------
from scipy.optimize import minimize

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
ama_d={'label':label_ama,'data':ama,'Num':ama.shape[0]}
########################################################
         
mat2 = scipy.io.loadmat('Caltech10_SURF_L10.mat')

#web上のやつは辞書形式になっているのが多いので
cal=mat2['fts']#N*D
cal_l=mat2['labels']#N

label_cal=one_hot_maker(cal_l,10)
cal_d={'label':label_cal,'data':cal,'Num':cal.shape[0]}
#################################################################
mat3 = scipy.io.loadmat('webcam_SURF_L10.mat')
web=mat3['fts']#N*D
web_l=mat3['labels']#N

label_web=one_hot_maker(web_l,10)
web_d={'label':label_web,'data':web,'Num':web.shape[0]}
############################################################

mat4 = scipy.io.loadmat('dslr_SURF_L10.mat')
dslr=mat4['fts']#N*D
dslr_l=mat4['labels']#N

label_dslr=one_hot_maker(dslr_l,10)
dslr_d={'label':label_dslr,'data':dslr,'Num':dslr.shape[0]}
#################################################################


X=np.vstack((np.vstack((dslr,web)),cal))
Y=np.vstack((np.vstack((label_dslr,label_web)),label_cal))
N=X.shape[0]
D=X.shape[1]
class_N=Y.shape[1]

N_p=3#testに用いる確率分布（人の数）

label1=np.zeros((dslr_d['Num'],N_p))
label1[:,0]=1
label2=np.zeros((web_d['Num'],N_p))
label2[:,1]=2
label3=np.zeros((cal_d['Num'],N_p))
label3[:,2]=3
Domain_label=np.vstack((np.vstack((label1,label2)),label3))

M = 100
Hiddenlayerdim1=500
Hiddenlayerdim2=500
Q = 20
max_iteration = 1000
batch_size=300


#初期値の設定


minibatch = np.random.choice(N,N,replace=False)
X=X[minibatch]
X=X.astype(np.float64)
Y=Y[minibatch]
Domain_label=Domain_label[minibatch]

import theano
X_tr = theano.shared(X)
Y_tr = theano.shared(Y)
Xlabel_share = theano.shared(Domain_label)



dggplvm = DGGPLVM_opt(D,M,Q,N_p,X_tr,Xlabel_share,batch_size,Hiddenlayerdim1,Hiddenlayerdim2)


def onestep_opt():
    epoch = 0
    max_epoch=1000
    start = time.time()
    n_train_batches = X.shape[0] // batch_size
    iteration=0
    param_saver={}
    while iteration < 1000:
        #index=random.choice(n_train_batches)
        for index in range(n_train_batches):
            optming_params={}
            dggplvm.opt_one_step(iteration, index)
            KL_U=dggplvm.f['KL_U']()
            KL_X=dggplvm.f['KL_X']()
            LL=dggplvm.dggplvm.estimate_f(index,100)
            print ('iter ' + str(iteration) + ': KL_U is:' + str(KL_U))
            print ('KL_X is:' + str(KL_X)+'and LL is ' + str(LL))
            iteration +=1
            for i in dggplvm.wrt:
                optming_params[str(i)]=dggplvm.wrt[i].get_value()
            optming_params['param_updates']=dggplvm.param_updates
            optming_params['moving_mean_squared']=dggplvm.moving_mean_squared
            optming_params['learning_rates']=dggplvm.learning_rates
            
            param_saver[str(iteration)]=optming_params
            elapsed_time = time.time() - start
            print(elapsed_time)
#    while epoch < max_epoch:
#        index=random.choice(n_train_batches)
#        dggplvm.opt_one_step2(iteration, index)
#        KL_U=dggplvm.f['KL_U']()
#        KL_X=dggplvm.f['KL_X']()
#        LL=dggplvm.dggplvm.estimate_f(index,50)
#        print ('iter ' + str(iteration) + ': KL_U is:' + str(KL_U))
#        print ('KL_X is:' + str(KL_X)+'and LL is ' + str(LL))
#        iteration +=1
#        elapsed_time = time.time() - start
#        print(elapsed_time)
#        epoch += 1

    #データの保存
    saving_learning_rates={'param_updates':dggplvm.param_updates,'moving_mean_squared':dggplvm.moving_mean_squared,'learning_rates':dggplvm.learning_rates}
    saving_params={}
    for i in dggplvm.wrt:
        saving_params[str(i)]=dggplvm.wrt[i].get_value()
    save=[saving_params,saving_learning_rates,epoch,iteration]

    with open('params.dump', 'wb') as f:
        pickle.dump(save, f)

def grad_variance_checker():
    start = time.time()
    n_train_batches = X.shape[0] // batch_size
    iteration=0
    while iteration < 1000:
        index=random.choice(n_train_batches)
        dggplvm.opt_one_step(iteration, index)
        KL_U=dggplvm.f['KL_U']()
        KL_X=dggplvm.f['KL_X']()
        LL=dggplvm.dggplvm.estimate_f(index,50)
        print ('iter ' + str(iteration) + ': KL_U is:' + str(KL_U))
        print ('KL_X is:' + str(KL_X)+'and LL is ' + str(LL))
        if iteration%5==0:
            variance=[]
            name=[]
            for i in dggplvm.wrt:
                variance.append(np.max(dggplvm.estimate(i, index)[1]))
                name.append(i)
            n=np.argmax(variance)
            print('the largest variance is:'+str(name[n])+'_and value is'+str(variance[n]))
        iteration +=1
        elapsed_time = time.time() - start
        print(elapsed_time)

if __name__ == '__main__':
    onestep_opt()
        

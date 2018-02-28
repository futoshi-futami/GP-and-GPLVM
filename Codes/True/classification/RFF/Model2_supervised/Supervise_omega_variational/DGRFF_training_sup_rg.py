import sys; sys.path.append("./dataset")
import os
from DGRFF_opt_sup_rg import Dgrff_opt
import pickle
import numpy as np

import sys; sys.path.append("./")
import sys; sys.path.append("../")

from pandas import Series, DataFrame
import pandas as pd

import time

Dataset_originall=pd.read_csv('parkinson.csv')#ファイルの読み込み
Dataset=Dataset_originall.values#DataframeオブジェクトをNumpy配列に入れる



#42このデータのうち30個を学習に、12個をテストに使う
from numpy import random
training_person=random.choice(list(range(1,43)),30,replace=False)#重複なしに取り出す
training_person.sort()#ガウス過程で突っ込む際に順番に並んであった方がわかりやすいので
#まず各人のデータを区別して保存していきます
#
#
#

Training_Dataset=[]#トレーニングデータ用の人のデータをここに格納していきます
for i in training_person:
    X=Dataset_originall[Dataset_originall['subject#']==i]#subjectが1（1番の人）全体をdataframeそのものとして取り出す
    X1=[X.iloc[:,4].values,X.iloc[:,5].values,X.iloc[:,6:].values]#取り出したDataFrameからラベルｙと特徴量Xを取り出す
    Training_Dataset.append(X1)
#[[1番目の人のデータ]、[2番目の人のデータ]・・・・]とリストになっている
#[1番目の人のデータ]＝[array[yの値＊149個分]、array[y２の値*149]、array[Xの値[[149],[16]]と格納されている。

#testdataの方も格納していきます
test_person=set(range(1,43))-set(training_person)#test用の人の番号
test_person=list(test_person)
test_person.sort()
Test_Dataset=[]
for i in test_person:
    X=Dataset_originall[Dataset_originall['subject#']==i]#subjectが1（1番の人）全体をdataframeそのものとして取り出す
    X1=[X.iloc[:,4].values,X.iloc[:,5].values,X.iloc[:,6:].values]#取り出したDataFrameからラベルｙと特徴量Xを取り出す
    Test_Dataset.append(X1)

#    
#次に各人のデータを区別せずにまとめて入れます
#
training_person_set=set(training_person)

TRAININGDATA=DataFrame()
for i in training_person:
    TRAININGDATA =TRAININGDATA.add(Dataset_originall[Dataset_originall['subject#']==i],fill_value=0)
X_training=TRAININGDATA.iloc[:,6:].values
y1_training=TRAININGDATA.iloc[:,4].values
y2_training=TRAININGDATA.iloc[:,5].values
                           
                             
TESTDATA=DataFrame()
for i in test_person:
    TESTDATA =TESTDATA.add(Dataset_originall[Dataset_originall['subject#']==i],fill_value=0)
X_test=TESTDATA.iloc[:,6:].values
y1_test=TESTDATA.iloc[:,4].values
y2_test=TESTDATA.iloc[:,5].values

#特徴量の標準化 本来ならこうすべきだが、今回は全体の標準化量を計算する
def Scaler(X,mean,variance):
    return (X-mean)/variance

def Stan(X):
    mu=np.mean(X,axis=0)#平均
    var=np.sqrt(np.mean((X-mu)*(X-mu),axis=0))#分
    return mu,var
mu,var=Stan(X_training)

X_training_0=Scaler(X_training,mu,var)
X_test_0=Scaler(X_test,mu,var)

mu_y1,var_y1=Stan(y1_training)
mu_y2,var_y2=Stan(y2_training)
Y1_training_0=Scaler(y1_training,mu_y1,var_y1)
Y2_training_0=Scaler(y2_training,mu_y2,var_y2)
Y_training_0=np.concatenate((Y1_training_0[:,None],Y2_training_0[:,None]),1)


Y1_median=np.median(Y1_training_0)
Y2_median=np.median(Y2_training_0)

mu_y1t,var_y1t=Stan(y1_test)
Y1_test_0=y1_test-mu_y1#=Scaler(y1_test,mu_y1,var_y1)#Scaler(y1_test,mu_y1t,var_y1t)

mu_y2t,var_y2t=Stan(y2_test)
Y2_test_0=y2_test-mu_y2#Scaler(y2_test,mu_y2,var_y2)#Scaler(y2_test,mu_y2t,var_y2t)
Y_test_0=np.concatenate((Y1_test_0[:,None],Y2_test_0[:,None]),1)

#順番にそれぞれの人の中で正規化していきます
#考えた結果ｙの正規化はやめたのでした
for i in range(len(Training_Dataset)):
    Training_Dataset[i][2]=Scaler(Training_Dataset[i][2],mu,var)
    #Training_Dataset[i][1]=Scaler(Training_Dataset[i][1])
    #Training_Dataset[i][0]=Scaler(Training_Dataset[i][0])

N_p=len(Training_Dataset)#トレーニングに用いる確率分布（人の数）
N=len(X_training_0)#トレーニングに用いるすべてのデータの数

D=len(Training_Dataset)
l=[]
for i in range(len(Training_Dataset)):
    n=len(Training_Dataset[i][2])
    d=np.zeros((n,D))
    d[:,i]=1
    l.append(d)

Xlabel=l[0]
for i in range(1,len(Training_Dataset)):
    Xlabel=np.vstack((Xlabel,l[i]))
     
     
#test用の人にも同じようにやりましょう
for i in range(len(Test_Dataset)):
    Test_Dataset[i][2]=Scaler(Test_Dataset[i][2],mu,var)
    #Test_Dataset[i][1]=Scaler(Test_Dataset[i][1])
    #Test_Dataset[i][0]=Scaler(Test_Dataset[i][0])

N_p_t=len(Test_Dataset)#testに用いる確率分布（人の数）
N_t=len(X_test_0)#テストに用いるすべてのデータの数
   
mu_y1=np.mean(y1_training)
A=(y1_training-mu_y1).reshape(len(y1_training),1)
B=(y1_training-mu_y1).reshape(1,len(y1_training))
Cov_y1=A.dot(B)/len(y1_training)
  
mu_y2=np.mean(y2_training)
A=(y2_training-mu_y2).reshape(len(y2_training),1)
B=(y2_training-mu_y2).reshape(1,len(y2_training))
Cov_y2=A.dot(B)/len(y2_training)

N=3500
X=X_training_0[:N]
Y=Y_training_0[:N]

X_val=X_training_0[N:N+600]
Y_val=Y_training_0[N:N+600]

X_te=X_test_0
Y_te=Y_test_0


import theano

N=X.shape[0]
D=X.shape[1]
N_te=X_te.shape[0]

minibatch = np.random.choice(N,N,replace=False)
X=X[minibatch]
X=X.astype(theano.config.floatX)
Y=Y[minibatch]
Y=Y.astype(theano.config.floatX)
Domain_label=Xlabel[minibatch]
Domain_label=Domain_label.astype(theano.config.floatX)


minibatch_te = np.random.choice(N_te,N_te,replace=False)
X_te=X_te[minibatch_te]
X_te=X_te.astype(theano.config.floatX)
Y_te=Y_te[minibatch_te]
Y_te=Y_te.astype(theano.config.floatX)

N_val=X_val.shape[0]
minibatch_val = np.random.choice(N_val,N_val,replace=False)
X_val=X_training_0[minibatch_val]
Y_val=Y_training_0[minibatch_val]
X_val=X_val.astype(theano.config.floatX)
Y_val=Y_val.astype(theano.config.floatX)

#X = np.loadtxt('concrete_ARD_Xtrain__FOLD_1', delimiter=' ')
#Y = np.loadtxt('concrete_ARD_ytrain__FOLD_1', delimiter=' ')
#Y=Y[:,None]
#X_val = np.loadtxt('concrete_ARD_Xtrain__FOLD_2', delimiter=' ')
#Y_val = np.loadtxt('concrete_ARD_ytrain__FOLD_2', delimiter=' ')
#Y_val=Y_val[:,None]
#X_te = np.loadtxt('concrete_ARD_Xtest__FOLD_1', delimiter=' ')
#X_te2 = np.loadtxt('concrete_ARD_Xtest__FOLD_2', delimiter=' ')
#X_te = np.concatenate((X_te,X_te2))
#Y_te = np.loadtxt('concrete_ARD_ytest__FOLD_1', delimiter=' ')
#Y_te2 = np.loadtxt('concrete_ARD_ytest__FOLD_2', delimiter=' ')
#Y_te = np.concatenate((Y_te,Y_te2))
#Y_te=Y_te[:,None]

X = np.loadtxt('powerplant_ARD_Xtrain__FOLD_1', delimiter=' ')
Y = np.loadtxt('powerplant_ARD_ytrain__FOLD_1', delimiter=' ')
Y=Y[:,None]
X_val = np.loadtxt('powerplant_ARD_Xtrain__FOLD_2', delimiter=' ')
Y_val = np.loadtxt('powerplant_ARD_ytrain__FOLD_2', delimiter=' ')
Y_val=Y_val[:,None]
X_te = np.loadtxt('powerplant_ARD_Xtest__FOLD_1', delimiter=' ')
X_te2 = np.loadtxt('powerplant_ARD_Xtest__FOLD_2', delimiter=' ')
X_te = np.concatenate((X_te,X_te2))
Y_te = np.loadtxt('powerplant_ARD_ytest__FOLD_1', delimiter=' ')
Y_te2 = np.loadtxt('powerplant_ARD_ytest__FOLD_2', delimiter=' ')
Y_te = np.concatenate((Y_te,Y_te2))
Y_te=Y_te[:,None]

N=3500
X=X_training_0[:N]
Y=Y_training_0[:N]

X_val=X_training_0[N:N+600]
Y_val=Y_training_0[N:N+600]

X_te=X_test_0
Y_te=Y_test_0


import theano

N=X.shape[0]
D=X.shape[1]
N_te=X_te.shape[0]

minibatch = np.random.choice(N,N,replace=False)
X=X[minibatch]
X=X.astype(theano.config.floatX)
Y=Y[minibatch]
Y=Y.astype(theano.config.floatX)
Domain_label=Xlabel[minibatch]
Domain_label=Domain_label.astype(theano.config.floatX)


minibatch_te = np.random.choice(N_te,N_te,replace=False)
X_te=X_te[minibatch_te]
X_te=X_te.astype(theano.config.floatX)
Y_te=Y_te[minibatch_te]
Y_te=Y_te.astype(theano.config.floatX)

N_val=X_val.shape[0]
minibatch_val = np.random.choice(N_val,N_val,replace=False)
X_val=X_training_0[minibatch_val]
Y_val=Y_training_0[minibatch_val]
X_val=X_val.astype(theano.config.floatX)
Y_val=Y_val.astype(theano.config.floatX)

X=X.astype(theano.config.floatX)
Y=Y.astype(theano.config.floatX)
X_val=X_val.astype(theano.config.floatX)
Y_val=Y_val.astype(theano.config.floatX)
X_te=X_te.astype(theano.config.floatX)
Y_te=Y_te.astype(theano.config.floatX)

X_tr = theano.shared(X)
Y_tr = theano.shared(Y)
X_validate = theano.shared(X_val)
Y_validate = theano.shared(Y_val)
X_test = theano.shared(X_te)
Y_test = theano.shared(Y_te)
Xlabel_share = theano.shared(Domain_label)
Ydim=2

N=X.shape[0]
D=X.shape[1]
Ydim=Y.shape[1]

Domain_number=N_p
Hiddenlayerdim1,Hiddenlayerdim2=1,2
num_MC=10
Q=2 
n_rff=100
df=1
batch_size=500
batch_size2=100

DGP=Dgrff_opt(N,D,Q,Domain_number,Ydim,Hiddenlayerdim1,Hiddenlayerdim2,num_MC,n_rff,X_tr,Xlabel_share,Y_tr,batch_size,Y_validate,X_validate,Y_test,X_test,batch_size2)


def one_step_opt():
    n_valid_batches = X_validate.get_value(borrow=True).shape[0] // batch_size2
    n_test_batches = X_test.get_value(borrow=True).shape[0] // batch_size2
    epoch = 0
    max_epoch=1000
    start = time.time()
    n_train_batches = X.shape[0] // batch_size
    n_valid_batches = X_validate.get_value(borrow=True).shape[0] // batch_size2
    iteration=0
    param_saver={}
    while iteration < 1000000:
        #index=random.choice(n_train_batches)
        for index in range(n_train_batches):
            optming_params={}
            #alpha,beta=DG%edit P.dgrff.train_model(index,iteration,0.01,0)
            
            losses = [DGP.dgrff.train_model(i)[0]
                                     for i in range(n_train_batches)]
            alpha = np.mean(losses,0)
            #alpha,beta=DGP.dgrff.train_model(index)
            print ('iter ' + str(iteration) +'cost is _'+str(alpha))# +'error is _'+str(beta))
            if iteration%10==0:
                LL_Y=DGP.f['LL_Y'](index)
                #LL_X=DGP.f['LL_X'](index)
                KL_WX=DGP.f['KL_WX'](index)
                KL_WY=DGP.f['KL_WY'](index)
                #MMD=DGP.f['MMD'](index)
                #KL_hidden=DGP.f['KL_hidden'](index)
                
                print(': KL_WX is:' + str(KL_WX)+ ': KL_WY is:' + str(KL_WY))
                print ('and LL_Y is ' + str(LL_Y))#+'and MMD ' + str(MMD))
            
                #validation_losses = [DGP.validate_model(i) for i in range(n_valid_batches)]
                #this_validation_loss = np.mean(validation_losses)
                #print('current_loss is : _'+str(this_validation_loss))
             #   print('current_loss is : _'+str(beta))
                elapsed_time = time.time() - start
                print(elapsed_time)
                
                validation_losses = [DGP.validate_model(i)
                                     for i in range(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses,0)
                print ('validation_loss is _ ' + str(this_validation_loss))
                
                test_losses = [DGP.test_model(i) for i in range(n_test_batches)]
                test_score = np.mean(test_losses,0)
                print("test score is _"+str(test_score))
                
            iteration +=1
            #for i in dggplvm.wrt:
            #    optming_params[str(i)]=dggplvm.wrt[i].get_value()
            #optming_params['param_updates']=dggplvm.param_updates
            #optming_params['moving_mean_squared']=dggplvm.moving_mean_squared
            #optming_params['learning_rates']=dggplvm.learning_rates
            
            #param_saver[str(iteration)]=optming_params
                

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
    while iteration < 10000:
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

def early_stop_optimization(n_epochs=100000):
    n_train_batches = X.shape[0] // batch_size
    
    n_valid_batches = X_validate.get_value(borrow=True).shape[0] // batch_size2
    n_test_batches = X_test.get_value(borrow=True).shape[0] // batch_size2
                             
    #param_saver={}
    
    print('... training the model using early stopping')
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = np.inf
    test_score = 0.
    start_time = time.time()
    iteration=0
    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            DGP.dgrff.train_model(minibatch_index)
            
            iteration +=1
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [DGP.validate_model(i)
                                     for i in range(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )
                LL_Y=DGP.f['LL_Y'](minibatch_index)
                #LL_X=DGP.f['LL_X'](index)
                KL_WX=DGP.f['KL_WX'](minibatch_index)
                KL_WY=DGP.f['KL_WY'](minibatch_index)
                #KL_hidden=DGP.f['KL_hidden'](index)
                
                print(': KL_WX is:' + str(KL_WX)+ ': KL_WY is:' + str(KL_WY))
                print ('and LL_Y is ' + str(LL_Y))
                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                        improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [DGP.test_model(i)
                                   for i in range(n_test_batches)]
                    test_score = np.mean(test_losses)
                    print("test score is _"+str(test_score*100))
                    #print(
                    #    (
                    #        '     epoch %i, minibatch %i/%i, test error of'
                    #        ' best model %f %%'
                    #    ) 
                    #    (
                    #        epoch,
                    #        minibatch_index + 1,
                     #       n_train_batches,
                            #test_score * 100.
                      #  )
                    #)

                    # save the best model
                    #with open('best_model.pkl', 'wb') as f:
                    #    pickle.dump(classifier, f)

            if patience <= iter:
                done_looping = True
                break

    end_time = time.time()
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print('The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time)))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)
if __name__ == '__main__':
    one_step_opt()
    #early_stop_optimization(n_epochs=100000)

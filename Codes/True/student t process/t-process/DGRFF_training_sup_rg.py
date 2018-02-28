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

import theano

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
Q=3 
n_rff=100
df=1
batch_size=500
batch_size2=40
free_param=10

#DGP=Dgrff_opt(N,D,Q,Domain_number,Ydim,Hiddenlayerdim1,Hiddenlayerdim2,num_MC,n_rff,X_tr,Xlabel_share,Y_tr,batch_size,Y_validate,X_validate,Y_test,X_test,batch_size2,free_param)


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
    while iteration < 50000:
        #index=random.choice(n_train_batches)
        for index in range(n_train_batches):
            optming_params={}
            #alpha,beta=DG%edit P.dgrff.train_model(index,iteration,0.01,0)
            
            #losses = [DGP.dgrff.train_model(i)[0]
            #                         for i in range(n_train_batches)]
            #alpha = np.mean(losses,0)
            alpha,beta=DGP.dgrff.train_model(index)
            print ('iter ' + str(iteration) +'cost is _'+str(alpha) +'error is _'+str(beta))
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
                
            #    validation_losses = [DGP.validate_model(i)
             #                        for i in range(n_valid_batches)]
              #  this_validation_loss = np.mean(validation_losses,0)
                ##print ('validation_loss is _ ' + str(this_validation_loss))
                
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
#if __name__ == '__main__':
#    one_step_opt()
    #early_stop_optimization(n_epochs=100000)

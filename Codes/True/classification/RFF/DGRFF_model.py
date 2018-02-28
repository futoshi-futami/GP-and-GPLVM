# To speed Theano up, create ram disk: mount -t tmpfs -o size=512m tmpfs /mnt/ramdisk
# Then use flag THEANO_FLAGS='base_compiledir=/mnt/ramdisk' python script.py
import sys; sys.path.insert(0, "../Theano"); sys.path.insert(0, "../../Theano")
import theano; import theano.tensor as T; import theano.sandbox.linalg as sT
import numpy as np

import lasagne

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from mlp import HiddenLayer
from RFF_layer import RFFLayer

rng = np.random.RandomState(1234)

print ('Theano version: ' + theano.__version__ + ', base compile dir: ' + theano.config.base_compiledir)
#theano.config.mode = 'FAST_RUN'
#theano.config.optimizer = 'fast_run'
#theano.config.reoptimize_unpickled_function = False

    
class Dgrff_model:
    def __init__(self,N_tot,D,Q,Domain_number,Ydim,Hiddenlayerdim1,Hiddenlayerdim2,num_MC,n_rff):
        ########################################
        #BCなXの設定　後でこれもレイヤー化する
        self.Xlabel=T.matrix('Xlabel')

        
        self.X=T.matrix('X')
        self.Y=T.matrix('Y')
        N=self.X.shape[0]
        
        self.Weight=T.matrix('Weight')
        
        self.hiddenLayer_x = HiddenLayer(rng=rng,input=self.X,n_in=D,n_out=Hiddenlayerdim1,activation=T.nnet.relu,number='_x')
        #self.hiddenLayer_hidden = HiddenLayer(rng=rng,input=self.hiddenLayer_x.output,n_in=Hiddenlayerdim1,n_out=Hiddenlayerdim2,activation=T.nnet.relu,number='_h')
        self.hiddenLayer_m = HiddenLayer(rng=rng,input=self.hiddenLayer_x.output,n_in=Hiddenlayerdim2,n_out=Q,activation=T.nnet.relu,number='_m')
        self.hiddenLayer_S = HiddenLayer(rng=rng,input=self.hiddenLayer_x.output,n_in=Hiddenlayerdim2,n_out=Q,activation=T.nnet.relu,number='_S')
        
        self.loc_params= []
        self.loc_params.extend(self.hiddenLayer_x.params)
        #self.loc_params.extend(self.hiddenLayer_hidden.params)
        self.loc_params.extend(self.hiddenLayer_m.params)
        self.loc_params.extend(self.hiddenLayer_S.params)

        self.local_params={}
        for i in self.loc_params:
            self.local_params[str(i)]=i
        
        ########################
        #hiddenlyaerの作成
        srng = RandomStreams(seed=234)
        sample_Hidden_layer = srng.normal((num_MC,N,Q))
        S_1=T.exp(self.hiddenLayer_S.output)
        S=T.sqrt(S_1)
        Hidden_variable=((self.hiddenLayer_m.output)[None,:,:])+sample_Hidden_layer*(S[None,:,:])
        
        ##########################################
        ####X側の推論
        
        self.RFF_X=RFFLayer(rng, Hidden_variable, n_in=Q, n_out=D, num_MC=num_MC,num_FF=n_rff,Domain_number=Domain_number,number="X",Domain_consideration=True)
        
        self.params = self.RFF_X.all_params
        self.hyp_params=self.RFF_X.hyp_params
        self.variational_params=self.RFF_X.variational_params
        ##############################################################################################
        ###Y側の計算
        self.RFF_Y=RFFLayer(rng, Hidden_variable, n_in=Q, n_out=Ydim, num_MC=num_MC,num_FF=n_rff,number="Y",Domain_consideration=False)
   
        self.params.extend(self.RFF_Y.all_params)
        self.hyp_params.extend(self.RFF_Y.hyp_params)
        self.variational_params.extend(self.RFF_Y.variational_params)
        
        ##########################################
        #パラメータの格納   
        #self.no_updates=self.RFF_X.no_update
        
        self.variational_params.extend(self.loc_params)
        
        self.params.extend(self.loc_params)
        
        self.wrt={}
        for i in self.params:
            self.wrt[str(i)]=i

        ###########################################
        ###目的関数
        #############X側
        
        self.LL_X = self.RFF_X.likelihood_domain(self.X,self.Xlabel)*N_tot/(N*num_MC)
        self.KL_WX = self.RFF_X.KL_W        
        
        #############Y側
        self.LL_Y =  self.RFF_Y.classification_liklihood(self.Y)*N_tot/(N*num_MC)
        self.KL_WY = self.RFF_Y.KL_W
        #y=self.Gaussian_layer_Y.softmax_class()
        #self.LLY = -T.mean(T.nnet.categorical_crossentropy(y, self.Y))*N
        #############真ん中と予測
        self.KL_hidden = self.KLD_X(self.hiddenLayer_m.output,S_1)
        
        self.error = self.RFF_Y.error_classification(self.Y)
        ###########################################
        #self.MMD=self.Gaussian_layer_Y.MMD_central_penalty(self.Y,self.Xlabel)
        
        
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

    
    def prediction_validation(self,Y_validate,X_validate,Y_test,X_test,batch_size):
        
        index = T.iscalar()

        self.test_model = theano.function(
        inputs=[index],
        outputs=self.error,
        givens={
                self.X: X_test[index * batch_size: (index + 1) * batch_size],
                self.Y: Y_test[index * batch_size: (index + 1) * batch_size]
                },on_unused_input='ignore'
            )

        self.validate_model = theano.function(
        inputs=[index],
        outputs=self.error,
        givens={
            self.X: X_validate[index * batch_size: (index + 1) * batch_size],
            self.Y: Y_validate[index * batch_size: (index + 1) * batch_size]
                },on_unused_input='ignore'
            )
        
    def lasagne_optimizer(self,train_set_x,train_set_y,train_label,batch_size):
        
        index = T.lscalar()

        print ('Modeling...')
        
        loss_0 = self.LL_X + self.LL_Y - self.KL_WX - self.KL_WY - self.KL_hidden + 0.0*sum([T.sum(v) for v in self.params])   
        loss=T.cast(loss_0,'float32')
        updates = lasagne.updates.adam(-loss, self.params, learning_rate=0.01)
        #updates = lasagne.updates.apply_momentum(updates, self.params, momentum=0.9)
        
        self.train_model=theano.function(
                [index],
                outputs=loss,
                givens={
                self.X: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.Xlabel: train_label[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.Y: train_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]},
                on_unused_input='ignore',
                updates=updates
            )
            
        self.f = {n: theano.function([index], f, name=n, 
                givens={
                self.X: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.Xlabel: train_label[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.Y: train_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
                },on_unused_input='ignore') 
                for n,f in zip(['LL_X','LL_Y','KL_WX', 'KL_WY','KL_hidden'], [self.LL_X,self.LL_Y,self.KL_WX,self.KL_WY,self.KL_hidden])}            
        
    def cal_check(self,train_set_x,train_set_y,train_label,batch_size):
        
        index = T.iscalar()

        print ('Modeling...')
        
        loss = self.LL_X + self.LL_Y - self.KL_hidden# - self.KL_WX - self.KL_WY  + 0.0*sum([T.sum(v) for v in self.params])   
        
        updates = lasagne.updates.rmsprop(-loss, self.params, learning_rate=0.0001)
        #updates = lasagne.updates.apply_momentum(updates, self.params, momentum=0.9)
        
        self.train_model=theano.function(
                [index],
                outputs=loss,#self.LL_X + self.LL_Y - self.KL_WX - self.KL_WY - self.KL_hidden + 0.0*sum([T.sum(v) for v in self.params]),
                givens={
                self.X: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.Xlabel: train_label[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.Y: train_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]},
                on_unused_input='ignore',
                updates=updates
                #no_default_updates=self.no_updates
            )
            
    def lasagne_optimizer2(self,train_set_x,train_set_y,train_label,batch_size):
        
        index = T.lscalar()
        iteration = T.lscalar()
        a=T.fscalar()
        b=T.fscalar()
        print ('Modeling...')
        
        loss_0 = self.LL_X + self.LL_Y - self.KL_WX - self.KL_WY - self.KL_hidden + 0.0*sum([T.sum(v) for v in self.params])   
        loss=T.cast(loss_0,theano.config.floatX)
        
        from theano.ifelse import ifelse

        #params = ifelse(T.gt(iteration, 1000), self.params, self.params)
        gparams = T.grad(-loss, self.variational_params)
        
        updates1 = lasagne.updates.adam(gparams, self.variational_params, learning_rate=0.01)
        #updates1 = lasagne.updates.apply_momentum(updates1, self.variational_params, momentum=0.9)
        
        gparams2 = T.grad(-loss, [self.RFF_X.lhyp,self.RFF_X.ls,self.RFF_Y.lhyp])
        
        updates2 = lasagne.updates.adam(gparams2, [self.RFF_X.lhyp,self.RFF_X.ls,self.RFF_Y.lhyp], learning_rate = ifelse(T.gt(iteration, 2000), a, b))
        #updates2 = lasagne.updates.apply_momentum(updates2, [self.RFF_X.lhyp,self.RFF_X.ls,self.RFF_Y.lhyp,self.RFF_Y.ls], momentum=0.9)
        print(updates2)
        
        updates2.update(updates1)
        print('merge')
        print(updates2)
        
        
        self.train_model=theano.function(
                [index,iteration,a,b],
                outputs=loss,
                givens={
                self.X: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.Xlabel: train_label[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.Y: train_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]},
                on_unused_input='ignore',
                updates=updates2,
                allow_input_downcast=True
            )
            
        self.f = {n: theano.function([index], f, name=n, 
                givens={
                self.X: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.Xlabel: train_label[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.Y: train_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
                },on_unused_input='ignore') 
                for n,f in zip(['LL_X','LL_Y','KL_WX', 'KL_WY','KL_hidden'], [self.LL_X,self.LL_Y,self.KL_WX,self.KL_WY,self.KL_hidden])}            

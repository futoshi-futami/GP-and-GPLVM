# To speed Theano up, create ram disk: mount -t tmpfs -o size=512m tmpfs /mnt/ramdisk
# Then use flag THEANO_FLAGS='base_compiledir=/mnt/ramdisk' python script.py
import sys; sys.path.insert(0, "../Theano"); sys.path.insert(0, "../../Theano")
import theano; import theano.tensor as T; import theano.sandbox.linalg as sT

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import numpy as np

from mlp import HiddenLayer
from kernel_layer_rff import KernelLayer

rng = np.random.RandomState(1234)

print ('Theano version: ' + theano.__version__ + ', base compile dir: ' + theano.config.base_compiledir)
theano.config.mode = 'FAST_RUN'
#theano.config.optimizer = 'fast_run'
#theano.config.reoptimize_unpickled_function = False




class DGGPLVM_model:
    def __init__(self,N_tot,D,M,Q,Domain_number,M_Y,Ydim,num_MC,Hiddenlayerdim1,Hiddenlayerdim2):
        ########################################
        # set type
        self.Xlabel=T.matrix('Xlabel')
        self.X=T.matrix('X')
        self.Y=T.matrix('Y')
        self.Weight=T.matrix('Weight')
                
        N=self.X.shape[0]
        #############################################
        #BCなXの設定　後でこれもレイヤー化する  MCsample 分を生成することにします。
        #when we use the back constrained model....
        #srng = RandomStreams(seed=234)
        #eps_NQ = srng.normal((num_MC,self.N,Q))
        #BCでもとめた平均と分散を入力します。対角成分のみのガウス分布とします。
        #self.input_X = eps_NQ * S[None,:,:] + m[None,:,:]

        #普通のsupervised な場合 MCサンプル分コピーしときます。
        self.Data_input=T.tile(self.X,(num_MC,1,1))
        
        ##########################################
        ####X側の推論
        self.Gaussian_layer_X=KernelLayer(rng,self.Data_input, D,M,Domain_number,num_MC,Domain_consideration=True,number='_X')

        self.params = self.Gaussian_layer_X.params
        self.Z_params_list=self.Gaussian_layer_X.Z_params_list
        self.global_param_list=self.Gaussian_layer_X.global_params_list
        self.hyp_list=self.Gaussian_layer_X.hyp_params_list
        
        self.hidden_layer=self.Gaussian_layer_X.output
        
        ##############################################################################################
        ###Y側の計算
        self.Gaussian_layer_Y=KernelLayer(rng,self.hidden_layer,Q, Ydim,M_Y,liklihood='Gaussian',Domain_consideration=False,number='_Y',kernel_name='Y')
        
        self.params.extend(self.Gaussian_layer_Y.params)
        self.Z_params_list.extend(self.Gaussian_layer_Y.Z_params_list)
        self.global_param_list.extend(self.Gaussian_layer_Y.global_params_list)
        self.hyp_list.extend(self.Gaussian_layer_Y.hyp_params_list)

        ###########################################
        ###目的関数
        
        self.LL = self.Gaussian_layer_X.likelihood_domain(self.X,self.Xlabel)
        self.KL_X = self.Gaussian_layer_X.KL_X
        self.KL_U = self.Gaussian_layer_X.KL_U
        self.KL_UY=self.Gaussian_layer_Y.KL_U
        #y=self.Gaussian_layer_Y.softmax_class()
        #self.LLY = -T.mean(T.nnet.categorical_crossentropy(y, self.Y))*N
        #self.LLY=T.sum(T.log(T.maximum(T.sum(self.Y * y, 1), 1e-16)))
        #self.error = self.Gaussian_layer_Y.error_classification(self.Y)
        
        
        ###########################################
        #domain checker MMD と　クラス分類        
        self.MMD=self.Gaussian_layer_Y.MMD_class_penalty(self.Y,self.Xlabel)
                
        ##########################################
        #パラメータの格納
        self.hyp_params={}
        for i in self.hyp_list:
            self.hyp_params[str(i)]=i
        
        self.Z_params={}
        for i in self.Z_params_list:
            self.Z_params[str(i)]=i
                         
        self.global_params={}
        for i in self.global_param_list:
            self.global_params[str(i)]=i        
        
        self.params.extend(self.loc_params)
        
        self.wrt={}
        for i in self.params:
            self.wrt[str(i)]=i

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
        
        loss_0 = self.LL_Y - self.KL_WX - self.KL_WY  + 0.0*sum([T.sum(v) for v in self.params]) -0.1*self.MMD
        loss=T.cast(loss_0,theano.config.floatX)
        updates = lasagne.updates.rmsprop(-loss, self.params, learning_rate=0.001)
        #updates = lasagne.updates.apply_momentum(updates, self.params, momentum=0.9)
        
        self.train_model=theano.function(
                [index],
                outputs=[loss,self.error],#[self.RFF_X.mean_mu,self.RFF_Y.mean_mu],
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
                for n,f in zip(['LL_Y','KL_WX', 'KL_WY','MMD'], [self.LL_Y,self.KL_WX,self.KL_WY,self.MMD])}                 
        
    def cal_check(self,train_set_x,train_set_y,train_label,batch_size):
        
        index = T.iscalar()

        print ('Modeling...')
        
        loss = self.LL_Y  - self.KL_WX - self.KL_WY-self.MMD  + 0.0*sum([T.sum(v) for v in self.params])   
        
        updates = lasagne.updates.adam(-loss, self.params, learning_rate=0.01)
        #updates = lasagne.updates.apply_momentum(updates, self.params, momentum=0.9)
        
        self.train_model_checker=theano.function(
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
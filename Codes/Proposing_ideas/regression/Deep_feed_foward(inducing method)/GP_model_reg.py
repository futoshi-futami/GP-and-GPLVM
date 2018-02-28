# To speed Theano up, create ram disk: mount -t tmpfs -o size=512m tmpfs /mnt/ramdisk
# Then use flag THEANO_FLAGS='base_compiledir=/mnt/ramdisk' python script.py
import sys,os
sys.path.append("./")
sys.path.append("../")
sys.path.append(os.pardir)
sys.path.insert(0, "../Theano")
sys.path.insert(0, "../../Theano")


import theano; import theano.tensor as T; import theano.sandbox.linalg as sT

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import lasagne
import numpy as np

from mlp import HiddenLayer
from kernel_layer_rff import KernelLayer
from kernel_layer_inducing_fix import KernelLayer_fix
rng = np.random.RandomState(1234)

print ('Theano version: ' + theano.__version__ + ', base compile dir: ' + theano.config.base_compiledir)
#theano.config.mode = 'FAST_RUN'
#theano.config.optimizer = 'fast_run'
#theano.config.reoptimize_unpickled_function = False




class GP_model:
    def __init__(self,N_tot,D_in,D_out,M,Domain_number,dim_int,Hiddenlayerdim1,Hiddenlayerdim2,num_MC,inducing_points):
        ########################################
        # set type
        self.Xlabel=T.matrix('Xlabel')
        self.X=T.matrix('X')
        self.Y=T.matrix('Y')
        self.Weight=T.matrix('Weight')
                
        N=self.X.shape[0]
        #############################################

        self.Data_input=T.tile(self.X,(num_MC,1,1))        
        ##########################################
        ####X側の推論
        
        self.Gaussian_layer_X=KernelLayer_fix(self.Data_input, D_in=D_in, D_out=dim_int,num_MC=num_MC,inducing_number=M,fixed_z=inducing_points,Domain_number=None,Domain_consideration=False,number='_X')

        self.params = self.Gaussian_layer_X.params
        self.Z_params_list=self.Gaussian_layer_X.Z_params_list
        self.global_param_list=self.Gaussian_layer_X.global_params_list
        self.hyp_list=self.Gaussian_layer_X.hyp_params_list
        
        self.hidden_layer=self.Gaussian_layer_X.output
        
        ##############################################################################################
        ###出力層の計算
        self.Gaussian_layer_Y=KernelLayer(self.hidden_layer,D_in=dim_int, D_out=D_out,num_MC=num_MC,inducing_number=M,Domain_number=None,Domain_consideration=False,number='_Y',kernel_name='Y')
        
        self.params.extend(self.Gaussian_layer_Y.params)
        self.Z_params_list.extend(self.Gaussian_layer_Y.Z_params_list)
        self.global_param_list.extend(self.Gaussian_layer_Y.global_params_list)
        self.hyp_list.extend(self.Gaussian_layer_Y.hyp_params_list)

        ###########################################
        ###目的関数
        
        self.LL = self.Gaussian_layer_Y.liklihood_nodomain(self.Y)*N_tot/N
        #self.KL_X = self.Gaussian_layer_X.KL_X
        self.KL_UX = self.Gaussian_layer_X.KL_U
        self.KL_UY = self.Gaussian_layer_Y.KL_U
        #y=self.Gaussian_layer_Y.softmax_class()
        #self.LLY = -T.mean(T.nnet.categorical_crossentropy(y, self.Y))*N
        #self.LLY=T.sum(T.log(T.maximum(T.sum(self.Y * y, 1), 1e-16)))
        #self.error = self.Gaussian_layer_Y.error_classification(self.Y)
        
        pred = T.mean(self.Gaussian_layer_Y.output,0)
        self.error = (T.mean((self.Y - pred)**2,0))**0.5
        
        
        ###########################################
        #domain checker MMD と　クラス分類        
        #self.MMD=self.Gaussian_layer_Y.MMD_class_penalty(self.Y,self.Xlabel)
                
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
        
        #self.params.extend(self.loc_params)
        
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
        
        loss_0 = self.LL - self.KL_UY + 0.0*sum([T.sum(v) for v in self.params])- self.KL_UX# -0.1*self.MMD
        loss=T.cast(loss_0,theano.config.floatX)
        updates = lasagne.updates.rmsprop(-loss, self.params, learning_rate=0.001)
        updates = lasagne.updates.apply_momentum(updates, self.params, momentum=0.9)
        
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
                for n,f in zip(['LL','KL_UX','KL_UY'], [self.LL,self.KL_UX,self.KL_UY])}                 
        
    def cal_check(self,train_set_x,train_set_y,train_label,batch_size):
        
        index = T.iscalar()

        print ('Modeling...')
        
        loss = self.LL  - self.KL_UX- self.KL_UY+ 0.0*sum([T.sum(v) for v in self.params])   
        
        updates = lasagne.updates.adam(-loss, self.params, learning_rate=0.01)
        #updates = lasagne.updates.apply_momentum(updates, self.params, momentum=0.9)
        
        self.train_model_checker=theano.function(
                [index],
                outputs=self.LL,#self.LL_X + self.LL_Y - self.KL_WX - self.KL_WY - self.KL_hidden + 0.0*sum([T.sum(v) for v in self.params]),
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
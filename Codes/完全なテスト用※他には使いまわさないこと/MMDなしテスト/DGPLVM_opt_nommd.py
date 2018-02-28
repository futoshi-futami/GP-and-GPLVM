from DGPLVM_model_nommd import DGGPLVM_model
import sys; sys.path.append("./")
import numpy as np
from copy import deepcopy
import time

class DGGPLVM_opt:
    def __init__(self, D, M,Q,Domain_number,train_set_x,train_set_y,train_weight,train_label,batch_size,D_Y,M_Y):
        self.dggplvm = DGGPLVM_model( D, M,Q,Domain_number,D_Y,M_Y)
        self.wrt = self.dggplvm.wrt
        
        self.dggplvm.compile_F(train_set_x,train_set_y,train_weight,train_label,batch_size)
        self.f = self.dggplvm.f

        self.estimate = self.dggplvm.estimate
        self.estimateY = self.dggplvm.estimateY
        
        self.callback_counter = [0]
        self.print_interval = 10
        
        self.correct=train_set_x.get_value().shape[0]/batch_size
        
        #RMSPROPのため
        self.param_updates = {n: np.zeros_like(v.get_value(borrow=True)) for n, v in self.wrt.items()}#update用の同じサイズの空の箱を用意
                              
        self.moving_mean_squared = {n: np.zeros_like(v.get_value(borrow=True)) for n, v in self.wrt.items()}#ＲＭＳＰＲＯＰ用に更新の履歴（γを入れておく器をパラメータごとに用意
                                    
        self.learning_rates = {n: 1e-2*np.ones_like(v.get_value(borrow=True)) for n, v in self.wrt.items()}#行進用の同じサイズの箱を用意                              

    def get_grad_Y(self, param_name, index):
        grad1,grad_std=self.estimateY(param_name,index)
        grad = self.dggplvm.g[param_name]['KL_UY']() + grad1*self.correct
                                  
        # DEBUG
        if param_name == 'lhyp_Y' and np.any(np.abs(grad) < grad_std / np.sqrt(50)):
                #print 'Large noise, recomputing. lhyp grad mean:', grad, ', std:', grad_std / np.sqrt(self.clgp.samples)
            grad_ls, grad_std = self.estimateY(param_name,index,300) 
            grad = self.dggplvm.g[param_name]['KL_UY']() + grad_ls
            self.grad_std = grad_std

        return np.array(grad)
    
    def get_grad_X(self, param_name, index):
        grad1,grad_std=self.estimate(param_name,index)
        grad = self.dggplvm.g[param_name]['KL_U']() + grad1*self.correct
                                  
        # DEBUG
        if param_name == 'lhyp' and np.any(np.abs(grad) < grad_std / np.sqrt(50)):
                #print 'Large noise, recomputing. lhyp grad mean:', grad, ', std:', grad_std / np.sqrt(self.clgp.samples)
            grad_ls, grad_std = self.estimate(param_name,index,300) 
            grad = self.dggplvm.g[param_name]['KL_U']() + grad_ls
            self.grad_std = grad_std

        return np.array(grad)
    
    def get_grad_local(self, param_name, index):
        #wrt = {'Z': Z, 'm': m, 'S_b': S_b, 'mu': mu, 'Sigma_b': Sigma_b, 'lhyp': lhyp, 'ls': ls, 'KmmInv': KmmInv}               
        grad =  self.dggplvm.g[param_name]['KL_X']() + (self.estimate(param_name,index)[0]+self.estimateY(param_name,index)[0])*self.correct
                              
        return np.array(grad)

    def opt_one_step(self, iteration,index, opt = 'rmsprop', learning_rate_adapt = 0.2, use_einsum = True):        
        
        for param_name in self.dggplvm.local_params:
            self.rmsprop_one_step_local(param_name, index, [param_name, index], learning_rate_adapt = learning_rate_adapt)#, momentum = 0.9 - 0.4 * 100 / (iteration + 100.0))
            
        for param_name in  self.dggplvm.global_params_X:
            self.rmsprop_one_step_globalX(param_name, index, [param_name, index], learning_rate_adapt = learning_rate_adapt)#, momentum = 0.9 - 0.4 * 100 / (iteration + 100.0))

        for param_name in  self.dggplvm.global_params_Y:
            self.rmsprop_one_step_globalY(param_name, index, [param_name, index], learning_rate_adapt = learning_rate_adapt)#, momentum = 0.9 - 0.4 * 100 / (iteration + 100.0))

    
    def grad_ascent_one_step(self, param_name, minibatch, grad_args, momentum = 0.9, learning_rate_decay = 1):
        #grad_args=[param_name, self.Y, KmmInv_grad, self.mask]
        self.dggplvm.params[param_name][minibatch] += (learning_rate_decay*self.learning_rates[param_name][minibatch]* self.param_updates[param_name][minibatch])
        grad = self.get_grad(*grad_args)
        self.param_updates[param_name][minibatch] = grad
                         
    def rmsprop_one_step_local(self, param_name, index, grad_args, decay = 0.9, momentum = 0, learning_rate_adapt = 0.05, 
        learning_rate_min = 1e-6, learning_rate_max = 10):
        # RMSPROP: Tieleman, T. and Hinton, G. (2012), Lecture 6.5 - rmsprop, COURSERA: Neural Networks for Machine Learning
        # Implementation based on https://github.com/BRML/climin/blob/master/climin/rmsprop.py
        
        # We use Nesterov momentum: first, we make a step according to the momentum and then we calculate the gradient.
        step1 = self.param_updates[param_name] * momentum
        self.wrt[param_name].set_value(self.wrt[param_name].get_value(borrow=True)+step1,borrow=True)
        grad = self.get_grad_local(*grad_args)

        self.moving_mean_squared[param_name] = (decay * self.moving_mean_squared[param_name] + (1 - decay) * grad ** 2)
        step2 = self.learning_rates[param_name] * grad / (self.moving_mean_squared[param_name] + 1e-8)**0.5

        # DEBUG
        if param_name == 'lhyp':
            step2 = np.clip(step2, -0.1, 0.1)
        
        self.wrt[param_name].set_value(self.wrt[param_name].get_value(borrow=True)+step2,borrow=True)
        #self.params[param_name] += step2

        step = step1 + step2

        # Step rate adaption. If the current step and the momentum agree, we slightly increase the step rate for that dimension.
        if learning_rate_adapt:
            # This code might look weird, but it makes it work with both numpy and gnumpy.
            step_non_negative = step > 0
            step_before_non_negative = self.param_updates[param_name] > 0
            agree = (step_non_negative == step_before_non_negative) * 1.#０か１が出る
            adapt = 1 + agree * learning_rate_adapt * 2 - learning_rate_adapt
            self.learning_rates[param_name] *= adapt
            self.learning_rates[param_name] = np.clip(self.learning_rates[param_name], learning_rate_min, learning_rate_max)

        self.param_updates[param_name] = step

    def rmsprop_one_step_globalX(self, param_name, index, grad_args, decay = 0.9, momentum = 0, learning_rate_adapt = 0.05, 
        learning_rate_min = 1e-6, learning_rate_max = 10):
        # RMSPROP: Tieleman, T. and Hinton, G. (2012), Lecture 6.5 - rmsprop, COURSERA: Neural Networks for Machine Learning
        # Implementation based on https://github.com/BRML/climin/blob/master/climin/rmsprop.py
        
        # We use Nesterov momentum: first, we make a step according to the momentum and then we calculate the gradient.
        step1 = self.param_updates[param_name] * momentum
        self.wrt[param_name].set_value(self.wrt[param_name].get_value(borrow=True)+step1,borrow=True)
        grad = self.get_grad_X(*grad_args)

        self.moving_mean_squared[param_name] = (decay * self.moving_mean_squared[param_name] + (1 - decay) * grad ** 2)
        step2 = self.learning_rates[param_name] * grad / (self.moving_mean_squared[param_name] + 1e-8)**0.5

        # DEBUG
        if param_name == 'lhyp':
            step2 = np.clip(step2, -0.1, 0.1)
        
        self.wrt[param_name].set_value(self.wrt[param_name].get_value(borrow=True)+step2,borrow=True)
        #self.params[param_name] += step2

        step = step1 + step2

        # Step rate adaption. If the current step and the momentum agree, we slightly increase the step rate for that dimension.
        if learning_rate_adapt:
            # This code might look weird, but it makes it work with both numpy and gnumpy.
            step_non_negative = step > 0
            step_before_non_negative = self.param_updates[param_name] > 0
            agree = (step_non_negative == step_before_non_negative) * 1.#０か１が出る
            adapt = 1 + agree * learning_rate_adapt * 2 - learning_rate_adapt
            self.learning_rates[param_name] *= adapt
            self.learning_rates[param_name] = np.clip(self.learning_rates[param_name], learning_rate_min, learning_rate_max)

        self.param_updates[param_name] = step

    def rmsprop_one_step_globalY(self, param_name, index, grad_args, decay = 0.9, momentum = 0, learning_rate_adapt = 0.05, 
        learning_rate_min = 1e-6, learning_rate_max = 10):
        # RMSPROP: Tieleman, T. and Hinton, G. (2012), Lecture 6.5 - rmsprop, COURSERA: Neural Networks for Machine Learning
        # Implementation based on https://github.com/BRML/climin/blob/master/climin/rmsprop.py
        
        # We use Nesterov momentum: first, we make a step according to the momentum and then we calculate the gradient.
        step1 = self.param_updates[param_name] * momentum
        self.wrt[param_name].set_value(self.wrt[param_name].get_value(borrow=True)+step1,borrow=True)
        grad = self.get_grad_Y(*grad_args)

        self.moving_mean_squared[param_name] = (decay * self.moving_mean_squared[param_name] + (1 - decay) * grad ** 2)
        step2 = self.learning_rates[param_name] * grad / (self.moving_mean_squared[param_name] + 1e-8)**0.5

        # DEBUG
        if param_name == 'lhyp':
            step2 = np.clip(step2, -0.1, 0.1)
        
        self.wrt[param_name].set_value(self.wrt[param_name].get_value(borrow=True)+step2,borrow=True)
        #self.params[param_name] += step2

        step = step1 + step2

        # Step rate adaption. If the current step and the momentum agree, we slightly increase the step rate for that dimension.
        if learning_rate_adapt:
            # This code might look weird, but it makes it work with both numpy and gnumpy.
            step_non_negative = step > 0
            step_before_non_negative = self.param_updates[param_name] > 0
            agree = (step_non_negative == step_before_non_negative) * 1.#０か１が出る
            adapt = 1 + agree * learning_rate_adapt * 2 - learning_rate_adapt
            self.learning_rates[param_name] *= adapt
            self.learning_rates[param_name] = np.clip(self.learning_rates[param_name], learning_rate_min, learning_rate_max)

        self.param_updates[param_name] = step


#########################################################################################################################################S
###L-BFGSでの最適化

    def unpack(self, x):
        x_param_values = [x[self.sizes[i-1]:self.sizes[i]].reshape(self.shapes[i-1]) for i in range(1,len(self.shapes)+1)]#これは['m', 'ls', 'lhyp', 'S', 'Z']の5つ分
        #それぞれのパラメータについてリストで与えられたとしても、例えばｍの場合i=1よってx[sizes[0]:sizes[1]]つまり適切な数のx[ｍの始まり:mの終わり]が出され、それがreshape((ｍのサイズ))で変換される
        params = {n:v for (n,v) in zip(self.opt_param_names, x_param_values)}
        if 'lhyp' in params:
            params['lhyp']=params['lhyp'].squeeze()
        
        if 'ls' in params:
            params['ls']=params['ls'].reshape(1) 

        return params
    
    def _convert_to_array(self, params):
        return np.hstack((params['Z'].flatten(),params['m'].flatten(),params['S_b'].flatten(),params['mu'].flatten(),params['Sigma_b'].flatten(),params['lhyp'].flatten(),params['ls'].flatten()))
    #'Z', 'm', 'S_b', 'mu', 'Sigma_b', 'lhyp', 'ls'
    def _optimizer_f(self, hypInArray):
        params=self.unpack(hypInArray)
        self.params=params
        cost=self.ELBO(self.X,self.N)
        return -cost[0]
    
    def _optimizer_g(self, hypInArray):
        params=self.unpack(hypInArray)
        self.params=params
        gradient=[]
        minibatch = np.arange(self.N)
        for i in self.opt_param_names:
            g = self.get_grad(i, self.X, minibatch)
            gradient=np.hstack((gradient,g.flatten()))        
        return gradient
    
    def train_by_optimizer(self,batch_size=None):
        
        print ('start to optimize')
        likelihood = self.dggplvm.ELBO(self.X,self.N)
        print ('BEGINE Training, Log Likelihood = %.2f'% likelihood[0])
        #import minimize 
        #opt_results = minimize.run(self._optimizer_f, self._get_hypArray(params),length=number_epoch,verbose=True)
        init=[]
        from scipy.optimize import minimize
        
        init=self._convert_to_array(self.params)
            
        opt_results = minimize(self._optimizer_f, init, method='L-BFGS-B', jac=self._optimizer_g, options={'ftol':0 , 'disp':True, 'maxiter': 500}, tol=0,callback=self.callback)
        optimalHyp = deepcopy(opt_results.x)
        hype=self.unpack(optimalHyp)
        self.params=hype
        
        likelihood = self.dggplvm.ELBO(self.X,self.N)
        print ('END Training, Log Likelihood = %.2f'% likelihood[0])
        
    def callback(self, x):
        #インターバル毎にコールバックが出る
        if self.callback_counter[0]%self.print_interval == 0:
            opt_params = self.unpack(x)
            self.params=opt_params
            cost=self.ELBO(self.X,self.N)
            print ('iter ' + str(self.callback_counter) + ': ' + str(cost[0]) + ' +- ' + str(cost[1]))           
        self.callback_counter[0] += 1
    

##################################################################################################################################################
                         
    def train_by_optimizer_local_and_global(self,batch_size=None):
        iteration = 0
        max_iteration = 100
        print ('start to optimize')
        likelihood = self.dggplvm.ELBO(self.X,self.N)
        print ('BEGINE Training, Log Likelihood = %.2f'% likelihood[0])
        #import minimize 
        #opt_results = minimize.run(self._optimizer_f, self._get_hypArray(params),length=number_epoch,verbose=True)
        init=[]
        from scipy.optimize import minimize
        start = time.time()
        while iteration < max_iteration:   
            
            init=np.hstack((self.params['m'].flatten(),self.params['S_b'].flatten()))
            
            opt_results = minimize(self.local_optimizer_f, init, method='L-BFGS-B', jac=self.local_optimizer_g, options={'ftol':0 , 'disp':True, 'maxiter': 5000}, tol=0,callback=self.callback_local)
            optimalHyp = deepcopy(opt_results.x)
            hype=self.unpack_local(optimalHyp)
            for param_name in self.opt_local_names:
                self.params[param_name]=hype[param_name]
            
            init=np.hstack((self.params['Z'].flatten(),self.params['mu'].flatten(),self.params['Sigma_b'].flatten(),self.params['lhyp'].flatten(),self.params['ls'].flatten()))
            
            opt_results = minimize(self.global_optimizer_f, init, method='L-BFGS-B', jac=self.global_optimizer_g, options={'ftol':0 , 'disp':True, 'maxiter': 5000}, tol=0,callback=self.callback_global)
            optimalHyp = deepcopy(opt_results.x)
            hype=self.unpack_global(optimalHyp)
            print('finished_local, Now iter' + str(self.callback_counter))
            for param_name in self.opt_global_names:
                self.params[param_name]=hype[param_name]
            
            likelihood = self.dggplvm.ELBO(self.X,self.N)
            print('finished_global, Now iter' + str(self.callback_counter))
            print(iteration)
            iteration += 1
        
        likelihood = self.dggplvm.ELBO(self.X,self.N)
        elapsed_time = time.time() - start
        print(elapsed_time)
        print ('END Training, Log Likelihood = %.2f'% likelihood[0])
        
    def unpack_local(self, x):
        x_param_values = [x[self.sizes_local[i-1]:self.sizes_local[i]].reshape(self.shapes_local[i-1]) for i in range(1,len(self.shapes_local)+1)]#これは['m', 'ls', 'lhyp', 'S', 'Z']の5つ分
        #それぞれのパラメータについてリストで与えられたとしても、例えばｍの場合i=1よってx[sizes[0]:sizes[1]]つまり適切な数のx[ｍの始まり:mの終わり]が出され、それがreshape((ｍのサイズ))で変換される
        params = {n:v for (n,v) in zip(self.opt_local_names, x_param_values)}

        return params
    
    def unpack_global(self, x):
        x_param_values = [x[self.sizes_global[i-1]:self.sizes_global[i]].reshape(self.shapes_global[i-1]) for i in range(1,len(self.shapes_global)+1)]#これは['m', 'ls', 'lhyp', 'S', 'Z']の5つ分
        #それぞれのパラメータについてリストで与えられたとしても、例えばｍの場合i=1よってx[sizes[0]:sizes[1]]つまり適切な数のx[ｍの始まり:mの終わり]が出され、それがreshape((ｍのサイズ))で変換される
        params = {n:v for (n,v) in zip(self.opt_global_names, x_param_values)}
        if 'lhyp' in params:
            params['lhyp']=params['lhyp'].squeeze()
        
        if 'ls' in params:
            params['ls']=params['ls'].reshape(1) 

        return params
    
    def local_optimizer_f(self, hypInArray):
        params=self.unpack_local(hypInArray)
        for param_name in self.opt_local_names:
            self.params[param_name]=params[param_name]
        cost=self.ELBO(self.X,self.N)
        return -cost[0]
    
    def local_optimizer_g(self, hypInArray):
        params=self.unpack_local(hypInArray)
        for param_name in self.opt_local_names:
            self.params[param_name]=params[param_name]
        gradient=[]
        minibatch = np.arange(self.N)
        for i in self.opt_local_names:
            g = self.get_grad(i, self.X, minibatch)
            gradient=np.hstack((gradient,g.flatten()))        
        return gradient
    
    def global_optimizer_f(self, hypInArray):
        params=self.unpack_global(hypInArray)
        for param_name in self.opt_global_names:
            self.params[param_name]=params[param_name]
        cost=self.ELBO(self.X,self.N)
        return -cost[0]
    
    def global_optimizer_g(self, hypInArray):
        params=self.unpack_global(hypInArray)
        for param_name in self.opt_global_names:
            self.params[param_name]=params[param_name]
        gradient=[]
        minibatch = np.arange(self.N)
        for i in self.opt_global_names:
            g = self.get_grad(i, self.X, minibatch)
            gradient=np.hstack((gradient,g.flatten()))        
        return gradient
    
    def callback_global(self, x):
        #インターバル毎にコールバックが出る
        if self.callback_counter[0]%self.print_interval == 0:
            opt_params = self.unpack_global(x)
            for param_name in self.opt_global_names:
                self.params[param_name]=opt_params[param_name]
            cost=self.ELBO(self.X,self.N)
            print ('iter ' + str(self.callback_counter) + ': ' + str(cost[0]) + ' +- ' + str(cost[1]))           
        self.callback_counter[0] += 1
    
    def callback_local(self, x):
        #インターバル毎にコールバックが出る
        if self.callback_counter[0]%self.print_interval == 0:
            opt_params = self.unpack_local(x)
            for param_name in self.opt_local_names:
                self.params[param_name]=opt_params[param_name]
            cost=self.ELBO(self.X,self.N)
            print ('iter ' + str(self.callback_counter) + ': ' + str(cost[0]) + ' +- ' + str(cost[1]))           
        self.callback_counter[0] += 1
                             
#############################for experiment
                     
    def experiment_train_by_optimizer_local_and_global(self,batch_size=None):
        iteration = 0
        max_iteration = 100
        print ('start to optimize')
        likelihood = self.dggplvm.ELBO(self.X,self.N)
        print ('BEGINE Training, Log Likelihood = %.2f'% likelihood[0])
        #import minimize 
        #opt_results = minimize.run(self._optimizer_f, self._get_hypArray(params),length=number_epoch,verbose=True)
        init=[]
        from scipy.optimize import minimize
        
        while iteration < max_iteration:   
            
            init=np.hstack((self.params['m'].flatten(),self.params['S_b'].flatten()))
            
            opt_results = minimize(self.local_optimizer_f, init, method='L-BFGS-B', jac=self.local_optimizer_g, options={'ftol':0 , 'disp':True, 'maxiter': 500}, tol=0,callback=self.callback_local)
            optimalHyp = deepcopy(opt_results.x)
            hype=self.unpack_local(optimalHyp)
            for param_name in self.opt_local_names:
                self.params[param_name]=hype[param_name]
            print('finished_local, Now iter' + str(self.callback_counter))
            test=0
            while test < 20:
                init=np.hstack((self.params['Z'].flatten(),self.params['mu'].flatten(),self.params['Sigma_b'].flatten(),self.params['lhyp'].flatten(),self.params['ls'].flatten()))
            
                opt_results = minimize(self.global_optimizer_f, init, method='L-BFGS-B', jac=self.global_optimizer_g, options={'ftol':1.0e-6 , 'disp':True, 'maxiter': 200}, tol=0,callback=self.callback_global)
                optimalHyp = deepcopy(opt_results.x)
                hype=self.unpack_global(optimalHyp)
                
                for param_name in self.opt_global_names:
                    self.params[param_name]=hype[param_name]
                if self.callback_counter[0]%20 == 0:
                    print('Now_global_iter:' + str(test))
                test +=1

            likelihood = self.dggplvm.ELBO(self.X,self.N)
            print('finished_global, Now iter' + str(self.callback_counter))
            print('finished_global, Now iter' + str(iteration))
            iteration += 1
        
        likelihood = self.dggplvm.ELBO(self.X,self.N)
        print ('END Training, Log Likelihood = %.2f'% likelihood[0])
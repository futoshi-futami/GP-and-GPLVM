import DGPLVM_theano_model
from DGPLVM_theano_model import DGGPLVM_model
import numpy as np
from copy import deepcopy
import time

class DGGPLVM_opt:
    def __init__(self, params, X, Y,Xinfo, samples = 500,batch_size=None):
        self.Y = Y
        self.X = X

        self.N = X.shape[0]
        
        if batch_size is None:
            batch_size = self.N
        
        correct=self.N/batch_size
        
        self.dggplvm = DGGPLVM_model(params,correct,Xinfo,samples=samples,batch_size=None)
        
        self.ELBO = self.dggplvm.ELBO
        self.f = self.dggplvm.f
        self.params = self.dggplvm.params 
        #self.KmmInv = self.dggplvm.KmmInv
        self.exec_f = self.dggplvm.exec_f
        self.estimate = self.dggplvm.estimate
        self.callback_counter = [0]
        self.print_interval = 10
        
        #RMSPROPのため
        self.param_updates = {n: np.zeros_like(v) for n, v in params.items()}#update用の同じサイズの空の箱を用意
        self.moving_mean_squared = {n: np.zeros_like(v) for n, v in params.items()}#ＲＭＳＰＲＯＰ用に更新の履歴（γを入れておく器をパラメータごとに用意
        self.learning_rates = {n: 1e-2*np.ones_like(v) for n, v in params.items()}#行進用の同じサイズの箱を用意
        
        #Scipyに入れるよう
        self.opt_param_names = ['Z', 'm', 'S_b', 'mu', 'Sigma_b', 'lhyp', 'ls','ga','Zlabel']
        self.opt_param_values = [np.atleast_2d(params[n]) for n in self.opt_param_names]
        
        self.shapes = [v.shape for v in self.opt_param_values]
        #Zの形は(40, 16)見たいな風にリストに入れてある
        self.sizes = [sum([np.prod(x) for x in self.shapes[:i]]) for i in range(len(self.shapes)+1)]
        #[0, 40, 41, 59, 1659, 2299]な感じで累積和でパラメータの大きさの格納を行う  
        
        #2段階用
        self.opt_local_names = ['m', 'S_b']
        self.opt_local_values = [np.atleast_2d(params[n]) for n in self.opt_local_names]
        
        self.shapes_local = [v.shape for v in self.opt_local_values]
        #Zの形は(40, 16)見たいな風にリストに入れてある
        self.sizes_local = [sum([np.prod(x) for x in self.shapes_local[:i]]) for i in range(len(self.shapes_local)+1)]
         
        self.opt_global_names = ['Z', 'mu', 'Sigma_b', 'lhyp', 'ls']
        self.opt_global_values = [np.atleast_2d(params[n]) for n in self.opt_global_names]
        
        self.shapes_global = [v.shape for v in self.opt_global_values]
        #Zの形は(40, 16)見たいな風にリストに入れてある
        self.sizes_global = [sum([np.prod(x) for x in self.shapes_global[:i]]) for i in range(len(self.shapes_global)+1)]


    def get_grad(self, param_name, X, minibatch):
        #wrt = {'Z': Z, 'm': m, 'S_b': S_b, 'mu': mu, 'Sigma_b': Sigma_b, 'lhyp': lhyp, 'ls': ls, 'KmmInv': KmmInv}
        
        if param_name in ['m', 'S_b']:
            grad =  self.exec_f(self.dggplvm.g[param_name]['KL_X'], X, minibatch) + self.estimate(self.dggplvm.g[param_name]['LL'], minibatch, X)[0]
        
        if param_name in ['mu', 'Sigma_b']:
            grad = self.exec_f(self.dggplvm.g[param_name]['KL_U'], X) + self.estimate(self.dggplvm.g[param_name]['LL'], minibatch, X)[0]
            
        if param_name in ['Z', 'lhyp','ls']:
            grad_ls, grad_std = self.estimate(self.dggplvm.g[param_name]['LL'], minibatch, X)
            grad = self.exec_f(self.dggplvm.g[param_name]['KL_U'], X) + grad_ls
    
        if param_name in ['Zlabel', 'ga']:
            grad_ls, grad_std = self.estimate(self.dggplvm.g[param_name]['LL'], minibatch, X)
            grad = self.exec_f(self.dggplvm.g[param_name]['KL_U'], X) + grad_ls+self.exec_f(self.dggplvm.g[param_name]['KL_X'], X, minibatch)
        
        # DEBUG
        if param_name == 'lhyp' and np.any(np.abs(grad) < grad_std / np.sqrt(self.dggplvm.samples)):
                #print 'Large noise, recomputing. lhyp grad mean:', grad, ', std:', grad_std / np.sqrt(self.clgp.samples)
            samples = self.dggplvm.samples * 10
            grad_ls, grad_std = self.estimate(self.dggplvm.g[param_name]['LL'], minibatch, X, samples=samples)
            grad = self.exec_f(self.dggplvm.g[param_name]['KL_U'], X) + grad_ls
            self.grad_std = grad_std

        return np.array(grad)

    #どのサンプルで最適化するのかminibatchで指定してもらう
    def opt_one_step(self, params, iteration, minibatch, opt = 'rmsprop', learning_rate_adapt = 0.2, use_einsum = True):        
        
        for param_name in params:
            # DEBUG
            if param_name in ['S_b']:#lsはｘの分散
                self.grad_ascent_one_step(param_name,minibatch, [param_name, self.X,minibatch], learning_rate_decay = learning_rate_adapt * 100 / (iteration + 100.0))
            
            elif param_name in ['m']:
                self.rmsprop_one_step_minibatch(param_name, minibatch, [param_name, self.X,minibatch], learning_rate_adapt = learning_rate_adapt)#, momentum = 0.9 - 0.4 * 100 / (iteration + 100.0))
            
            else:
                self.rmsprop_one_step(param_name, minibatch, [param_name, self.X,minibatch], learning_rate_adapt = learning_rate_adapt)#, momentum = 0.9 - 0.4 * 100 / (iteration + 100.0))        
    
            if param_name in ['lhyp']:
                self.params[param_name] = np.clip(self.params[param_name], -8, 8)
            
            #if param_name in ['lhyp', 'Z']:
            #    self.dggplvm.update_KmmInv_cache()

    def opt_local_step(self, local_params, iteration, minibatch, opt = 'rmsprop', learning_rate_adapt = 0.2, use_einsum = True):        
        
        for param_name in local_params:
            # DEBUG
            if param_name in ['S_b']:#lsはｘの分散
                self.grad_ascent_one_step(param_name,minibatch, [param_name, self.X,minibatch], learning_rate_decay = learning_rate_adapt * 100 / (iteration + 100.0))
            
            elif param_name in ['m']:
                self.rmsprop_one_step_minibatch(param_name, minibatch, [param_name, self.X,minibatch], learning_rate_adapt = learning_rate_adapt)#, momentum = 0.9 - 0.4 * 100 / (iteration + 100.0))
   
    def opt_global_step(self, global_params, iteration, minibatch, opt = 'rmsprop', learning_rate_adapt = 0.2, use_einsum = True):        
        
        for param_name in global_params:
            # DEBUG
            self.rmsprop_one_step(param_name, minibatch, [param_name, self.X,minibatch], learning_rate_adapt = learning_rate_adapt)#, momentum = 0.9 - 0.4 * 100 / (iteration + 100.0))        
        
        if param_name in ['lhyp']:
                self.params[param_name] = np.clip(self.params[param_name], -8, 8)
    
    
    def grad_ascent_one_step(self, param_name, minibatch, grad_args, momentum = 0.9, learning_rate_decay = 1):
        #grad_args=[param_name, self.Y, KmmInv_grad, self.mask]
        self.dggplvm.params[param_name][minibatch] += (learning_rate_decay*self.learning_rates[param_name][minibatch]* self.param_updates[param_name][minibatch])
        grad = self.get_grad(*grad_args)
        self.param_updates[param_name][minibatch] = grad
                         
    def rmsprop_one_step_minibatch(self, param_name,minibatch, grad_args, decay = 0.9, momentum = 0, learning_rate_adapt = 0.05, 
        learning_rate_min = 1e-6, learning_rate_max = 10):
        # RMSPROP: Tieleman, T. and Hinton, G. (2012), Lecture 6.5 - rmsprop, COURSERA: Neural Networks for Machine Learning
        # Implementation based on https://github.com/BRML/climin/blob/master/climin/rmsprop.py
        
        # We use Nesterov momentum: first, we make a step according to the momentum and then we calculate the gradient.
        step1 = self.param_updates[param_name][minibatch] * momentum
        self.params[param_name][minibatch] += step1
        grad = self.get_grad(*grad_args)

        self.moving_mean_squared[param_name][minibatch] = (decay * self.moving_mean_squared[param_name][minibatch] + (1 - decay) * grad ** 2)
        step2 = self.learning_rates[param_name][minibatch] * grad / (self.moving_mean_squared[param_name][minibatch] + 1e-8)**0.5

        self.params[param_name][minibatch] += step2

        step = step1 + step2

        # Step rate adaption. If the current step and the momentum agree, we slightly increase the step rate for that dimension.
        if learning_rate_adapt:
            # This code might look weird, but it makes it work with both numpy and gnumpy.
            step_non_negative = step > 0
            step_before_non_negative = self.param_updates[param_name][minibatch] > 0
            agree = (step_non_negative == step_before_non_negative) * 1.#０か１が出る
            adapt = 1 + agree * learning_rate_adapt * 2 - learning_rate_adapt
            self.learning_rates[param_name][minibatch] *= adapt
            self.learning_rates[param_name][minibatch] = np.clip(self.learning_rates[param_name][minibatch], learning_rate_min, learning_rate_max)

        self.param_updates[param_name][minibatch] = step

    def rmsprop_one_step(self, param_name, minibatch, grad_args, decay = 0.9, momentum = 0, learning_rate_adapt = 0.05, 
        learning_rate_min = 1e-6, learning_rate_max = 10):
        # RMSPROP: Tieleman, T. and Hinton, G. (2012), Lecture 6.5 - rmsprop, COURSERA: Neural Networks for Machine Learning
        # Implementation based on https://github.com/BRML/climin/blob/master/climin/rmsprop.py
        
        # We use Nesterov momentum: first, we make a step according to the momentum and then we calculate the gradient.
        step1 = self.param_updates[param_name] * momentum
        self.params[param_name] += step1
        grad = self.get_grad(*grad_args)

        self.moving_mean_squared[param_name] = (decay * self.moving_mean_squared[param_name] + (1 - decay) * grad ** 2)
        step2 = self.learning_rates[param_name] * grad / (self.moving_mean_squared[param_name] + 1e-8)**0.5

        # DEBUG
        if param_name == 'lhyp':
            step2 = np.clip(step2, -0.1, 0.1)

        self.params[param_name] += step2

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


    def choose_best_z(self, ind, Y_true, mask, samples=20):
        """
        Assign m[i] to the best location among all the inducing points.
        """
        orig_params = {'m': self.params['m'], 'ls': self.params['ls']}
        N = len(ind)
        M = self.params['Z'].shape[0]

        self.params['ls'] = self.params['ls'][ind]
        f = np.zeros((M + 1, N))
        for m in range(M + 1):
            if m < M:
                self.params['m'] = np.tile(self.params['Z'][m], (N, 1))
            else:
                self.params['m'] = orig_params['m'][ind]

            # KL.
            kl_x = self.exec_f(self.f['KL_X_all'])
            f[m] += kl_x

            # Likelihood.
            for modality in range(len(Y_true)):
                S, _ = self.estimate(self.f['S'], modality=modality, samples=samples)

                Y_ind = Y_true[modality][ind]
                mask_ind = mask[:, modality][ind]
                f[m] += np.log(np.maximum(np.sum(S * Y_ind, 1), 1e-16)) * mask_ind

        self.params['m'], self.params['ls'] = orig_params['m'], orig_params['ls']

        best_z = np.argmax(f, 0)

        # Do not change m if best_z == M.
        self.params['m'][ind[best_z < M]] = self.params['Z'][best_z[best_z < M]]

        return best_z

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
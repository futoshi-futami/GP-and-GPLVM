# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 11:58:34 2017

@author: Futami
"""
from copy import deepcopy
import sys; sys.path.insert(0, "../Theano"); sys.path.insert(0, "../../Theano")
import theano; import theano.tensor as T; import theano.sandbox.linalg as sT
import numpy as np
import pickle

#from scipy.stats import multivariate_normal as mvn

print ('Theano version: ' + theano.__version__ + ', base compile dir: ' + theano.config.base_compiledir)
theano.config.mode = 'FAST_RUN'
theano.config.optimizer = 'fast_run'
theano.config.reoptimize_unpickled_function = False

eps = 1e-4
class kernel:
    def RBF(self, sf2, l, X1, X2 = None):
        _X2 = X1 if X2 is None else X2
        dist = ((X1 / l)**2).sum(1)[:, None] + ((_X2 / l)**2).sum(1)[None, :] - 2*(X1 / l).dot((_X2 / l).T)
        RBF = sf2 * T.exp(-dist / 2.0)
        return (RBF + eps * T.eye(X1.shape[0])) if X2 is None else RBF
    def RBFnn(self, sf2, l, X):
        return sf2 + eps
    def LIN(self, sl2, X1, X2 = None):
        _X2 = X1 if X2 is None else X2
        LIN = sl2 * (X1.dot(_X2.T) + 1)
        return (LIN + eps * T.eye(X1.shape[0])) if X2 is None else LIN
    def LINnn(self, sl2, X):
        return sl2 * (T.sum(X**2, 1) + 1) + eps


def shared_scalar(val=0., dtype=theano.config.floatX,name=None):
    return theano.shared(np.cast[dtype](val))

class DGGPLVM:
    def __init__(self,initial_params):
        self.ker=kernel()
        try:
            print ('Trying to load model...')
            with open('modelN.save', 'rb') as file_handle:
                self.f, self.g = pickle.load(file_handle)
                print ('Loaded!')
            return
        except:
            print ('Failed. Creating a new model...')

        print ('Setting up variables...')
        #initial_params = {'m':m,'S_b':S_b,'mu':mu,'Sigma_b':Sigma_b,'Z':Z,'lhyp':lhyp,'ls':ls}
        
        self.m = shared_scalar(initial_params['m'])
        self.S_b = shared_scalar(initial_params['S_b'])
        self.mu = shared_scalar(initial_params['mu'])
        self.Sigma_b = shared_scalar(initial_params['Sigma_b'])
        self.Z = shared_scalar(initial_params['Z'])
        self.lhyp = shared_scalar(initial_params['lhyp'])
        self.ls = shared_scalar(initial_params['ls'])
        
        self.opt_param_names = ['m','S_b','mu','Sigma_b','Z','lhyp','ls']
        self.opt_param_values = [np.atleast_2d(initial_params[n]) for n in self.opt_param_names]
        
        self.shapes = [v.shape for v in self.opt_param_values]
        #Zの形は(40, 16)見たいな風にリストに入れてある
        self.sizes = [sum([np.prod(x) for x in self.shapes[:i]]) for i in range(len(self.shapes)+1)]
        #[0, 40, 41, 59, 1659, 2299]な感じで累積和でパラメータの大きさの格納を行う   
        
        # Variables
        X,Y,X_test = T.dmatrices('X','Y','X_test')

        print ('Compiling model ...')
        inputs = {'X': X, 'Y': Y, 'X_test': X_test}
        
        KL_X,KL_U,LL=self.get_model(X, Y, X_test)
        
        z = 0.0*sum([T.sum(v) for v in inputs.values()]) # solve a bug with derivative wrt inputs not in the graph
        #f = zip(['opt_A_mean', 'opt_A_cov', 'EPhi', 'EPhiTPhi', 'Y_pred_mean', 'Y_pred_var', 'LL', 'KL'],[opt_A_mean, opt_A_cov, EPhi, EPhiTPhi, Y_pred_mean, Y_pred_var, LL, KL])
        #self.f = {n: theano.function(list(inputs.values()), f+z, name=n, on_unused_input='ignore') for n,f in zip(['LL'],[LL])}
        #        z = 0.0*sum([T.sum(v) for v in inputs.values()])
        f = zip(['LL','KL_X','KL_U'],[LL,KL_X,KL_U])
        self.f = {n: theano.function(list(inputs.values()), f+z, name=n, on_unused_input='ignore')
                     for n, f in f}
        
        #g = zip(['LL', 'KL'], [LL, KL])
        wrt = {'m':self.m,'S_b':self.S_b,'mu':self.mu,'Sigma_b':self.Sigma_b,'Z':self.Z,'lhyp':self.lhyp,'ls':self.ls}
        zz = 0.0*sum([T.sum(v) for v in wrt.values()])
        #このｚｚは実にいいエラー回避にテクニックだ。ＫＬには含まれないパラメータを無理やり含ませる
        self.gLL = {vn: theano.function(list(inputs.values()), T.grad(LL+zz,vv),
                                      name='dLL'+'_d'+vn,on_unused_input='ignore')
                                      for vn, vv in wrt.items()}
        
        self.gKL_X = {vn: theano.function(list(inputs.values()), T.grad(KL_X+zz,vv),
                                      name='dKL_X'+'_d'+vn,on_unused_input='ignore')
                                      for vn, vv in wrt.items()}
        
        self.gKL_U = {vn: theano.function(list(inputs.values()), T.grad(KL_U+zz,vv),
                                      name='dKL_U'+'_d'+vn,on_unused_input='ignore')
                                      for vn, vv in wrt.items()}
        
        #self.g = {vn: {gn: theano.function(list(inputs.values()), T.grad(gv+z, vv), name='d'+gn+'_d'+vn,
        #    on_unused_input='ignore') for gn,gv in zip(['LL', 'KL'], [LL, KL])} for vn, vv in wrt.items()}
        
        #with open('modelN.save', 'wb') as file_handle:
        #    print ('Saving model...')
        #    sys.setrecursionlimit(2000)
        #    pickle.dump([self.f,self.g], file_handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_model(self,X, Y, X_test):
        #initial_params = {'m':m,'S_b':S_b,'mu':mu,'Sigma_b':Sigma_b,'Z':Z,'lhyp':lhyp,'ls':ls}
        (M, D), N, Q = self.Z.shape, X.shape[0], X.shape[1]
        
        #変数の正の値への制約条件
        beta, sf2, l = T.exp(self.ls), T.exp(self.lhyp[0]), T.exp(self.lhyp[1:])
        S=T.exp(self.S_b)
        #Sigma=T.exp(self.Sigma_b)
        
        #xについてはルートを取らなくても対角行列なので問題なし
        #uについては対角でないのでコレスキー分解するとかして三角行列を作る必要がある
        Sigma = T.tril(self.Sigma_b - T.diag(T.diag(self.Sigma_b)) + T.diag(T.exp(T.diag(self.Sigma_b))))
        
        #スケール変換
        mu_scaled, Sigma_scaled = sf2**0.5 * self.mu, sf2**0.5 * Sigma
        
        #reparametarizationのための乱数
        srng = T.shared_randomstreams.RandomStreams(234)
        eps_NQ = srng.normal(self.m.shape)
        eps_M = srng.normal(self.mu.shape)
        
        #サンプルの生成。バッチでやるので一回だけのＭＣ
        Xtilda = self.m + S * eps_NQ
        U = mu_scaled+Sigma_scaled*eps_M

        
        Kmm = self.ker.RBF(sf2, l, self.Z)
        KmmInv = sT.matrix_inverse(Kmm) 
        #KmmDet=theano.sandbox.linalg.det(Kmm)
    
        Kmn = self.ker.RBF(sf2,l,self.Z,Xtilda)
        Knn = self.ker.RBF(sf2,l,Xtilda,Xtilda)
        
        Ktilda=Knn-T.dot(Kmn.T,T.dot(KmmInv,Kmn))
        
        Kinterval=T.dot(KmmInv,Kmn)
              
        mean_U=T.dot(Kinterval.T,U)
        Covariance = beta       
        
        LL = self.log_mvn(X, mean_U, Covariance) - 0.5*beta*T.sum((T.eye(N)*Ktilda))
        
        #KL_X = -0.5 * (-T.sum(T.log(T.sum(Sigma,0))) + T.dot(m.T,T.dot(KmmInv,m)).squeeze() + T.sum((Sigma*KmmInv)) - M)-0.5*T.log(KmmDet)
        
        KL_X = self.KLD_X(self.m,S)
        
        KL_U = self.KLD_U(mu_scaled , Sigma_scaled , Kmm)
        
        return KL_X,KL_U,LL
    
    def log_mvn(self, y, mean,beta):#対角ノイズ、YはＮ×Ｄのデータ,それの正規分布の対数尤度
        N = y.shape[0]
        D = y.shape[1]
        return -0.5 * N * D * T.log(2 * np.pi/beta) - 0.5 * beta * T.sum((y - mean)**2)
    
    def KLD_X(self,m,S):
        N = m.shape[0]
        Q = m.shape[1]
        
        KL_X = T.sum(m*m)+T.sum(S-T.log(S)) - Q*N
        
        return 0.5*KL_X
    
    def KLD_U(self, m, L_scaled, Kmm):#N(u|m,S)とN(u|0,Kmm) S=L*L.T(コレスキー分解したのを突っ込みましょう)
        M = m.shape[0]
        D = m.shape[1]
        KmmInv = sT.matrix_inverse(Kmm)
        
        KL_U = D * (T.sum(KmmInv.T * L_scaled.dot(L_scaled.T)) - M - 2.0*T.sum(T.log(T.diag(L_scaled))) + 2.0*T.sum(T.log(T.diag(sT.cholesky(Kmm)))))
        KL_U += T.sum(T.dot(KmmInv,m)*m) 
        
        return 0.5*KL_U
    
    def get_outputs(self, X, Y, X_test):
        '''
        Input numpy array, output posterior distributions.
        Note: This function is independent of Theano
        '''
        inputs = {'X':X, 'Y':Y, 'X_test':X_test}
        outputs = {n: self.f[n](*inputs.values()) for n in self.f.keys()}
        return outputs
    
    #def get_prediction(self, X, Y,X_test):
    #    inputs = {'X':X, 'Y':Y, 'x_test':X_test}
    #    ymu = self.f['y_test_mu'](*inputs.values())
    #    ys2 = self.f['y_test_var'](*inputs.values())
    #    return ymu, ys2
    
    def get_likelihood(self,X, Y):
        inputs = {'X':X, 'Y':Y, 'x_test':X}
        LL = self.f['LL'](*inputs.values())
        KL_X = self.f['KL_X'](*inputs.values())
        KL_U = self.f['KL_U'](*inputs.values())
        return KL_X, KL_U, LL
 
    def get_cost_grads(self, X, Y):
        '''
        get the likelihood and gradients 
        '''
        inputs = {'X':X, 'Y':Y, 'x_test':X}
        #outputs = {n: self.f[n](*inputs.values()) for n in self.f.keys()}
        gradsLL = {n: self.gLL[n](*inputs.values()) for n in self.gLL.keys()}
        gradsKL_X = {n: self.gKL_X[n](*inputs.values()) for n in self.gKL_X.keys()}
        gradsKL_U = {n: self.gKL_U[n](*inputs.values()) for n in self.gKL_U.keys()}
        
        return gradsKL_X, gradsKL_U, gradsLL,#, outputs
    

    def opt(self, train_x_val, train_y_val,params, 
            lr, momentum = 0., decay=None,
            nesterov=False, updates={},opt_method='SGD'):
        '''
        Gradient based optimizations.
        '''
        if len(updates) == 0:
            for n in params.keys():
                updates[n] = 0.
        if opt_method=='SGD':
            grads = self.get_cost_grads(train_x_val, train_y_val)
            for n in params.keys():
                g,p = grads[n], params[n]
                updates[n] = lr * g
        elif opt_method =='rmsprop':
            # RMSPROP: Tieleman, T. and Hinton, G. (2012), Lecture 6.5 - rmsprop, COURSERA:
            # Neural Networks for Machine Learning.
            if nesterov and momentum > 0.:
                # nesterov momentum, make a move according to momentum first
                # then calculate the gradients.
                for n in params.keys():
                    params[n].set_value( params[n].get_value() + momentum * updates[n])
            grads = self.get_cost_grads(train_x_val, train_y_val)
            for n in params.keys():
                g, p = grads[n], params[n]
                self.moving_mean_squared[n] = (decay * self.moving_mean_squared[n] + 
                                               (1.-decay) * g ** 2)
                updates[n] = lr * g / (np.sqrt(self.moving_mean_squared[n])+ 1e-8)
        else:
            raise NotImplementedError
        return updates

    def unpack(self, x):
        x_param_values = [x[self.sizes[i-1]:self.sizes[i]].reshape(self.shapes[i-1]) for i in range(1,len(self.shapes)+1)]#これは['m', 'ls', 'lhyp', 'S', 'Z']の5つ分
        #それぞれのパラメータについてリストで与えられたとしても、例えばｍの場合i=1よってx[sizes[0]:sizes[1]]つまり適切な数のx[ｍの始まり:mの終わり]が出され、それがreshape((ｍのサイズ))で変換される
        #params = OrderedDict({n:v for (n,v) in zip(self.opt_param_names, x_param_values)})#そして変換しなおしたやつを辞書に適切に入れる
        params = {n:v for (n,v) in zip(self.opt_param_names, x_param_values)}
        for vn in params.keys():
            params[vn]=params[vn].squeeze() 

        return params
    ##############################################
    ## BEGIN TRAIN MODEL by EXTERNAL OPTIMIZERS ##
    ##############################################
    def estimate_grads(self):
        batch_size = self.batch_size
        '''
        Estimate gradient by averaging mini-batch.
        '''
        params  = {'Z':self.Z,'m':self.m,'S':self.S,'ls':self.ls,'lhyp':self.lhyp}

        N = self.X.shape[0]
        
        if batch_size is None:
            batch_size = N
        
        num_batches = N / batch_size
        
        if N%batch_size!=0:
            
            num_batches =np.floor(num_batches)+ 1
        
        train_index = np.arange(0,N)
        
        grads_list,est_grads = {}, {}
        
        for n in params.keys():
            grads_list[n] = [] 
        
        for i in range(int(num_batches)):
            np.random.shuffle(train_index)
            batch_x_val = self.X[train_index[:batch_size],:]
            batch_y_val = self.Y[train_index[:batch_size],:]
            grads = self.get_cost_grads(batch_x_val, batch_y_val)
            for n in params.keys():
                grads_list[n].append(grads[n])
        
        for n in params.keys():
            est_grads[n] = -np.cumsum(grads_list[n],0)[-1]/int(num_batches) # NOTE: negative grads

        return est_grads
    
    def _apply_hyp(self, hypInArray):
        '''
        Keep the order: mean, sigma_n, sigma_f, l_k
        '''
        self.Z.set_value(hypInArray['Z'])
        self.m.set_value(hypInArray['m'])
        self.S.set_value(hypInArray['S'])
        self.ls.set_value(hypInArray['ls'])
        self.lhyp.set_value(hypInArray['lhyp'])
        
    def _get_hypArray(self,params):
        return np.hstack((params['Z'].get_value().flatten(),
                         params['m'].get_value().flatten(),
                         params['S'].get_value().flatten(),
                         params['ls'].get_value().flatten(),
                         params['lhyp'].get_value().flatten()))


    def _convert_to_array(self, grads):
        return np.hstack((grads['Z'].flatten(),grads['m'].flatten(),grads['S'].flatten(),grads['ls'].flatten(),grads['lhyp'].flatten()))

    def _optimizer_f(self, hypInArray):
        hypeInarray=self.unpack(hypInArray)
        self._apply_hyp(hypeInarray)
        if self.batch_size is not None:
            splits = int(self.N / self.batch_size)
            split = np.random.randint(splits)
            inputs = {'X': self.X[split::splits], 'Y': self.Y[split::splits],'x_test':self.X[split::splits]}
            LL = self.N / self.batch_size * self.f['LL'](*inputs.values())        
            KL,_ = self.get_likelihood(self.X, self.Y)
        else:
            KL,LL = self.get_likelihood(self.X, self.Y)
        
                
        cost = -(KL+LL) # negative log-likelihood
        #est_grads = self.estimate_grads()
        #grads_list = self._convert_to_array(est_grads)
        return cost
    
    def _optimizer_g(self, hypInArray):
        hypeInarray=self.unpack(hypInArray)
        self._apply_hyp(hypeInarray)
        #ll = self.get_likelihood(self.X, self.Y)
        #cost = -ll # negative log-likelihood
        
        if self.batch_size is not None:
            splits =int(self.N / self.batch_size)
            split = np.random.randint(splits)
            inputs = {'X': self.X[split::splits], 'Y': self.Y[split::splits],'x_test':self.X[split::splits]}
            dLL = {n: self.gLL[n](*inputs.values()) for n in self.gLL.keys()}
            dKL,_ = self.get_cost_grads(self.X, self.Y)
        else:
            dKL,dLL = self.get_cost_grads(self.X, self.Y)
        #grads += [-(dLL - dKL)]
        
        #est_grads = self.estimate_grads()
        #print(est_grads)
        LL_grads_list = self._convert_to_array(dLL)
        KL_grads_list = self._convert_to_array(dKL)
        return -(LL_grads_list + KL_grads_list)

    def train_by_optimizer(self, X, Y, batch_size=None):
        params  = {'Z':self.Z,'m':self.m,'S':self.S,'ls':self.ls,'lhyp':self.lhyp}
        
        if batch_size is None:
            self.batch_size = len(X)
        else:
            self.batch_size = batch_size
        self.X = X
        self.Y = Y
        self.N=X.shape[0]
        print ('start to optimize')
        KL,LL = self.get_likelihood(X, Y)
        likelihood=KL+LL
        print ('BEGINE Training, Log Likelihood = %.2f'% likelihood)
        #import minimize 
        #opt_results = minimize.run(self._optimizer_f, self._get_hypArray(params),length=number_epoch,verbose=True)
        from scipy.optimize import minimize
        opt_results = minimize(self._optimizer_f, self._get_hypArray(params), method='L-BFGS-B', jac=self._optimizer_g, options={'ftol':1.0e-50 , 'disp':True, 'maxiter': 5000}, tol=0,)
        optimalHyp = deepcopy(opt_results.x)
        hype=self.unpack(optimalHyp)
        self._apply_hyp(hype)
        
        KL,LL = self.get_likelihood(X, Y)
        likelihood=KL+LL
        print ('END Training, Log Likelihood = %.2f'% likelihood)
        
        
    def RMSPROP_optimizer(self, X, Y, batch_size=None,max_iteration = 20): 
        params  = {'m':self.m,'S_b':self.S_b,'mu':self.mu,'Sigma_b':self.Sigma_b,'Z':self.Z,'lhyp':self.lhyp,'ls':self.ls}
        if batch_size is None:
            self.batch_size = len(X)
        else:
            self.batch_size = batch_size
        self.X = X
        self.Y = Y
        self.N=X.shape[0]
        
        self.param_updates = {n: np.zeros_like(v) for n, v in params.items()}#update用の同じサイズの空の箱を用意
        self.moving_mean_squared = {n: np.zeros_like(v) for n, v in params.items()}#ＲＭＳＰＲＯＰ用に更新の履歴（γを入れておく器をパラメータごとに用意
        self.learning_rates = {n: 1e-2*np.ones_like(v) for n, v in params.items()}#行進用の同じサイズの箱を用意
        
        print ('Optimising...')
        iteration = 0
        while iteration < max_iteration:
            self.opt_one_step(params.keys(), iteration)
            KL_X,KL_U,LL = self.get_likelihood(X, Y)
            likelihood=-KL_X-KL_U+LL
            print ('iter ' + str(iteration) + ': ' + str(likelihood))
            #if iteration%10 == 0:
            #    print ('error ' + str(get_error(clgp, clgp.Y, ~clgp.mask)[0]))
            iteration += 1
                
    def opt_one_step(self, params, iteration, opt = 'rmsprop', learning_rate_adapt = 0.2, use_einsum = True):
        
        for param_name in params:
            # DEBUG
            if opt == 'grad_ascent' or param_name in ['ls']:#lsはｘの分散
                self.grad_ascent_one_step(param_name, [param_name, self.Y, self.mask], 
                    learning_rate_decay = learning_rate_adapt * 100 / (iteration + 100.0))
            elif opt == 'rmsprop':
                self.rmsprop_one_step(param_name, [param_name, self.Y, self.mask], 
                    learning_rate_adapt = learning_rate_adapt)#, momentum = 0.9 - 0.4 * 100 / (iteration + 100.0))
            if param_name in ['lhyp']:
                self.params[param_name] = np.clip(self.params[param_name], -8, 8)
            if param_name in ['lhyp', 'Z']:
                self.clgp.update_KmmInv_cache()


    def grad_ascent_one_step(self, param_name, grad_args, momentum = 0.9, learning_rate_decay = 1):
        #grad_args=[param_name, self.Y, KmmInv_grad, self.mask]
        self.clgp.params[param_name] += (learning_rate_decay*self.learning_rates[param_name]* self.param_updates[param_name])
        grad = self.get_grad(*grad_args)
        if param_name in ['lhyp']:
            self.param_updates[param_name] = momentum*self.param_updates[param_name] + (1. - momentum)*grad
        else:
            self.param_updates[param_name] = grad


    def rmsprop_one_step(self, param_name, grad_args, decay = 0.9, momentum = 0, learning_rate_adapt = 0.05, 
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
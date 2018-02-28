# To speed Theano up, create ram disk: mount -t tmpfs -o size=512m tmpfs /mnt/ramdisk
# Then use flag THEANO_FLAGS='base_compiledir=/mnt/ramdisk' python script.py
import sys; sys.path.insert(0, "../Theano"); sys.path.insert(0, "../../Theano")
import theano; import theano.tensor as T; import theano.sandbox.linalg as sT
import numpy as np
import pickle


print ('Theano version: ' + theano.__version__ + ', base compile dir: ' + theano.config.base_compiledir)
theano.config.mode = 'FAST_RUN'
theano.config.optimizer = 'fast_run'
theano.config.reoptimize_unpickled_function = False

eps = 1e-40
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

class DGGPLVM_model:
    def __init__(self, params,correct, samples = 20,batch_size=None):
        ker = kernel()
        self.samples = samples
        self.params =  params
        self.batch_size=batch_size
        
        #データの保存ファイル
        model_file_name = 'model2' + '.save'
                                    #もしこれまでに作ったのがあるならロードする
        try:
            print ('Trying to load model...')
            with open(model_file_name, 'rb') as file_handle:
                obj = pickle.load(file_handle)
                self.f, self.g,self.ES_US= obj
                print ('Loaded!')
            return
        except:
            print ('Failed. Creating a new model...')
        
        X,Y,X_test,m,S_b,Z,eps_NQ,eps_M=\
        T.dmatrices('X','Y','X_test','m','S_b','Z','eps_NQ','eps_M')

        mu,Sigma=T.dmatrices('mu','Sigma')

        lhyp = T.dvector('lhyp')
        ls=T.dvector('ls')
        
        N,Q= m.shape
        M=Z.shape[0]
        D=X.shape[1]
        
        #変数の正の値への制約条件
        beta = T.exp(ls[0])
        #beta=T.exp(lhyp[0])
        sf2, l = T.exp(lhyp[0]), T.exp(lhyp[1:1+Q])
        
        S=T.exp(S_b)

        
        Xtilda = m + S * eps_NQ

        print ('Setting up cache...')
        
        Kmm = ker.RBF(sf2, l, Z)
        KmmInv = sT.matrix_inverse(Kmm) 
        #KmmDet=theano.sandbox.linalg.det(Kmm)
        
        from theano.tensor.shared_randomstreams import RandomStreams
        srng = RandomStreams(seed=234)
        rv_u = srng.normal((2,N,Q))
        rv_s = srng.normal((2,N,Q)) #平均と分散で違う乱数を使う必要があるので別々に銘銘
        
        xx_s=m.reshape([1,N,Q])+S.reshape([1,N,Q])*rv_s
        xxx_s=xx_s.reshape([2,N,1,Q])
        zz=Z.reshape([1,1,M,Q])
        rbf_u=T.exp(-T.sum(((xxx_s-zz)**2)/(2*l.reshape([1,1,1,Q])),-1))*sf2#N×Ｍ
        A=Kmm+beta*T.sum(T.mean(rbf_u.reshape([2,M,1,N])*rbf_u.reshape([2,1,M,N]),0),-1)
        Ainv=sT.matrix_inverse(A)
        Sigma_f=T.dot(Kmm,T.dot(Ainv,Kmm))
                     
        xx=m.reshape([1,N,Q])+S.reshape([1,N,Q])*rv_u
        xxx=xx.reshape([2,N,1,Q])
        rbf=T.mean(T.exp(-T.sum(((xxx-zz)**2)/(2*l.reshape([1,1,1,Q])),-1)),0)#N×Ｍ
        RHS=T.sum(rbf.reshape([M,1,N])*X.reshape([1,D,N]),2)

        mu_f=beta*T.dot(Kmm,T.dot(Ainv,RHS)) 
        
        self.ES_US = theano.function([m,S_b,Z,X,lhyp,ls], [mu_f,Sigma_f],on_unused_input='ignore')
        
        rv_u_d = srng.normal((N,Q))
        rv_s_d = srng.normal((N,Q)) #平均と分散で違う乱数を使う必要があるので別々に銘銘
        Xtilda_u = m + S * rv_u_d
        Xtilda_s = m + S * rv_s_d
        Kmn_u = ker.RBF(sf2, l, Z, Xtilda_u)
        Kmn_s = ker.RBF(sf2, l, Z, Xtilda_s)
        
        
        print ('Modeling...')
        
        Kmn = ker.RBF(sf2,l,Z,Xtilda)
        Knn = ker.RBF(sf2,l,Xtilda,Xtilda)
        
        Ktilda=Knn-T.dot(Kmn.T,T.dot(KmmInv,Kmn))
        
        Kinterval=T.dot(KmmInv,Kmn)
        

        #スケール変換
        Sigma_L=sT.cholesky(Sigma)
        U = mu+Sigma_L.dot(eps_M)
        
        mean_U=T.dot(Kinterval.T,U)
        Covariance = beta       
        
        LL = (self.log_mvn(X, mean_U, Covariance) - 0.5*beta*T.sum((T.eye(N)*Ktilda)))*correct      
        KL_X = -self.KLD_X(m,S)*correct
        KL_U = -self.KLD_U(mu, Sigma_L, Kmm,KmmInv)
        
        print ('Compiling model ...')        


        inputs = {'X': X, 'Z': Z, 'm': m, 'S_b': S_b, 'mu': mu, 'Sigma': Sigma, 'lhyp': lhyp, 'ls': ls, 
            'eps_M': eps_M, 'eps_NQ': eps_NQ}
        
        z = 0.0*sum([T.sum(v) for v in inputs.values()]) # solve a bug with derivative wrt inputs not in the graph
        
        self.f = {n: theano.function(list(inputs.values()), f+z, name=n, on_unused_input='ignore')\
                  for n,f in zip(['X', 'U', 'LL', 'KL_U', 'KL_X'], [X, U, LL, KL_U, KL_X])}
        
        
        wrt = {'Z': Z, 'm': m, 'S_b': S_b, 'lhyp': lhyp, 'ls': ls}
        self.g = {vn: {gn: theano.function(list(inputs.values()), T.grad(gv+z, vv), name='d'+gn+'_d'+vn, 
            on_unused_input='ignore') for gn,gv in zip(['LL', 'KL_U', 'KL_X'], [LL, KL_U, KL_X])} for vn, vv in wrt.items()}

        with open(model_file_name, 'wb') as file_handle:
            print ('Saving model...')
            sys.setrecursionlimit(2000)
            pickle.dump([self.f, self.g,self.ES_US], file_handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    
    def log_mvn(self, y, mean,beta):#対角ノイズ、YはＮ×Ｄのデータ,それの正規分布の対数尤度
        N = y.shape[0]
        D = y.shape[1]
        return -0.5 * N * D * T.log(2 * np.pi/beta) - 0.5 * beta * T.sum((y - mean)**2)
    
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
              
    def update_KmmInv_cache(self):#実際に計算させるため
        self.KmmInv = self.f_KmmInv(self.params['Z'], self.params['lhyp']).astype(theano.config.floatX)

    def estimate_U(self,X):
        self.mu,self.Sigma = self.ES_US(self.params['m'],self.params['S_b'],self.params['Z'],X,self.params['lhyp'],self.params['ls'])      

    def exec_f(self, f, X = [[0]], minibatch=None):
        #params=['m','S_b','mu','Sigma_b','Z','lhyp','ls']

        inputs={}
        (N, Q)=self.params['m'].shape
        M=self.params['Z'].shape[0]
        D=X.shape[1]
        
        inputs['m']= self.params['m']
        inputs['S_b']= self.params['S_b']
        
        inputs['Z'] = self.params['Z']
        inputs['lhyp']= self.params['lhyp']
        inputs['ls']= self.params['ls']
        #inputs['KmmInv'] = self.KmmInv
        
        inputs['eps_M']= np.random.randn(M,D)
        inputs['eps_NQ']= np.random.randn(N,Q)
        
        inputs['X'] = X
        
        if minibatch is not None:
            inputs['X']= inputs['X'][minibatch]
            inputs['m']= inputs['m'][minibatch]
            inputs['S_b'] = inputs['S_b'][minibatch]
            inputs['eps_NQ']= inputs['eps_NQ'][minibatch] 
        inputs['mu'],inputs['Sigma'] = self.ES_US(inputs['m'],inputs['S_b'],inputs['Z'],X,inputs['lhyp'],inputs['ls'])
        
        return f(**inputs)


    def estimate(self, f, minibatch, X = [[0]], samples = None):
        
        #minibatch = np.random.choice(X,batch_size,replace=False)
        
        f_acc = np.array([self.exec_f(f, X, minibatch=minibatch) for s in range(samples if samples != None else self.samples)])
        
        return np.nanmean(f_acc, 0), np.nanstd(f_acc, 0) 
        #nanをつけた平均と分散、ベクトル中に欠損値 NaN が含まれている場合、以下のようなメソッドを利用することで、欠損値を無視して計算を行うことができる。

    def ELBO(self, X, batch_size):
        N=X.shape[0]
        ELBO=self.exec_f(self.f['KL_X'], X)#maskとは長さＮの値Ｆａｌｓｅを持つ列
        
        minibatch = np.random.choice(N,batch_size,replace=False)
        
        LS, std = self.estimate(self.f['LL'], minibatch,X)
        ELBO += self.exec_f(self.f['KL_U'], X) + LS
        
        return ELBO, std
    

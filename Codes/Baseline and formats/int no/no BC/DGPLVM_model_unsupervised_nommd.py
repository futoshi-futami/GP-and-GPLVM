# To speed Theano up, create ram disk: mount -t tmpfs -o size=512m tmpfs /mnt/ramdisk
# Then use flag THEANO_FLAGS='base_compiledir=/mnt/ramdisk' python script.py
import sys; sys.path.insert(0, "../Theano"); sys.path.insert(0, "../../Theano")
import theano; import theano.tensor as T; import theano.sandbox.linalg as sT
import numpy as np
import pickle
from mlp import HiddenLayer
from optimizer import rmsprop
rng = np.random.RandomState(1234)

print ('Theano version: ' + theano.__version__ + ', base compile dir: ' + theano.config.base_compiledir)
#theano.config.mode = 'FAST_RUN'
#theano.config.optimizer = 'fast_run'
#theano.config.reoptimize_unpickled_function = False


#Ｑは隠れ層Ｘの次元。Ｄは観測地の次元
eps = 1e-4
class kernel(object):
    
    def __init__(self, Q):
        lhyp_values = np.zeros(Q+1,dtype=theano.config.floatX)
        self.lhyp = theano.shared(value=lhyp_values, name='lhyp', borrow=True)
        self.params = [self.lhyp]
        
        self.sf2,self.l = T.exp(self.lhyp[0]), T.exp(self.lhyp[1:1+Q])
        
    def RBF(self,X1, X2 = None):
        _X2 = X1 if X2 is None else X2
        dist = ((X1 / self.l)**2).sum(1)[:, None] + ((_X2 / self.l)**2).sum(1)[None, :] - 2*(X1 / self.l).dot((_X2 / self.l).T)
        RBF = self.sf2 * T.exp(-dist / 2.0)
        return (RBF + eps * T.eye(X1.shape[0])) if X2 is None else RBF
    def RBFnn(self, sf2, l, X):
        return sf2 + eps
    def LIN(self, sl2, X1, X2 = None):
        _X2 = X1 if X2 is None else X2
        LIN = sl2 * (X1.dot(_X2.T) + 1)
        return (LIN + eps * T.eye(X1.shape[0])) if X2 is None else LIN
    def LINnn(self, sl2, X):
        return sl2 * (T.sum(X**2, 1) + 1) + eps

class MMD(object):
    def __init__(self,M,Domain_number):
        Zlabel_values=np.zeros((M,Domain_number),dtype=theano.config.floatX)+np.log(1/Domain_number,dtype=theano.config.floatX)
        self.Zlabel = theano.shared(value=Zlabel_values, name='Zlabel', borrow=True)
        
        self.Zlabel_T=T.exp(self.Zlabel)/T.sum(T.exp(self.Zlabel),1)[:,None]
        
        ga_values = np.ones(1,dtype=theano.config.floatX)*np.log(0.1,dtype=theano.config.floatX)
        self.ga = theano.shared(value=ga_values, name='ga', borrow=True)
        
        self.params = [self.Zlabel,self.ga]
        
        self.gamma=T.exp(self.ga[0])
    
    def MMD_kenel_Xonly(self,Label,Knn,Weight):
        Dn=Label.shape[1]
        DD1=T.tile(Label.T, (Dn, 1,1))
        tttt=DD1[:,:,:,None]*DD1.transpose((1,0,2))[:,:,None,:]

        Hh=T.sum(T.sum(tttt*Knn[None,None,:,:],-1),-1)
        Hh=Hh*Weight
        
        GH=T.tile(T.diag(Hh),(Dn,1))
        new=T.exp(-(GH.T+GH-2*Hh)/(2*self.gamma**2))#ここまででＤ×ＤのＭＭＤ距離になった。次はＲＢＦカーネルにかける
        
        KK=tttt*new[:,:,None,None]
        #KK1=T.where(T.eq(KK,0),1,KK)#これはＺ用。Ｘは一つしか０じゃない数はないが、Ｚが複数ある。全てを重みつきでかけたいが、０があると0になっちゃうので1に変換する
        KK2=T.sum(T.sum(KK,0),0)
        Kmmd_rbf=KK2*Knn#RBFカーネルにかける
        return Kmmd_rbf
    
    def MMD_kenel_ZX(self,Xlabel,Kmn,Weight):
        Dn=self.Zlabel_T.shape[1]
        DDX=T.tile(Xlabel.T, (Dn, 1,1))
        DDZ=T.tile(self.Zlabel_T.T, (Dn, 1,1))
        
        tttt=DDZ[:,:,:,None]*DDX.transpose((1,0,2))[:,:,None,:]#10*10*N_Z*Nとか

        Hh=T.sum(T.sum(tttt*Kmn[None,None,:,:],-1),-1)
        Hh=Hh*Weight
        
        GH=T.tile(T.diag(Hh),(Dn,1))
        new=T.exp(-(GH.T+GH-2*Hh)/(2*self.gamma**2))#ここまででＤ×ＤのＭＭＤ距離になった。次はＲＢＦカーネルにかける
        
        KK=tttt*new[:,:,None,None]
        #KK1=T.where(T.eq(KK,0),1,KK)#これはＺ用。Ｘは一つしか０じゃない数はないが、Ｚが複数ある。全てを重みつきでかけたいが、０があると0になっちゃうので1に変換する
        KK2=T.sum(T.sum(KK,0),0)
        Kmmd_rbf=KK2*Kmn#RBFカーネルにかける
        return Kmmd_rbf
    
class DGGPLVM_model:
    def __init__(self,N_0,D, M,Q,Domain_number,Hiddenlayerdim1,Hiddenlayerdim2):
        
        self.Xlabel=T.matrix('Xlabel')

        self.Nt=N_0
        self.X=T.matrix('X')
        N=self.X.shape[0]
        self.start=T.iscalar()
        self.end=T.iscalar()
        self.Weight=T.matrix('Weight')

        ker=kernel(Q)
        #mmd=MMD(M,Domain_number)
        
        mu_value = np.random.randn(M,D)* 1e-2
        Sigma_b_value = np.zeros((M,M))# + np.log(0.01)
    
        Z_value = np.random.randn(M,Q)

        ls_value=np.zeros(1)+np.log(0.1)
        
        m_value=np.random.randn(N_0,Q)
        S_value=np.zeros((N_0,Q)) + np.log(0.1)
        
        self.mu = theano.shared(value=mu_value, name='mu', borrow=True)
        self.Sigma_b = theano.shared(value=Sigma_b_value, name='Sigma_b', borrow=True)
        self.Z = theano.shared(value=Z_value, name='Z', borrow=True)
        self.ls = theano.shared(value=ls_value, name='ls', borrow=True)
        self.m_b = theano.shared(value=m_value, name='m_b', borrow=True)
        self.S_b = theano.shared(value=S_value, name='S_b', borrow=True)
        
        self.params = [self.mu,self.Sigma_b,self.Z,self.ls]
        
        #self.hiddenLayer_x = HiddenLayer(rng=rng,input=self.X,n_in=D,n_out=Hiddenlayerdim1,activation=T.nnet.relu,number='_x')
        #self.hiddenLayer_hidden = HiddenLayer(rng=rng,input=self.hiddenLayer_x.output,n_in=Hiddenlayerdim1,n_out=Hiddenlayerdim2,activation=T.nnet.relu,number='_h')
        #self.hiddenLayer_m = HiddenLayer(rng=rng,input=self.hiddenLayer_x.output,n_in=Hiddenlayerdim1,n_out=Q,activation=T.nnet.relu,number='_m')
        #self.hiddenLayer_S = HiddenLayer(rng=rng,input=self.hiddenLayer_x.output,n_in=Hiddenlayerdim1,n_out=Q,activation=T.nnet.relu,number='_S')
        
        self.loc_params= [self.m_b,self.S_b]
        #self.loc_params.extend(self.hiddenLayer_x.params)
        #self.loc_params.extend(self.hiddenLayer_hidden.params)
        #self.loc_params.extend(self.hiddenLayer_m.params)
        #self.loc_params.extend(self.hiddenLayer_S.params)

        self.local_params={}
        for i in self.loc_params:
            self.local_params[str(i)]=i
        
        self.params.extend(ker.params)
        #self.params.extend(mmd.params)
        
        self.hyp_params={}
        for i in [self.mu,self.Sigma_b,self.ls]:
            self.hyp_params[str(i)]=i
        
        self.Z_params={}
        for i in [self.Z]:
            self.Z_params[str(i)]=i
                         
        self.global_params={}
        for i in self.params:
            self.global_params[str(i)]=i        
        
        #self.params.extend(self.hiddenLayer_x.params)
        #self.params.extend(self.hiddenLayer_hidden.params)
        #self.params.extend(self.hiddenLayer_m.params)
        #self.params.extend(self.hiddenLayer_S.params)
        self.params.extend(self.loc_params)
        
        self.wrt={}
        for i in self.params:
            self.wrt[str(i)]=i
        
        m=self.m_b[self.start:self.end]#self.hiddenLayer_m.output
        S_0=self.S_b[self.start:self.end]#self.hiddenLayer_S.output
        S_1=T.exp(S_0)
        S=T.sqrt(S_1)
        
        from theano.tensor.shared_randomstreams import RandomStreams
        srng = RandomStreams(seed=234)
        eps_NQ = srng.normal((N,Q))
        eps_M = srng.normal((M,D))#平均と分散で違う乱数を使う必要があるので別々に銘銘
        eps_ND = srng.normal((N,D))
                          
        beta = T.exp(self.ls)
        #uについては対角でないのでコレスキー分解するとかして三角行列を作る必要がある

        Sigma = T.tril(self.Sigma_b - T.diag(T.diag(self.Sigma_b)) + T.diag(T.exp(T.diag(self.Sigma_b))))
        
        #スケール変換
        mu_scaled, Sigma_scaled = ker.sf2**0.5 * self.mu, ker.sf2**0.5 * Sigma
        
        Xtilda = m + S * eps_NQ
        self.U = mu_scaled+Sigma_scaled.dot(eps_M)
        
        Kmm = ker.RBF(self.Z)
        #Kmm=mmd.MMD_kenel_Xonly(mmd.Zlabel_T,Kmm,self.Weight)
        KmmInv = sT.matrix_inverse(Kmm) 
        
        Kmn = ker.RBF(self.Z,Xtilda)
        #Kmn=mmd.MMD_kenel_ZX(self.Xlabel,Kmn,self.Weight)
        
        Knn = ker.RBF(Xtilda)
        #Knn=mmd.MMD_kenel_Xonly(self.Xlabel,Knn,self.Weight)
        
        Ktilda=Knn-T.dot(Kmn.T,T.dot(KmmInv,Kmn))
        
        
        F = T.dot(Kmn.T,T.dot(KmmInv,self.U)) + T.dot(T.maximum(Ktilda, 1e-16)**0.5,eps_ND)
        
        #Kinterval=T.dot(KmmInv,Kmn)
              
        mean_U=F#T.dot(Kinterval.T,self.U)
        Covariance = beta
        
        self.LL = self.log_mvns(self.X, mean_U, Covariance)# - 0.5*T.sum(T.dot(betaI,Ktilda)))            
        self.KL_X = -self.KLD_X(m,S)
        self.KL_U = -self.KLD_U(mu_scaled , Sigma_scaled , Kmm,KmmInv)
        
    
    def log_mvn(self, y, mean,beta):#対角ノイズ、YはＮ×Ｄのデータ,それの正規分布の対数尤度
        N = y.shape[0]
        D = y.shape[1]
        return -0.5 *  D * T.sum(T.log(2 * np.pi*(1/T.diag(beta)))) - 0.5 * T.sum(T.dot(beta,(y - mean)**2))
    
    def log_mvns(self, y, mean,beta):#対角ノイズ、YはＮ×Ｄのデータ,それの正規分布の対数尤度
        N = y.shape[0]
        D = y.shape[1]
        return -0.5 *  D * T.sum(T.log(2 * np.pi*(1/beta))) - 0.5 * T.sum(beta*T.sum((y - mean)**2))
    
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

    def estimate_f(self, i, samples = 200):
        
        #minibatch = np.random.choice(X,batch_size,replace=False)
        
        f_acc = np.array([self.f['LL'](i) for s in range(samples if samples != None else self.samples)])
        
        return np.nanmean(f_acc, 0), np.nanstd(f_acc, 0) 
    
    def estimate(self,param,i, samples = 50):
        
        #minibatch = np.random.choice(X,batch_size,replace=False)
        
        f_acc = np.array([self.g[param]['LL'](i) for s in range(samples if samples != None else self.samples)])
        
        return np.nanmean(f_acc, 0), np.nanstd(f_acc, 0) 
        #nanをつけた平均と分散、ベクトル中に欠損値 NaN が含まれている場合、以下のようなメソッドを利用することで、欠損値を無視して計算を行うことができる。

    def ELBO(self, X, batch_size):
        N=X.shape[0]
        ELBO_X=self.exec_f(self.f['KL_X'], X)#maskとは長さＮの値Ｆａｌｓｅを持つ列
        
        minibatch = np.random.choice(N,batch_size,replace=False)
        ELBO_U=self.exec_f(self.f['KL_U'], X) 
        LS, std = self.estimate(self.f['LL'], minibatch,X)
        ELBO = ELBO_X +ELBO_U+LS
        return LS

    
    def compile_F(self,train_set_x,train_weight,train_label,batch_size):
        
        index=T.iscalar()
        print ('Modeling...')
        
        model_file_name = 'model2' + '.save'
                                    #もしこれまでに作ったのがあるならロードする
        try:
            print ('Trying to load model...')
            with open(model_file_name, 'rb') as file_handle:
                obj = pickle.load(file_handle)
                self.f, self.g= obj
                print ('Loaded!')
            return
        except:
            print ('Failed. Creating a new model...')
        
        self.f = {n: theano.function([], f, name=n, givens={self.X: train_set_x,self.Xlabel: train_label,self.Weight: train_weight,self.start:0,self.end:self.Nt
                },on_unused_input='ignore') for n,f in zip(['U', 'KL_U', 'KL_X'], [self.U, self.KL_U, self.KL_X])}
        
        self.f['LL']=theano.function(
                [index],
                outputs=self.LL,
                givens={
                self.X: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.Xlabel: train_label[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.Weight: train_weight
                ,self.start:index*batch_size
                ,self.end:(index+1)*batch_size
                },
                on_unused_input='ignore'
            )

        z= 0.0*sum([T.sum(v) for v in self.params])   
        self.g = {vn: {gn: theano.function([], T.grad(gv+z, vv), name='d'+gn+'_d'+vn, givens={self.X: train_set_x,self.Xlabel: train_label,self.Weight: train_weight,self.start:0,self.end:self.Nt
                },
            on_unused_input='ignore') for gn,gv in zip(['KL_U', 'KL_X'], [self.KL_U, self.KL_X])} for vn, vv in self.wrt.items()}
            
        for vn, vv in self.wrt.items():
            self.g[vn]['LL'] = theano.function([index], T.grad(self.LL+z, vv), name='dLL'+'_d'+vn, givens={
                self.X: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.Xlabel: train_label[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.Weight: train_weight
                ,self.start:index*batch_size
                ,self.end:(index+1)*batch_size
                },
            on_unused_input='ignore')  
            
        with open(model_file_name, 'wb') as file_handle:
            print ('Saving model...')
            sys.setrecursionlimit(100000)
            pickle.dump([self.f, self.g], file_handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    def import_F(self,train_set_x,train_weight,train_label,batch_size):
        
        index = T.iscalar()
        
        print ('import_Modeling...')
        
        self.f = {n: theano.function([], f, name=n, givens={self.X: train_set_x,self.Xlabel: train_label,self.Weight: train_weight
                },on_unused_input='ignore') for n,f in zip(['U', 'KL_U', 'KL_X'], [self.U, self.KL_U, self.KL_X])}
        
        self.f['LL']=theano.function(
                [index],
                outputs=self.LL,
                givens={
                self.X: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.Xlabel: train_label[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.Weight: train_weight
                },
                on_unused_input='ignore'
            )

        z= 0.0*sum([T.sum(v) for v in self.params])   
        self.g = {vn: {gn: theano.function([], T.grad(gv+z, vv), name='d'+gn+'_d'+vn, givens={self.X: train_set_x,self.Xlabel: train_label,self.Weight: train_weight
                },
            on_unused_input='ignore') for gn,gv in zip(['KL_U', 'KL_X'], [self.KL_U, self.KL_X])} for vn, vv in self.wrt.items()}
            
        for vn, vv in self.wrt.items():
            self.g[vn]['LL'] = theano.function([index], T.grad(self.LL+z, vv), name='dLL'+'_d'+vn, givens={
                self.X: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.Xlabel: train_label[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.Weight: train_weight
                },
            on_unused_input='ignore')  
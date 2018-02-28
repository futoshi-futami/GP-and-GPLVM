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

class MMD:
    def MMD_kenel_Xonly(self,gamma,Label,Knn,Weight):
        Dn=Label.shape[1]
        DD1=T.tile(Label.T, (Dn, 1,1))
        tttt=DD1[:,:,:,None]*DD1.transpose((1,0,2))[:,:,None,:]

        Hh=T.sum(T.sum(tttt*Knn[None,None,:,:],-1),-1)
        Hh=Hh*Weight
        
        GH=T.tile(T.diag(Hh),(Dn,1))
        new=T.exp(-(GH.T+GH-2*Hh)/(2*gamma**2))#ここまででＤ×ＤのＭＭＤ距離になった。次はＲＢＦカーネルにかける
        
        KK=tttt*new[:,:,None,None]
        #KK1=T.where(T.eq(KK,0),1,KK)#これはＺ用。Ｘは一つしか０じゃない数はないが、Ｚが複数ある。全てを重みつきでかけたいが、０があると0になっちゃうので1に変換する
        KK2=T.sum(T.sum(KK,0),0)
        Kmmd_rbf=KK2*Knn#RBFカーネルにかける
        return Kmmd_rbf
    
    def MMD_kenel_ZX(self,gamma,Zlabel,Xlabel,Kmn,Weight):
        Dn=Zlabel.shape[1]
        DDX=T.tile(Xlabel.T, (Dn, 1,1))
        DDZ=T.tile(Zlabel.T, (Dn, 1,1))
        
        tttt=DDZ[:,:,:,None]*DDX.transpose((1,0,2))[:,:,None,:]#10*10*N_Z*Nとか

        Hh=T.sum(T.sum(tttt*Kmn[None,None,:,:],-1),-1)
        Hh=Hh*Weight
        
        GH=T.tile(T.diag(Hh),(Dn,1))
        new=T.exp(-(GH.T+GH-2*Hh)/(2*gamma**2))#ここまででＤ×ＤのＭＭＤ距離になった。次はＲＢＦカーネルにかける
        
        KK=tttt*new[:,:,None,None]
        #KK1=T.where(T.eq(KK,0),1,KK)#これはＺ用。Ｘは一つしか０じゃない数はないが、Ｚが複数ある。全てを重みつきでかけたいが、０があると0になっちゃうので1に変換する
        KK2=T.sum(T.sum(KK,0),0)
        Kmmd_rbf=KK2*Kmn#RBFカーネルにかける
        return Kmmd_rbf
    
class DGGPLVM_model:
    def __init__(self, params,correct,Xinfo, samples = 500,batch_size=None):
        ker = kernel()
        mmd = MMD()
        self.samples = samples
        self.params =  params
        self.batch_size=batch_size
        self.Xlabel_value=Xinfo["Xlabel_value"]
        self.Weight_value=Xinfo["Weight_value"]
        
        #データの保存ファイル
        model_file_name = 'model_MMD_kernel' + '.save'
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
        
        X,Y,X_test,m,S_b,mu,Sigma_b,Z,eps_NQ,eps_M =\
        T.dmatrices('X','Y','X_test','m','S_b','mu','Sigma_b','Z','eps_NQ','eps_M')
        
        Xlabel=T.dmatrix('Xlabel')
        Zlabel=T.dmatrix('Zlabel')
        
        Zlabel_T=T.exp(Zlabel)/T.sum(T.exp(Zlabel),1)[:,None]#ラベルは確率なので正の値でかつ、企画化されている
        
        Weight=T.dmatrix('Weight')
        
        lhyp = T.dvector('lhyp')
        ls=T.dvector('ls')
        ga=T.dvector('ga')
        
        (M, D), N, Q = Z.shape, X.shape[0], X.shape[1]

        
        #変数の正の値への制約条件
        beta = T.exp(ls)
        gamma=T.exp(ga[0])
        #beta=T.exp(lhyp[0])
        sf2, l = T.exp(lhyp[0]), T.exp(lhyp[1:1+Q])
        
        S=T.exp(S_b)
        #Sigma=T.exp(self.Sigma_b)
        
        #xについてはルートを取らなくても対角行列なので問題なし
        #uについては対角でないのでコレスキー分解するとかして三角行列を作る必要がある
        Sigma = T.tril(Sigma_b - T.diag(T.diag(Sigma_b)) + T.diag(T.exp(T.diag(Sigma_b))))
        
        #スケール変換
        mu_scaled, Sigma_scaled = sf2**0.5 * mu, sf2**0.5 * Sigma
        
        Xtilda = m + S * eps_NQ
        U = mu_scaled+Sigma_scaled.dot(eps_M)

        print ('Setting up cache...')
        
        Kmm = ker.RBF(sf2, l, Z)
        Kmm=mmd.MMD_kenel_Xonly(gamma,Zlabel_T,Kmm,Weight)
        KmmInv = sT.matrix_inverse(Kmm) 
        #KmmDet=theano.sandbox.linalg.det(Kmm)
        
        #KmmInv_cache = sT.matrix_inverse(Kmm)
        #self.fKmm = theano.function([Z, lhyp], Kmm, name='Kmm')
        #self.f_KmmInv = theano.function([Z, lhyp], KmmInv_cache, name='KmmInv_cache')
        #復習：これは員数をＺ，lhypとした関数kmmInv_cacheをコンパイルしている。つまり逆行列はｚとハイパーパラメタの関数になった
        #self.update_KmmInv_cache()#実際に数値を入れてkinnvを計算させている
        #逆行列の微分関数を作っている
        
        #self.dKmm_d = {'Z': theano.function([Z, lhyp], T.jacobian(Kmm.flatten(), Z), name='dKmm_dZ'),
        #               'lhyp': theano.function([Z, lhyp], T.jacobian(Kmm.flatten(), lhyp), name='dKmm_dlhyp')}

        
        print ('Modeling...')
        
        Kmn = ker.RBF(sf2,l,Z,Xtilda)
        Kmn=mmd.MMD_kenel_ZX(gamma,Zlabel_T,Xlabel,Kmn,Weight)
        
        Knn = ker.RBF(sf2,l,Xtilda,Xtilda)
        Knn=mmd.MMD_kenel_Xonly(gamma,Xlabel,Knn,Weight)
        
        Ktilda=Knn-T.dot(Kmn.T,T.dot(KmmInv,Kmn))
        
        Kinterval=T.dot(KmmInv,Kmn)
              
        mean_U=T.dot(Kinterval.T,U)
        betaI=T.diag(T.dot(Xlabel,beta))
        Covariance = betaI       
        
        LL = (self.log_mvn(X, mean_U, Covariance) - 0.5*T.sum(T.dot(betaI,Ktilda)))*correct              
        KL_X = -self.KLD_X(m,S)*correct
        KL_U = -self.KLD_U(mu_scaled , Sigma_scaled , Kmm,KmmInv)
        
        print ('Compiling model ...')        


        inputs = {'X': X, 'Z': Z, 'm': m, 'S_b': S_b, 'mu': mu, 'Sigma_b': Sigma_b, 'lhyp': lhyp, 'ls': ls, 
            'eps_M': eps_M, 'eps_NQ': eps_NQ,'ga':ga,'Zlabel':Zlabel,'Weight':Weight,'Xlabel':Xlabel}
        
        z = 0.0*sum([T.sum(v) for v in inputs.values()]) # solve a bug with derivative wrt inputs not in the graph
        
        self.f = {n: theano.function(list(inputs.values()), f+z, name=n, on_unused_input='ignore')\
                  for n,f in zip(['X', 'U', 'LL', 'KL_U', 'KL_X'], [X, U, LL, KL_U, KL_X])}
        
        
        wrt = {'Z': Z, 'm': m, 'S_b': S_b, 'mu': mu, 'Sigma_b': Sigma_b, 'lhyp': lhyp, 'ls': ls,'ga':ga,'Zlabel':Zlabel}
        self.g = {vn: {gn: theano.function(list(inputs.values()), T.grad(gv+z, vv), name='d'+gn+'_d'+vn, 
            on_unused_input='ignore') for gn,gv in zip(['LL', 'KL_U', 'KL_X'], [LL, KL_U, KL_X])} for vn, vv in wrt.items()}

        with open(model_file_name, 'wb') as file_handle:
            print ('Saving model...')
            sys.setrecursionlimit(10000)
            pickle.dump([self.f, self.g], file_handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    
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
              
    def update_KmmInv_cache(self):#実際に計算させるため
        self.KmmInv = self.f_KmmInv(self.params['Z'], self.params['lhyp']).astype(theano.config.floatX)


    def exec_f(self, f, X = [[0]], minibatch=None):
        #params=['m','S_b','mu','Sigma_b','Z','lhyp','ls']

        inputs={}
        (M, D)= self.params['mu'].shape 
        (N, Q)=self.params['m'].shape
        
        inputs['m']= self.params['m']
        inputs['S_b']= self.params['S_b']

        inputs['mu']=self.params['mu']
        inputs['Sigma_b'] = self.params['Sigma_b']
        
        inputs['Z'] = self.params['Z']
        inputs['lhyp']= self.params['lhyp']
        inputs['ls']= self.params['ls']
        inputs['ga']= self.params['ga']

        inputs['Zlabel']= self.params['Zlabel']
        
        inputs['eps_M']= np.random.randn(M,D)
        inputs['eps_NQ']= np.random.randn(N,Q)
                
        inputs['X'] = X
        inputs['Xlabel']=self.Xlabel_value
        inputs['Weight']=self.Weight_value
        
        if minibatch is not None:
            inputs['X']= inputs['X'][minibatch]
            inputs['m']= inputs['m'][minibatch]
            inputs['S_b'] = inputs['S_b'][minibatch]
            inputs['eps_NQ']= inputs['eps_NQ'][minibatch] 
            inputs['Xlabel']=inputs['Xlabel'][minibatch]
        
        return f(**inputs)


    def estimate(self, f, minibatch, X = [[0]], samples = 50):
        
        #minibatch = np.random.choice(X,batch_size,replace=False)
        
        f_acc = np.array([self.exec_f(f, X, minibatch=minibatch) for s in range(samples if samples != None else self.samples)])
        
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
    

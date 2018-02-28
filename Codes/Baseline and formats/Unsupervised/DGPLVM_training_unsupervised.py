import sys; sys.path.append("./")

from DGPLVM_opt_unsupervised_nommd import DGGPLVM_opt
import pickle
import numpy as np

import time
from numpy.random import randn, rand
from pandas import Series, DataFrame
import pandas as pd
from scipy.spatial.distance import pdist,squareform
from scipy import exp
from scipy.linalg import eigh

Dataset_originall=pd.read_csv('../parkinson.csv')#ファイルの読み込み
Dataset=Dataset_originall.values#DataframeオブジェクトをNumpy配列に入れる

m=5#特徴空間の射影は次元でとってくるのか
sigma=10#カーネル関数の分散
beta=0.5#ガウス回帰の値

#42このデータのうち30個を学習に、12個をテストに使う
from numpy import random
Domain_number=30
training_person=random.choice(list(range(1,43)),Domain_number,replace=False)#重複なしに取り出す
training_person.sort()#ガウス過程で突っ込む際に順番に並んであった方がわかりやすいので
#まず各人のデータを区別して保存していきます
#
#
#

Training_Dataset=[]#トレーニングデータ用の人のデータをここに格納していきます
for i in training_person:
    X=Dataset_originall[Dataset_originall['subject#']==i]#subjectが1（1番の人）全体をdataframeそのものとして取り出す
    X1=[X.iloc[:,4].values,X.iloc[:,5].values,X.iloc[:,6:].values]#取り出したDataFrameからラベルｙと特徴量Xを取り出す
    Training_Dataset.append(X1)
#[[1番目の人のデータ]、[2番目の人のデータ]・・・・]とリストになっている
#[1番目の人のデータ]＝[array[yの値＊149個分]、array[y２の値*149]、array[Xの値[[149],[16]]と格納されている。

#testdataの方も格納していきます
test_person=set(range(1,43))-set(training_person)#test用の人の番号
test_person=list(test_person)
test_person.sort()
Test_Dataset=[]
for i in test_person:
    X=Dataset_originall[Dataset_originall['subject#']==i]#subjectが1（1番の人）全体をdataframeそのものとして取り出す
    X1=[X.iloc[:,4].values,X.iloc[:,5].values,X.iloc[:,6:].values]#取り出したDataFrameからラベルｙと特徴量Xを取り出す
    Test_Dataset.append(X1)

#    
#次に各人のデータを区別せずにまとめて入れます
#
training_person_set=set(training_person)

TRAININGDATA=DataFrame()
for i in training_person:
    TRAININGDATA =TRAININGDATA.add(Dataset_originall[Dataset_originall['subject#']==i],fill_value=0)
X_training=TRAININGDATA.iloc[:,6:].values
y1_training=TRAININGDATA.iloc[:,4].values
y2_training=TRAININGDATA.iloc[:,5].values
    
TESTDATA=DataFrame()
for i in test_person:
    TESTDATA =TESTDATA.add(Dataset_originall[Dataset_originall['subject#']==i],fill_value=0)
X_test=TESTDATA.iloc[:,6:].values
y1_test=TESTDATA.iloc[:,4].values
y2_test=TESTDATA.iloc[:,5].values


#特徴量の標準化
def Scaler(X):
    mu=mu=np.mean(X,axis=0)#平均
    sigma=np.sqrt(np.mean((X-mu)*(X-mu),axis=0))#分散
    return (X-mu)/sigma

X_training_0=Scaler(X_training)
X_test_0=Scaler(X_test)


#順番にそれぞれの人の中で正規化していきます
for i in range(len(Training_Dataset)):
    Training_Dataset[i][2]=Scaler(Training_Dataset[i][2])
    Training_Dataset[i][1]=Scaler(Training_Dataset[i][1])
    Training_Dataset[i][0]=Scaler(Training_Dataset[i][0])

N_p=len(Training_Dataset)#トレーニングに用いる確率分布（人の数）
N=len(X_training_0)#トレーニングに用いるすべてのデータの数

#test用の人にも同じようにやりましょう
for i in range(len(Test_Dataset)):
    Test_Dataset[i][2]=Scaler(Test_Dataset[i][2])
    Test_Dataset[i][1]=Scaler(Test_Dataset[i][1])
    Test_Dataset[i][0]=Scaler(Test_Dataset[i][0])

N_p_t=len(Test_Dataset)#testに用いる確率分布（人の数）
N_t=len(X_test_0)#テストに用いるすべてのデータの数
   
D=len(Training_Dataset)
l=[]
for i in range(len(Training_Dataset)):
    n=len(Training_Dataset[i][2])
    d=np.zeros((n,D))
    d[:,i]=1
    l.append(d)

Xlabel=l[0]
for i in range(1,len(Training_Dataset)):
    Xlabel=np.vstack((Xlabel,l[i]))
       
Size=[]
for i in training_person:
    l=len(Dataset_originall[Dataset_originall['subject#']==i])
    Size.append(l)
Weight=np.array(Size)[:,None]*np.array(Size)[None,:]
Weight=1/Weight

#計算用のボックスの作成3次元の直方体的な
#Xdomain=np.tile(Xlabel.T, (Domain_number, 1,1))
#N×Ｎ殻の変換用
#tttt=Xdomain[:,:,:,None]*Xdomain.transpose((1,0,2))[:,:,None,:]
       
       
       
#ここまでがデータの読み込みと前処理
#
#
#
#
#計算高速化のためのある行列を与えたときにそのすべてのペアのユークリッド距離をN×Nでかえす
def GetDistanceMatrix(X):
    H = np.tile(np.diag(np.dot(X, X.T)) , (np.size(X , 0) , 1))
    G = np.dot(X , X.T)
    DistMat = H - 2 * G + H.T
    return DistMat

#違う行列与えられたときにも対応X=N1,Y=N2個のデータが含まれているとき、N1×N2行列を返す
def GetDistanceMatrix2(X,Y):
    H = np.tile(np.diag(np.dot(X, X.T)) , (np.size(Y , 0) , 1))
    H1 =np.tile(np.diag(np.dot(Y, Y.T)) , (np.size(X , 0) , 1))
    G = np.dot(X , Y.T)
    DistMat = H.T - 2 * G + H1
    return DistMat
#上で計算したのは単なるユークリッド距離の行列なのでexp-を作用させておく
def hyperK(sigma):
    return lambda X, Y: np.exp(-GetDistanceMatrix2(X,Y)/(2*sigma**2))
KERNEL=hyperK(sigma)
#2とおりの方法で得られたときに要素の形があっているかチェックする機械
#S=0
#for j in range(30):
#    for i in range(30):
#        if not K2_H[j][i].shape==K2[j][i].shape:
#            S +=1
#とりあえずカーネル関数の一般形を置いておきます
def gaussian_kernel(sigma):
    return lambda x1, x2: np.exp(-np.sum((x1-x2)**2) / (2 * sigma ** 2))
#とりあえずこいつをカーネルにしとく

kernel=gaussian_kernel(sigma)
#学習に使ったグラム行列K=k(xn,xm)と新しい入力に対するカーネル関数の値k(x_new,x_old)とk(x_new,x_new)を計算する必要


g=[]
k_bet=[]
for i in range(N_p):
    for j in range(N_p):
        wo=KERNEL(Training_Dataset[i][2],Training_Dataset[j][2])
        #woというのはあるPiとPjについてそのPiの中のｋ成分とpｊの中のｌ成分についてカーネルをとっています。
        #i番目の人のデータはリストのi番目であるTraining_Dataset[i]に入っておりXデータは[2]で指定します。
        #最後にi番目の人の持っているデータ総数len(Training_Dataset[i][2])と同じくｊ番目の人の持っているデータ総数でreshapeします.
        g.append(np.sum(wo)/(len(Training_Dataset[i][2])*len(Training_Dataset[j][2])))#このｇはGijです。ｋとｌ成分について和をとるのでsumをとっています.
        k_bet.append(wo)
K_p=np.array(k_bet).reshape(N_p,N_p)#k2にあたります。ひとつ上でリストに横並びに入れたので、ここでちゃんとreshapeします。k2のことです。
G=np.array(g).reshape(N_p,N_p)#ｇについてもリストに横一列に格納しているのでreshapeして使いやすくしておきます。

#次にk1の値を計算します.K1は論文からGijの成分から計算できるとされているのでそのとおり計算します。
k1=[]
for i in range(N_p):
    for j in range(N_p):
        k1.append(np.exp(-(G[i][i]+G[j][j]-2*G[i][j])/(2*3**2)))
K1=np.array(k1).reshape(N_p,N_p)

#さいごにi,jについて要素ごとの積を考えてカーネルの導出を終えます
K2=K_p#pooling SVMとかに使うってことでｋ１の分布間のカーネルは考えません
Kdist=K1*K2#どちらもN_p×N_pのndarrayで構成されていることを利用し掛け算で処理できます。

K_tot=[]
for i in range(N_p):#それぞれの行ごとに呼び出して、横方向にくっつけていきます。するとN_p×N行列がまず得られます。
    J=Kdist[i][0]
    for j in range(1,N_p):
        J=np.hstack((J,Kdist[i][j]))
    K_tot.append(J)
    Ktot=np.array(K_tot)

#次に縦方向をくっつける    
JJ=Ktot[0]#上でN_p×N行列になっているところを縦方向も連結してしまうことでN×Nにします。
for i in range(1,N_p):
    JJ=np.vstack((JJ,Ktot[i]))
Ktrue=JJ

#以上データの読み込み--------------------------
from scipy.optimize import minimize

N=2000
X=X_training_0[:N]
X_test=X_training_0[100:150]
Y1=y1_training[:N]
Y_test=y1_training[100:150]
Y1=Y1[:,None]
Y2=y2_training[:100]

N=X.shape[0]
D=X.shape[1]

D=16
M = 100
Hiddenlayerdim1=30
Hiddenlayerdim2=10
Q = 6
max_iteration = 1000
batch_size=50


#初期値の設定
from PCA_EM import PCA_EM_missing

try:
    print ('Trying to load parameters...')
    with open(str(Q)+'_m.dump', 'rb') as file_handle:
        obj = pickle.load(file_handle)
        m= obj
        print ('Loaded!')
except:
    print ('Failed. Calculate parameters...')
    m = PCA_EM_missing(X,Q)
    m.learn(100)
    m=m.m_Z
    with open(str(Q)+'_m.dump', 'wb') as f:
        pickle.dump(m, f)



Xinfo={'Xlabel_value':Xlabel,'Weight_value':Weight}

#dggplvm = DGGPLVM_model(D,M,Q,Domain_number)

#import time
#start = time.time()
#LL=np.mean(np.array([LL(1) for s in range(500)]))
#elapsed_time = time.time() - start
#print(elapsed_time)
import theano
X_tr = theano.shared(X)
Weight_share = theano.shared(Weight)
Xlabel_share = theano.shared(Xlabel)
m_tr=theano.shared(m)

try:
    print ('Trying to load pre_parameters...')
    with open('pre_params.dump', 'rb') as file_handle:
        obj = pickle.load(file_handle)
        pre_params= obj
        print ('Loaded!')
except:
    print ('Failed. Calculate back_constrained_parameters...')
    from pretraining_BC import back_constrained_model
    preBC=back_constrained_model(D,Q,Hiddenlayerdim1,Hiddenlayerdim2)
    preBC.pretraining_mlp(X_tr,m_tr)

    def pre_tr():
        start = time.time()
        n_train_batches = X.shape[0] // batch_size
        iteration=0
        while iteration < 15000:
            index=random.choice(n_train_batches)
            preBC.training_model(index)
            iteration +=1
            if iteration%2000 ==0:
                print ('iter ' + str(iteration) + ': current error is:' + str(preBC.training_model(index)))
                iteration +=1
                elapsed_time = time.time() - start
                print(elapsed_time)
    
    if __name__ == '__main__':
        pre_tr()
    saving_params={}
    for i,j in enumerate(preBC.params):
        saving_params[str(j)]=preBC.params[i].get_value()
    with open('pre_params.dump', 'wb') as f:
        pickle.dump(saving_params, f)
    pre_params=saving_params

try:
    print ('Trying to load pre_U_parameters...')
    with open('Pre_Uparams.dump', 'rb') as file_handle:
        obj = pickle.load(file_handle)
        Pre_U= obj
        print ('Loaded!')
except:
    print ('Failed. Calculate U_parameters...')
    from pretrainingK import pretraining
    minibatch = np.random.choice(N,M,replace=False)
    PRE=pretraining(0.1,0.1,0.1,m,m[minibatch],X)
    Mu,Sigma=PRE.preU()
    Pre_U={'mu':Mu,'Sigma_b':Sigma}
    with open('Pre_Uparams.dump', 'wb') as f:
        pickle.dump(Pre_U, f)
dggplvm = DGGPLVM_opt(D,M,Q,Domain_number,X_tr,Weight_share,Xlabel_share,50,m,pre_params,Pre_U,Hiddenlayerdim1,Hiddenlayerdim2)


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
            LL=dggplvm.dggplvm.estimate_f(index,50)
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
    while iteration < 1000:
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

#if __name__ == '__main__':
#    onestep_opt()
        

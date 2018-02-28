import sys; sys.path.append("./")
from DGPLVM_theano_opt import DGGPLVM_opt

import numpy as np


from numpy.random import randn, rand
from pandas import Series, DataFrame
import pandas as pd
from scipy.spatial.distance import pdist,squareform
from scipy import exp
from scipy.linalg import eigh

Dataset_originall=pd.read_csv('parkinson.csv')#ファイルの読み込み
Dataset=Dataset_originall.values#DataframeオブジェクトをNumpy配列に入れる

m=5#特徴空間の射影は次元でとってくるのか
sigma=10#カーネル関数の分散
beta=0.5#ガウス回帰の値

#42このデータのうち30個を学習に、12個をテストに使う
from numpy import random
training_person=random.choice(list(range(1,43)),30,replace=False)#重複なしに取り出す
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

X=X_training_0[:300]
X_test=X_training_0[100:150]
Y1=y1_training[:300]
Y_test=y1_training[100:150]
Y1=Y1[:,None]
Y2=y2_training[:100]

N=X.shape[0]
D=X.shape[1]

def test(X,Y,N,D, M = 100, Q = 5,  max_iteration = 20, batch_size=100):
    # Random initialisation 
    #K=1であるのでＹの各ＤはＫ＝０，１の2つを取れる
        
    D=16

    #初期値の設定
    m = np.random.randn(N,Q)
    S_b = np.zeros((N,Q)) + np.log(0.1)
    
    mu = np.random.randn(M,D)
    Sigma_b = np.zeros((M,D)) + np.log(0.01)

    Z = np.random.randn(M,Q)

    lhyp = np.zeros((Q+1,1)) + np.log(0.1)
    ls=np.zeros((1,1)) + np.log(0.1)

    params = {'m':m,'S_b':S_b,'mu':mu,'Sigma_b':Sigma_b,'Z':Z,'lhyp':lhyp,'ls':ls}
    

    dggplvm = DGGPLVM_opt(params, np.array([[0]]),batch_size=batch_size)
    
    dggplvm.X = X

    print ('Optimising...')
    iteration = 0
    
    while iteration < max_iteration:
        if batch_size is None:
            batch_size = N
        else:
            batch_size = batch_size
            
        minibatch = np.random.choice(N,batch_size,replace=False)
        
        dggplvm.opt_one_step(params.keys(), minibatch)
        current_ELBO = dggplvm.ELBO(X)
        print ('iter ' + str(iteration) + ': ' + str(current_ELBO[0]) + ' +- ' + str(current_ELBO[1]))
        #if iteration%10 == 0:
        #    print ('error ' + str(get_error(clgp, clgp.Y, ~clgp.mask)[0]))
        #iteration += 1

test(X,Y1,N,D, M = 100, Q = 5,  max_iteration = 20, batch_size=100)    
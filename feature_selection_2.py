import numpy as np
import time
from database import *
from sklearn.model_selection import  train_test_split
import asyncio
loop = asyncio.get_event_loop()
import ast
import warnings
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")
import pandas as pd
from functools import reduce

def predata():
    #df = pd.read_csv('data/Heart-C.csv', header=None)
    #df = df.as_matrix()
    from scipy import io
    mat = io.loadmat('CLL_SUB_111')   #COIL20    1000维
    X = mat['X']  # data
    y = mat['Y']  # label
    y = y[:, 0]
    #X = df[:, 0:-1]
    #y = df[:, -1]
    #y=np.where(y=='democrat',1,0)
    #X=X.astype(np.float)
    n_samples, n_features = np.shape(X)
    print(n_samples, n_features)



    t=0 #数据是否需要转换
    if t==1:
        for x in range(X.shape[1]):
            max = np.max(X[:,x])
            min = np.min(X[:,x])
            if max == min:
                X[:,x] = 0
            else:
                X[:,x] = (X[:,x] - min) / (max - min)

    r = []
    for x in range(n_features):
        x = X[:, x]
        t = np.vstack((x, y))
        r1 = np.corrcoef(t)[0, 1]
        if np.isnan(r1):
            r1 = 0.0
        r1 =-r1
        r.append(r1)
    idx = np.argsort(r)
    idx=idx.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
    return X_train, X_test, y_train, y_test,idx


def serch_root(X_train, X_test, y_train, y_test,idx=[],abacc=0.0,idxx=[]):


    #idx  当前的初始节点
    #abacc   初始节点精度
    #idxx  搜索范围
    ai=[]

    aiacc=[]
    for i in idxx:
        if i in idx:
            continue
        else:
            idx.append(i)
        selected_features_train = X_train[:,idx]

        selected_features_test = X_test[:,idx]


        from sklearn import svm
        from sklearn.svm import SVC
        #clf = SVC()#采用高斯核函数
        clf = svm.LinearSVC()
        clf.fit(selected_features_train, y_train)

        y_predict = clf.predict(selected_features_test)

        acc = accuracy_score(y_test, y_predict)

        #删除相同元素
        l2 = []
        [l2.append(i) for i in idx if not i in l2]
        idx=l2

        if acc>(abacc+1e-4):
            abacc = acc
            k=idx.copy()   #为什么要写这句，是因为copy了一个实体，而非指针
            ai.append(k)
            aiacc.append(float('%.5f' % acc))
        else:
            pass
        idx.pop()
    return ai,aiacc



if __name__ == '__main__':
    start = time.time()
    X_train, X_test, y_train, y_test, idx = predata()
    n_samples, n_features = np.shape(X_train)
    ai_ai=[]
    acc_acc=[0]
    idx1 = idx[0:int(len(idx) / 3)]
    #固定前端
    while True:
        # 第一次和算法交互
        ai, aiacc =serch_root(X_train, X_test, y_train, y_test,idx=ai_ai,abacc=acc_acc[-1], idxx=idx1)
        if aiacc==[]:
            break
        ai = reduce(lambda x, y: x + y, ai)
        ai=ai.pop()
        if ai in idx1:
            t=idx1.index(ai)
            if t<int(len(idx1))/3:
                pass
            else:
                if int(len(idx))>int(len(idx1) * 1.5):
                    idx1 = idx[0:int(len(idx1)*1.5)]
                else:
                    idx1 = idx

        else:
            idx1 = idx
        if ai in ai_ai:
            pass
        else:
            ai_ai.append(ai)
        acc_acc.append(aiacc[-1])
        print(ai_ai)
        print(acc_acc[-1])
    #找后端
    while True:
        # 第一次和算法交互
        ai, aiacc =serch_root(X_train, X_test, y_train, y_test,idx=ai_ai,abacc=acc_acc[-1], idxx=idx)
        if aiacc==[]:
            break
        ai = reduce(lambda x, y: x + y, ai)
        ai=ai.pop()
        if ai in ai_ai:
            pass
        else:
            ai_ai.append(ai)
        acc_acc.append(aiacc[-1])
    new = time.time()
    print('全程分钟：')
    print((new - start))
    print(ai_ai)
    print(acc_acc[-1])
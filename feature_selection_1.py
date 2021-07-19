import numpy as np
from sklearn.model_selection import  train_test_split
import warnings
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")

import pandas as pd

#df = pd.read_csv('data/Hillvalley.csv', header=None)
#df = df.as_matrix()

#X = df[:, 0:-1]
#y = df[:, -1]
#y=np.where(y==True,1,0)
from scipy import io
mat = io.loadmat('data/PCMAC')   #COIL20    1000维
X = mat['X']  # data
y = mat['Y']  # label
X=X.toarray()
y = y[:, 0]

n_samples, n_features = np.shape(X)

t=0#数据是否需要转换
if t==1:
    for x in range(X.shape[1]):
        max = np.max(X[:,x])
        min = np.min(X[:,x])
        if max == min:
            X[:,x] = 0
        else:
            X[:,x] = (X[:,x] - min) / (max - min)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

print('-----------------------------------------')
#idx = [i for i in range(n_features-1)]

r = []
for x in range(n_features):
        x = X[:, x]
        t = np.vstack((x, y))
        r1 = np.corrcoef(t)[0, 1]
        if np.isnan(r1) :
            r1 = 0.0
        r1 = -r1
        r.append(r1)
#idx = np.argsort(r)
X=np.where(X==0,0,1)
t = np.sum(X, axis=0)
idx = np.argsort(-t)

aic = []
aiacccb = 0
aiaccc = []
idxxc =[]
j=-1
c=[0]
k=[]
while(True):
    j+=1
    m = -1
    for i in idx:
        m += 1
        idxxc.append(i)
        selected_features_train = X_train[:, idxxc]

        selected_features_test = X_test[:, idxxc]

        from sklearn import svm
        clf =svm.LinearSVC()

        clf.fit(selected_features_train, y_train)

        y_predict = clf.predict(selected_features_test)

        acc = accuracy_score(y_test, y_predict)
        if acc > aiacccb:
            aiacccb=acc
            aic.append(idxxc[:])
            aiaccc.append(acc)
        else:
            pass
        idxxc.pop()

    if c[-1]<aiacccb:
        pass
    else:
        break
    t=aic[-1]
    c.append(aiaccc[-1])
    idxxc.append(t[-1])
    a=np.where(idx==idxxc[-1])
    k.append(a[0][0])
    print(aiaccc[-1])
    #print(k)    #[0, 74, 3, 12, 101, 46]




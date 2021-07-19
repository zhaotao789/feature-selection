import pandas as pd
import scipy.io
import numpy as np
from tqdm import  tqdm
from sklearn.model_selection import  train_test_split
import matplotlib.pyplot as plt
from skfeature.function.similarity_based import fisher_score
import warnings
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")  # 忽略版本问题


#df = pd.read_csv('data/Lung.csv', header=None)
#df = df.as_matrix()
#n_samples, n_features = np.shape(df)
#X = df[:, 0:-1]
#y = df[:, -1]


from scipy import io
mat = io.loadmat('data/Isolet')   #COIL20    1000维
X = mat['X']  # data
y = mat['Y']  # label
#X=X.toarray()
y = y[:, 0]

t=0#数据是否需要转换
if t==1:
    for x in range(X.shape[1]):
        max = np.max(X[:,x])
        min = np.min(X[:,x])
        if max == min:
            X[:,x] = 0
        else:
            X[:,x] = (X[:,x] - min) / (max - min)

#y=np.where(y==True,1,0)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

selected_features_train = X_train
selected_features_test = X_test
from sklearn import svm

clf = svm.LinearSVC()
clf.fit(selected_features_train, y_train)

y_predict = clf.predict(selected_features_test)
acc = accuracy_score(y_test, y_predict)
print(acc)
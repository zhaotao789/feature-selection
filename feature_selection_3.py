
import numpy as np
import time
from scipy import io
from database import *
from sklearn.model_selection import  train_test_split
import asyncio
loop = asyncio.get_event_loop()
import ast
import warnings
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")  # 忽略版本问题
import pandas as pd

def predata():
    #
    df = pd.read_csv('data/Spect.csv', header=None)
    df = df.as_matrix()
    # numpy.mat(_x, dtype=float),
    #mat = io.loadmat('BASEHOCK')   #COIL20    1000维
    #X = mat['X']  # data

    #y = mat['Y']  # label
    #y = y[:, 0]

    X = df[:, :-1]
    #X = X.toarray()
    y = df[:, -1]
    #y=np.where(y==True,1,0)

    n_samples, n_features = np.shape(X)
    print(n_samples, n_features)

    if n_features > 2056:  # 稀疏矩阵
        idx = getIndex(X)
    else:
        r = []
        for x in range(n_features):
            x = X[:, x]
            t = np.vstack((x, y))
            r1 = np.corrcoef(t)[0, 1]
            if np.isnan(r1) :
                r1 = 0.0
            r1 = -r1
            r.append(r1)
        index = np.argsort(r)
        idx = index.tolist()



    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
    return X_train, X_test, y_train, y_test,idx

def getIndex(X):
    X=np.where(X==0,0,1)
    t = np.sum(X, axis=0)
    index = np.argsort(-t)
    return index


async def serch_root(X_train, X_test, y_train, y_test,idx=[],abacc=0.0,idxx=[],quert=1e-4):


    #idx  当前的初始节点
    #abacc   初始节点精度
    #idxx  搜索范围

    ai=[]

    aiacc=[]
    abacc=abacc
    for i in idxx:
        if i in idx:
            continue
        else:
            idx.append(i)
        selected_features_train = X_train[:,idx]

        selected_features_test = X_test[:,idx]


        from sklearn.svm import SVC
        #from sklearn. import c
        #clf = SVC(kernel='rbf', C=1000, gamma=0.0001)#采用高斯核函数
        clf = SVC()
        clf.fit(selected_features_train, y_train)

        y_predict = clf.predict(selected_features_test)

        acc = accuracy_score(y_test, y_predict)

        #删除相同元素
        l2 = []
        [l2.append(i) for i in idx if not i in l2]
        idx=l2

        if acc>(abacc+quert):
            #abacc=acc
            k=idx.copy()   #为什么要写这句，是因为copy了一个实体，而非指针
            ai.append(k)
            aiacc.append(float('%.4f' % acc))
        else:
            pass
        idx.pop()
    return ai,aiacc


def delete_top10():
    result = database().select('SELECT root_id,root_acc FROM root_search_1')
    result.sort(key=lambda k: k[1])

    de_id = result[0:(1+int(len(result) / 5))]

    for id, acc in de_id:
        x = id
        #database().insert1("INSERT INTO delete_search_1(seed_id) VALUES (%s)" % (x))   #"INSERT INTO delete_search(seed_id, root_acc) VALUES (%s, %s)" % (x, y)
        #插入删除表时，同时删除root表
        database().delete("DELETE FROM root_search_1 WHERE root_id=%s" %(x))


def delete_top10_1():
    result = database().select('SELECT root_id,root_acc FROM root_search_1')
    result.sort(key=lambda k: k[1], reverse=True)

    de_id = result[0:(1 + int(len(result) / 3))]

    for id, acc in de_id:
        x = id
        #database().insert1("INSERT INTO delete_search_1(seed_id) VALUES (%s)" % (x))   #"INSERT INTO delete_search(seed_id, root_acc) VALUES (%s, %s)" % (x, y)
        #插入删除表时，同时删除root表
        database().delete("DELETE FROM root_search_1 WHERE root_id=%s" %(x))


def serch_top10():
    result = database().select('SELECT root_id,root_acc FROM root_search_1')
    result.sort(key=lambda k: k[1], reverse=True)
    #对低维，取样本比例较多；对高维，取样本比例较少


    de_id = result[0:10]
    for id,acc in de_id:
        x=id
        y=acc
        database().insert1("INSERT INTO seed_search_1(seed_id, root_acc) VALUES (%s, %s)"%(x,y))   #"INSERT INTO seed_search(seed_id, root_acc) VALUES (%s, %s)"%(x,y)


def serch_top10_1(m):
    result=m
    result.sort(key=lambda k: k[-1], reverse=True)
    if len(result)>=10:
        de_id = result[0:5]
    else:
        de_id = result[0:(1 + int(len(result) / 2))]
    return de_id

def serch_locate1(X_train, X_test, y_train, y_test,login=0):
    sql="SELECT seed_id,root_acc FROM seed_search_1"
    result=database().select(sql)
    #十个种子
    for seed,seed_acc in result:
        #将seed包装成list
        if type(seed)==list:
            pass
        else:
            seed=[seed]
        #获取搜索空间
        location=database().select("SELECT root_id FROM root_search_1")
        idxj=[]
        for lo in location:
            idxj.append(lo[0])
        scoreid,scoreacc=loop.run_until_complete(serch_root(X_train, X_test, y_train, y_test,idx=seed,abacc=seed_acc,idxx=idxj,quert=1e-2))
        list_append = []
        if scoreacc==[]:
            login+=1
            continue
        else:
            k=-1
            for t in scoreid:
                k+=1
                x=t[-1]
                y=scoreacc[k]
                m=(x,y)
                list_append.append(m)
            pinjie(list_append=list_append,scoreacc=scoreacc,seed=seed)
            #返回结果,拼接成list   如[(5,0.18),(17,0.89)]    现如今是[[1, 2],[1,3]]   [0.57,0.54]
            # 对选中的下一步，将他们分别和seed_id和delete_id组合，看看是否存在奇迹
    return login,len(result)

def pinjie(list_append=[],scoreacc=[],seed=[]):

        #对返回的结果排序，取前1/2
        list_append=serch_top10_1(list_append)
        t="{0}".format(list_append)
        database().UPDATE(t,float('%.5f' % max(scoreacc)),seed[0])

from functools import reduce
def serch_locate2(X_train, X_test, y_train, y_test,login=0):
    sql="SELECT seed_id,root_acc FROM seed_search_1"
    result=database().select(sql)
    #删减root_search里的前1/10

    #十个种子
    for seed,seed_acc in result:
        #将seed包装成list
        if type(seed)==list:
            pass
        else:
            seed=[seed]
        #获取搜索空间
        location=database().select("SELECT locate_id FROM seed_search_1 WHERE seed_id=%s"%(seed[0]))
        #将字符串转化为list
        if location[0][0]==None:
            login+=1
            continue
        data_json = ast.literal_eval(location[0][0])
        #得到list，对list里每个元素搜索
        sco_vaild_id=[]
        sco_vaild_acc=[]
        for q in data_json:
            k=q[0:-1]

            k=[i for i in k]

            seed=seed.copy()+k
            seed_acc=q[-1]
            sql = "SELECT root_id FROM root_search_1"
            res = database().select(sql)
            idxj = []
            for lo in res:
                idxj.append(lo[0])
            scoreid,scoreacc=loop.run_until_complete(serch_root(X_train, X_test, y_train, y_test,idx=seed,abacc=seed_acc,idxx=idxj,quert=1e-2))
            if scoreacc!=[]:
                #更新数据库中该字段的值
                sco_vaild_id.append(scoreid)
                sco_vaild_acc.append(scoreacc)
            for i in range(len(k)):
                seed.pop()
        #返回最大值
        #sco_vaild_id,sco_vaild_acc排序，取前1/10
        if sco_vaild_acc!=[]:
            b = reduce(lambda x, y: x + y, sco_vaild_acc)
            a=reduce(lambda x, y: x + y, sco_vaild_id)
            #将数据转换成[()]
            for i in range(len(b)):
                x=a[i]+[float('%.5f' % b[i])]
                a[i]=tuple(x)
            #排序，取前1/2
            a.sort(key=lambda k: k[-1], reverse=True)
            #a=dock(a)
            de_id = a[0:int(len(result))]
            #data_json和de_id比较
            list_append=de_id
            pinjie(list_append=list_append,scoreacc=[seed_acc],seed=seed)
            #返回结果,拼接成list   如[(5,0.18),(17,0.89)]    现如今是[[1, 2],[1,3]]   [0.57,0.54]
            # 对选中的下一步，将他们分别和seed_id和delete_id组合，看看是否存在奇迹
        else:
            login+=1


    return login,len(result)

def last_():
    # 获取搜索空间
    location = database().select("SELECT locate_id FROM seed_search_1 ")
    # 将字符串转化为list
    b = reduce(lambda x, y: x + y, location)
    j=[]
    for i in b:
        if i==None:
            continue
        elif j==[]:
            j=ast.literal_eval(i)
        else:
            k=ast.literal_eval(i)
            j=j+k


    j.sort(key=lambda k: k[-1], reverse=True)
    serch_id__=dock(de_id=j)  #去重

    serch_id_=serch_id__[0:10]
    r=-1
    if serch_id_==[] or serch_id_==None:
        seed_id=database().select("SELECT seed_id,root_acc FROM seed_search_1")
        for sid in seed_id:
            r += 1
            database().insert2(r, sid[0], sid[1])
    else:
        for k in serch_id_:
            r+=1
            serch_id=k[0:-1]
            #TODO   serch_id存在问题
            serch_score=str((k[-1]))
            database().insert2(r,serch_id,serch_score)

#去重
def dock(de_id=[]):
    #de_id=[(20,27,5,0.86),()]
    serch_id_ = []
    de_id_ = []
    # 删除相同的结果
    for c in de_id:
        serch_id = c[0:-1]
        serch_id = list(serch_id)
        serch_id.sort()
        t=[c[-1]]
        serch_id = serch_id+t
        # serch_id=tuple(serch_id)
        de_id_.append(serch_id)
    for i in de_id_:
        if i not in serch_id_:
            serch_id_.append(i)
    serch_id__ = []
    for f in serch_id_:
        serch_id = tuple(f)
        serch_id__.append(serch_id)
    return serch_id__


def serch_locate3(X_train, X_test, y_train, y_test, login=0):
    sql = "SELECT id,search_id,root_acc FROM search_search_1"
    result = database().select(sql)
    # 删减root_search里的前1/10

    # 十个种子
    for r,seed_, seed_acc in  result:
        # 将seed包装成list
        seed_1=seed_
        seed = seed_1.strip('()').split(',')
        seed_new=[]
        for i in seed:
            if i=='':
                continue
            i=int(i)
            seed_new.append(i)
        seed=[i for i in seed]
        # 获取搜索空间
        location = database().select("SELECT root_id FROM root_search_1 ")
        # 将字符串转化为list
        location_new=[]
        for l in location:
            location_new.append(l[0])

        # 得到list，对list里每个元素搜索
        seed_acc=float(seed_acc)
        scoreid, scoreacc = loop.run_until_complete(serch_root(X_train, X_test, y_train, y_test, idx=seed_new, abacc=seed_acc, idxx=location_new,quert=1e-3))
        if scoreacc != []:

            # 将数据转换成[()]
            for i in range(len(scoreacc)):
                x = scoreid[i] + [float('%.5f' % scoreacc[i])]
                scoreid[i] = tuple(x)
            # 排序，取前1/2
            scoreid.sort(key=lambda k: k[-1], reverse=True)
            de_id = scoreid[0]
            # data_json和de_id比较
            #de_id=dock(de_id)
            a=str(de_id[0:-1])
            b=str(de_id[-1])
            database().update2(r, a, b)
            # 返回结果,拼接成list   如[(5,0.18),(17,0.89)]    现如今是[[1, 2],[1,3]]   [0.57,0.54]
            # 对选中的下一步，将他们分别和seed_id和delete_id组合，看看是否存在奇迹
        else:
            login += 1

    return login, len(result)

def serch_locate4(X_train, X_test, y_train, y_test,idx=[]):
    #对serch_serch_1排序选择前2/5
    data=database().select('SELECT search_id FROM search_search_1 order by root_acc desc ')
    data=data[0:int(len(data)*3/5)]
    acc = database().select('SELECT root_acc FROM search_search_1 order by root_acc desc ')
    acc = acc[0:int(len(acc) * 3 / 5)]
    ai, aiacc = loop.run_until_complete(serch_root(X_train, X_test, y_train, y_test, idxx=idx))
    ai = reduce(lambda x, y: x + y, ai)
    idxx=np.argsort(aiacc)[::-1]
    idxx=idxx[0:15]
    database().drop_table('search_search_1')
    database().create_search_table()
    step=-1
    for i in range(len(acc)):
        step+=1
        database().insert2(step,data[i][0],float(acc[i][0]))
    for i in range(len(idxx)):
        step+=1
        database().insert2(step,idxx[i],aiacc[idxx[i]])

def select_bert3(data):
    best_ai=[]
    best_acc=[]
    ai, aiacc = loop.run_until_complete(serch_root(X_train, X_test, y_train, y_test, idxx=data, quert=1e-3))
    idxx = np.argsort(aiacc)[::-1]
    best_ai = ai[idxx[0]]
    best_acc = aiacc[idxx[0]]
    print(best_acc)
    while (len(aiacc)>0):
        ai, aiacc = loop.run_until_complete(serch_root(X_train, X_test, y_train, y_test, idx=best_ai,idxx=data, quert=1e-4))

        idxx = np.argsort(aiacc)[::-1]
        if aiacc[idxx[0]]<=best_acc:
            break
        best_ai=ai[idxx[0]]
        best_acc=aiacc[idxx[0]]
        print(best_acc)
        print(best_ai)
    return best_ai,best_acc

def serch_locate5(index,data):
    # 对serch_serch_1排序选择前2/5
    data1 = database().select('SELECT search_id FROM search_search_1 order by root_acc desc ')

    #database().drop_table('search_search_1')
    #database().create_search_table()
    #并构造一个超强子集，因为每个区域会带来干扰。根据数据库子集构造
    #select_bert1(data1)
    #根据前b3子集构造
    #select_bert2(index,data)
    #根据all子集构造,并输出
    best_ai,best_acc=select_bert3(data)
    print(best_ai)
    print(best_acc)
    database().insert1("INSERT INTO delete_search_1(id,seed_id,root_acc) VALUES (%s,%s,%s)" % (0,str(best_ai),str(best_acc)))
    step = -1
    for i in range(len(data1)):
        step += 1
        database().insert2(step, data[i][0], float(data1[i][0]))    #数据库只会有2条数据，分别代表三条路线。最后一条最为先验值输出

if __name__ == '__main__':

    X_train, X_test, y_train, y_test, idx = predata()

    n_samples, n_features = np.shape(X_train)
    login = 0  # 判断已经完成了多少
    start = time.time()

    ai, aiacc = loop.run_until_complete(serch_root(X_train, X_test, y_train, y_test, idxx=idx,quert=1e-2))
    ai = reduce(lambda x, y: x + y, ai)
    database().insert(ai, aiacc)
    # 选取1/10作为初始点

    serch_top10()
    # 选取后1/10作为待删除点
    #delete_top10()

    # 以前1/10为起点，遍历root_search,第一次存在捡垃圾
    serch_locate1(X_train, X_test, y_train, y_test, login=login)
    # 第二次，坚决不捡垃圾。其实捡垃圾，精度会有提升，但那种方法又变成穷举法了
    j5, k = serch_locate2(X_train, X_test, y_train, y_test, login=login)
    while (j5 < k + 1):        #测试时关闭，穷举作业。低维时建议穷举，但高维时不建议。因为如果打开，会局部最优，但如果关闭，精度会相对低一些
        j5, k = serch_locate2(X_train, X_test, y_train, y_test, login=j5)
    last_()
    database().drop_table('root_search_1')
    database().create_root_table()
    database().insert(ai, aiacc)
    j5, k = serch_locate3(X_train, X_test, y_train, y_test, login=login)
    while (j5 < k + 1):  # 测试时关闭，穷举作业。低维时建议穷举，但高维时不建议。因为如果打开，会局部最优，但如果关闭，精度会相对低一些
        j5, k = serch_locate3(X_train, X_test, y_train, y_test, login=j5)
    acc = database().select('SELECT root_acc FROM search_search_1 order by root_acc desc ')
    data = database().select('SELECT search_id FROM search_search_1 order by root_acc desc ')
    print(data[0][0])
    print(acc[0][0])
    end = time.time()
    print(end-start)






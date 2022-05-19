import numpy as np
import random
import matplotlib.pyplot as plt
import time
import sklearn.cluster as skc
from sklearn import metrics
from sklearn.datasets import load_iris


def findNeighbor(j,X,eps):
    N=[]
    for p in range(X.shape[0]):   #找到所有领域内对象
        temp=np.sqrt(np.sum(np.square(X[j]-X[p])))   #欧氏距离
        if(temp<=eps):
            N.append(p)
    return N


def dbscan(X,eps,min_Pts):
    k=-1
    NeighborPts=[]      #array,某点领域内的对象
    Ner_NeighborPts=[]
    fil=[]                                      #初始时已访问对象列表为空
    gama=[x for x in range(len(X))]            #初始时将所有点标记为未访问
    cluster=[-1 for y in range(len(X))]
    while len(gama)>0:
        j=random.choice(gama)  #随机选择一个点
        gama.remove(j)  #未访问列表中移除
        fil.append(j)   #添加入访问列表
        NeighborPts=findNeighbor(j,X,eps)  #欧式距离计算，并统计邻域内的点
        if len(NeighborPts) < min_Pts:
            cluster[j]=-1   #标记为噪声点
        else:
            k=k+1
            cluster[j]=k
            for i in NeighborPts:#查询邻域内的边界点
                if i not in fil:
                    gama.remove(i)
                    fil.append(i)
                    Ner_NeighborPts=findNeighbor(i,X,eps)##计算该邻域内边界点的邻域
                    if len(Ner_NeighborPts) >= min_Pts:
                        for a in Ner_NeighborPts:
                            if a not in NeighborPts:
                                NeighborPts.append(a)
                    if (cluster[i]==-1):
                        cluster[i]=k
    return cluster


#导入数据集
X=load_iris().data
plt.scatter(X[:, 2], X[:, 3],c=load_iris().target)
plt.show()
# DBSCAN聚类
db= dbscan(X[:,2:3],eps=0.9, min_Pts=10)
label_pred = db

plt.scatter(X[:, 2], X[:, 3],c=label_pred)
plt.show()

db2= dbscan(X[:,2:3],eps=0.2,min_Pts=4)
label_pred2 = db2

plt.scatter(X[:, 2], X[:, 3],c=label_pred2)
plt.show()

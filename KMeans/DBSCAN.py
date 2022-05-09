import numpy as np
import sklearn.cluster as skc
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
#导入数据集
X=load_iris().data
# plt.scatter(X[:, 2], X[:, 3],c=load_iris().target)
# plt.show()
# DBSCAN聚类
db= skc.DBSCAN(eps=0.9, min_samples=10).fit(X[:,2:3])
label_pred = db.labels_

plt.scatter(X[:, 2], X[:, 3],c=label_pred)
plt.show()


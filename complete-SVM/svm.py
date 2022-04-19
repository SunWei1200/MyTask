import numpy as np
import copy
import matplotlib.pyplot as plt
#读入本地的数据，读出来的数据格式都是列表形式的
def load_data(filename):
    dataset=[]
    labelset=[]
    f=open(filename)
    for line in f.readlines():
        new_line=line.strip().split()
        dataset.append([float(new_line[0]),float(new_line[1])])
        labelset.append(int(new_line[2]))
    return dataset,labelset

def select_j(i,m):
    """parameter:i 表示第一个选定的alpha,对应第i行数据
                 m  表示数据集的总数
    """
    j=-1
    while(j==i):
        j=int(np.random.uniform(0,m))
    return j
#用来限定alpha值的上下限
def regular(H,L,a):
    if a>H:
        a=H
    if a<L:
        a=L
    return a

def SMO(data,label,C,max_iters):
    data=np.mat(data)
    label=np.mat(label).transpose()
    m,n=np.shape(data)
    alphas=np.mat(np.zeros((m,1)))
    b=0
    iter=0
    value = 0.01  # 精度值，KKT条件
    while(iter<max_iters):
        alpha_pairs_changed=0
        for i in range(m):
            Fxi=float(np.multiply(alphas,label).T*(data*data[i].T))+b
            Ei=Fxi-float(label[i])
            if ((label[i]*Ei<-value) and (alphas[i]<C)) or ((label[i]*Ei>value) and (alphas[i]>0)):
                ## 违反kkt条件，不断调整alpha,通过计算判别错误来选择错误的alpha，另外一个alpha可以简单的随机选择
                j=select_j(i,m)
                Fxj=float(np.multiply(alphas,label).T*(data*data[j].T))+b
                Ej=Fxj-float(label[j])
                alphaIold=copy.deepcopy(alphas[i])
                alphaJold=copy.deepcopy(alphas[j])
                if label[i]!=label[j]:
                    L=max(0,alphas[j]-alphas[i])
                    H=min(C,C+alphas[j]-alphas[i])
                else:
                    L=max(0,alphas[j]+alphas[i]-C)
                    H=min(C,alphas[j]+alphas[i])
                if H==L:continue
                eta=data[i]*data[i].T+data[j]*data[j].T-2*data[i]*data[j].T
                #迭代更新中需要n，相当于是最小化函数对alphaj求最小值时的二阶导数，凸函数二阶导数小于0
                if eta<=0:
                    continue
                alphas[j]+=label[j]*(Ei-Ej)/eta#往下均是迭代更新的公式
                alphas[j]=regular(H,L,alphas[j])
                if abs(alphas[j]-alphaJold)<0.00001:
                    continue
                alphas[i]+=label[i]*label[j]*(alphaJold-alphas[j])#alphas更新
                b1=b-Ei-label[i]*(alphas[i]-alphaIold)*data[i]*data[i].T-label[j]*\
                    (alphas[j]-alphaJold)*data[i]*data[j].T
                b2=b-Ej-label[i]*(alphas[i]-alphaIold)*data[i]*data[j].T-label[j]*\
                    (alphas[j]-alphaJold)*data[j]*data[j].T
                if 0<alphas[i]<C:
                    b=b1
                elif 0<alphas[j]<C:
                    b=b2
                else:
                    b=(b1+b2)/2
                alpha_pairs_changed+=1
        if alpha_pairs_changed==0:#这里的alpha_pairs_changedz主要是用来防止alpha在若干次循环中都没有更新的情况下还在继续无休止的循环下去
            iter+=1
        else:
            iter=0
    w=np.mat(np.zeros((n,1)))#往后直接根据公式求出w
    for i in range(m):
        w+=np.multiply(alphas[i]*label[i],data[i].T)
    return alphas,b,w

def plot():
    dataset,labelset=load_data('testSet.txt')
    alphas,b,w=SMO(dataset,labelset,0.01,200)
    w=np.array(w)
    b=np.array(b)
    xcord1=[]
    ycord1=[]
    xcord2=[]
    ycord2=[]
    for i in range(len(dataset)):
        if labelset[i]==1:
            xcord1.append(dataset[i][0])
            ycord1.append(dataset[i][1])
        elif labelset[i]==-1:
            xcord2.append(dataset[i][0])
            ycord2.append(dataset[i][1])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,c='blue',s=50)
    ax.scatter(xcord2,ycord2,c='green',s=70)
    x=list(np.arange(1,10,0.01))
    y=[(-w[0]*i-b[0])/w[1] for i in x]
    ax.plot(x,y)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()
plot()
                
                

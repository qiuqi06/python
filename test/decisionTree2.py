###################过拟合
######### min_samples_split min_samples_leaf min_weight_fraction_leaf
######### max_depth max_leaf_node min_features
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
iris=datasets.load_iris()
x=iris.data[:,2:]
y=iris.target
from sklearn.tree import DecisionTreeClassifier
dt_cfl=DecisionTreeClassifier(max_depth=2,criterion='entropy')
dt_cfl.fit(x,y)
def plot_decision_bouondary(model,axis):
    x0,x1=np.meshgrid(
        np.linspace(axis[0],axis[1],int((axis[1]-axis[0])*100)).reshape(),
        np.linspace(axis[2],axis[3],int((axis[3]-axis[2])*100)).reshape()
    )
    x_new=np.c_(x0.ravel(),x1.ravel())
    y_predict=model.predict(x_new)
    zz=y_predict.reshape(x0.shape)
    # custom_camp=ListedColormap('#EF9A9A','#FFD50D','#90CAF9')
    # plt.contourf(x0,x1,zz,linewith=5,camp=custom_camp)
plot_decision_bouondary(dt_cfl,axis=[0.5,7,5,0,3])
# plt.scatter(x[y==0,0],x[y==0,1])
# plt.scatter(x[y==1,0],x[y==1,1])
# plt.scatter(x[y==2,0],x[y==2,1])
# plt.show();

from collections import Counter
from math import log
def entropy(y):
    counter=Counter(y)
    res=0.0
    for num in counter.values():
        p=num/len(y)
        res+=-p*log(p)
    return res

def split(x,y,d,value):
    index_a=(x[:,d]<=value)
    index_b=(x[:,d]>value)
    return x[index_a],x[index_b],y[index_a],y[index_b]
def try_split(x,y):
    best_entropy=float('inf')
    best_d,best_v=-1,-1
    for d in range(x.shape[1]):
        sorted_index=np.argsort(x[:,d])
        for i in range(1,len(x)):
            if x[sorted_index[i-1]]!=x[sorted_index[i],d]:
                v=(x[sorted_index[i-1],d])+x[sorted_index[i],d]/2
                x_1,x_r,y_1,y_r=split(x,y,d,v)
                e=entropy(y_1)+entropy(y_r)
                if e<best_entropy:
                    best_entropy,best_d,best_v=e,d,v
    return best_entropy,best_d,best_v
try_split(x,y)





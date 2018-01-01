from roadInfo import *
from PIL import Image
import numpy as np
import pdb, random
import matplotlib.pyplot as plt

def mapping(darray,karray):
    num = len(darray)
    try:
        (m, n) = np.array(karray).shape
    except:
        (m, n) = len(karray),1
    tem1 = (np.abs(darray)-np.abs(np.tile(karray,[num,1])))**2
    tem2 = sum(tem1.reshape([m,num])).reshape([num,1])
    ret = np.exp(-1*tem2)
    return ret

def elm(X1,Y1,X2,c_num,lang=0.001,alpha=1):
    num = len(X1)
    list1 = [i for i in range(num)]
    random.shuffle(list1)
    Cgroup = []
    for i in range(c_num):
        Cgroup.append(X1[list1[i]])
    H = np.zeros((num,c_num))
    for j in range(c_num):
        H[:,j] = mapping(X1,Cgroup[j])[:,0]
    beta = np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(H),H)),np.transpose(H)),np.array(Y1))
    H2 = np.zeros((num,c_num))
    for j in range(c_num):
        H2[:,j] = mapping(X2,Cgroup[j])[:,0]
    Ypre = np.dot(H2,beta)
    return Ypre

X1 = np.random.random([100,1])*20-10
Y1 = np.sin(X1)/X1*np.cos(X1)
X2 = np.random.random([100,1])*20-10
Y2 = elm(X1,Y1,X2,50)

plt.plot(X1,Y1,'b*')
plt.plot(X2,Y2,'ro')
plt.show()
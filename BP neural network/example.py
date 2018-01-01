# a simple example for curve fitting
from BPnn import *
import numpy as np
import matplotlib.pyplot as plt
import random, pdb

random.seed(0)
TrainX = []
TrainY = []
for i in range(0,10):
    tempx = []
    tempx.append(np.round(random.random()*20-10,2))
    TrainX.append(tempx)
    tempy = []
    tempy.append(np.sin(tempx[0])/tempx[0] + np.random.randn()*0.05)
    TrainY.append(tempy)
TestX = []
TestY = []
for i in range(0,10):
    tempx = []
    tempx.append(np.round(random.random()*20-10,2))
    TestX.append(tempx)
    tempy = []
    tempy.append(np.sin(tempx[0])/tempx[0])
    TestY.append(tempy)

TestX = TrainX
TestY = TrainY
TestY = bpnn(TrainX,TrainY,TestX)

ax1=plt.subplot(211)
plt.plot(TrainX,TrainY,'b*')
plt.axis([-10,10,-0.25,1.25]) 
ax2=plt.subplot(212)
plt.plot(TestX,TestY,'ro')
plt.axis([-10,10,-0.25,1.25]) 
plt.show()
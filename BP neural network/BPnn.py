import numpy as np 
import pdb
import random
import matplotlib.pyplot as plt
def rand(a,b):
    return (b-a)*random.random()+a

def dataStandize(indata): # define method to standize training data
    temp = list(zip(*indata))  
    r,c = len(temp), len(temp[0])
    mValue = [np.mean(i) for i in temp]
    sValue = [np.std(i) for i in temp] 
    ret = [[(temp[i][j]-mValue[i])/sValue[i] for i in range(r)] for j in range(c)]
    return ret, mValue, sValue
def dataStandize2(indata,mValue, sValue): # define method to standize test data with the parameters as [dataStandize]
    temp = list(zip(*indata))    
    r,c = len(temp), len(temp[0])
    ret = [[(temp[i][j]-mValue[i])/sValue[i] for i in range(r)] for j in range(c)]
    return ret
def dataOppStandize(indata,mValue, sValue):  # define method to resume standized data 
    temp = list(zip(*indata)) 
    r,c = len(temp), len(temp[0])
    ret = [[temp[i][j]*sValue[i]+mValue[i] for i in range(r)] for j in range(c)]
    return ret

# define some frequent-used activation functions and its inverse functions
def sigmoid(x,mode = 0): 
    if mode == 0:
        return 1/(1+np.exp(-x))
    elif mode == 1:
        return x * (1-x) # the input of inverse function is the output of origin function 
    
def tanh(x,mode = 0):
    if mode == 0:
        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    elif mode == 1:
        return 1-x**2
def relu(x,mode = 0):
    if mode == 0:
        if x >= 0:
            return x 
        else:
            return 0
    elif mode == 1:
        if x >= 0:
            return 1
        else:
            return 0

# define a new class of neural unit
class Unit:
    def __init__(self, num, actfun):
        self.weight = [rand(-1,1) for i in range(num)]
        self.bias = rand(-1,1)
        self.actfun = actfun
    def calc(self, indata):
        self.indata = indata
        unitsum = sum([i * j for i, j in zip(self.weight, indata)]) + self.bias
        self.out = self.actfun(unitsum)
        return self.out
    def update(self, alpha, change):    
        #pdb.set_trace()
        self.change = change
        self.weight = [i + alpha * change * j for i, j in zip(self.weight, self.indata)] 
        self.bias = self.bias + alpha * change      
    def getinfo(self):        
        ret =  self.change, ['w: %.4f,in: %.4f\n'%(i,j) for i,j in zip(self.weight, self.indata)]
        return ret
# define a new class of neural layer
class Layer:
    def __init__(self, num_input, num_output, actfun):
        self.units = [Unit(num_input, actfun) for i in range(num_output)]
        self.output = num_output
        self.innum = num_input
    def calc(self, indata):
        self.indata = indata  
        self.out = [unit.calc(self.indata) for unit in self.units]        
        #pdb.set_trace()
        return self.out
    def update(self, alpha, changes):
        for unit, change in zip(self.units, changes):
            unit.update(alpha, change)
    def error(self, deltas):
        def _error(deltas, j):
            return sum([delta * unit.weight[j] for delta, unit in zip(deltas, self.units)])
        return [_error(deltas, j) for j  in range(self.innum)]
    def getinfo(self):
        ret =  [unit.getinfo() for unit in self.units]        
        #pdb.set_trace()
        return ret
# define a new class of neural network
class Nnet:
    def __init__(self, ni, nh, no, actfun):
        self.ni = ni
        self.nh = nh
        self.no = no
        self.actfun = actfun
        self.hlayer = [0]*len(self.nh)
        self.hlayer[0] = Layer(ni, nh[0], actfun)
        if len(self.nh)>1:
            self.hlayer[1:] = [Layer(nh[i-1], nh[i], actfun) for i in range(1, len(self.nh))]
        self.olayer = Layer(nh[-1], no, actfun)
    def calc(self, indata):
        temp = indata
        self.ah = []
        for i in range(len(self.nh)):
            self.ah.append(self.hlayer[i].calc(temp))
            temp = self.ah[-1]
        self.ao = self.olayer.calc(self.ah[-1]) 
        return self.ao
    def getmse(self, output):
        self.dev = [(i - j) for i,j in zip(output, self.ao)]
        self.mse = sum([i**2 for i in self.dev])
        return self.mse
    def update(self, alpha):
        out_deltas = [cost * self.actfun(ao,1) for cost, ao in zip(self.dev, self.ao)]
        hid_deltas = [0]*len(self.nh)
        hid_deltas[-1] = [cost * self.actfun(ah,1) for cost, ah in zip(self.olayer.error(out_deltas), self.ah[-1])]
        for i in range(len(self.nh)-2,-1,-1):
            hid_deltas[i] = [cost * self.actfun(ah,1) for cost, ah in zip(self.hlayer[i+1].error(hid_deltas[i+1]), self.ah[i])]
        self.olayer.update(alpha, out_deltas)
        for i in range(len(self.nh)-1,-1,-1):
            self.hlayer[i].update(alpha, hid_deltas[i])
    def getinfo(self):
        alllayer = [layer for layer in self.hlayer] + [self.olayer]
        return '%.4f'%self.dev[0], [layer.getinfo() for layer in alllayer]

    def train(self, trainX, trainY, iterations = 1000, alpha=1):
        if len(trainX[0]) != self.ni or len(trainY[0]) != self.no or len(trainX) != len(trainY):
            raise ValueError('wrong scale of training data')
        fout = open('traininfo.txt','w+')
        for i in range(iterations):
            mse = 0
            for j in range(len(trainX)):    
                self.calc(trainX[j])
                mse = mse + self.getmse(trainY[j])
                alpha = 0.1    
                self.update(1)
                info = self.getinfo()
                fout.write('\n****** iteration = %1d ******\n'%j)
                fout.write(str(info))
                #pdb.set_trace()
            if i % 100 == 0:
                print('training mse is %2f', mse/len(trainX))
    def test(self, testX):        
        ret = [self.calc(i) for i in testX]
        return ret

def bpnn(TrainX,TrainY,TestX):
    Xr_num,Xc_num = len(TrainX), len(TrainX[0]) # number of training inputs and the dimension of each input
    Yr_num,Yc_num = len(TrainY), len(TrainY[0]) # number of training outputs and the dimension of each output
    Xr_num2,Xc_num2 = len(TestX), len(TestX[0]) # number of test inputs and the dimension of each input
    #TrainX, xmValue, xsValue = dataStandize(TrainX)
    #TrainY, ymValue, ysValue = dataStandize(TrainY)
    #TestX = dataStandize2(TestX,xmValue, xsValue)
    funGroup = {'sigmoid': sigmoid, 'tanh': tanh, 'relu': relu}
    actfun = funGroup['sigmoid']
    n = Nnet(1, [10], 1, actfun)
    n.train(TrainX, TrainY)
    ret = n.test(TestX)
    return ret #dataOppStandize(ret,ymValue, ysValue)  





# convoluntional neural network
import numpy as np
# convoluntional function
# calculate dot product
def dp(A,B):    
    [h,w] = A.shape # A and B has the same scale
    C = np.zeros([h,w])    
    for i in range(h):
        for j in range(w):
            C[i,j] = A[i,j] * B[i,j]
    return C
def conv2(A,B,mode):
    [ha,wa] = A.shape
    [hb,wb] = B.shape
    if mode == 'valid':
        C = np.zeros([ha-hb+1,wa-wb+1])
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                C[i,j] = sum(map(sum,dp(A[i:i+hb,j:j+wb],B)))
    elif mode == 'full':
        An = np.zeros([ha+2*(hb-1),wa+2*(wb-1)])
        An[hb-1:ha+hb-1,wb-1:wa+wb-1] = A
        C = np.zeros([ha+hb-1,wa+wb-1])   
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                C[i,j] = sum(map(sum,dp(An[i:i+hb,j:j+wb],B)))
    return C   

# input layer
# convoluntional layer 1: c1 
# subsample layer 1: s2
# convoluntional layer 2: c3
# subsample layer 2: s4
# output layer
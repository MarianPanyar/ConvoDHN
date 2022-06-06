# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#NNN3
#layerwise training with labels as weights


import numpy as np
#import matplotlib.pyplot as plt
import importlib
#import scipy
#from scipy import ndimage


image_size = 28 # width and length
nlb = 10 #  i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size
data_path = "data/mnist/"
train_data = np.loadtxt(data_path + "mnist_train.csv", 
                        delimiter=",")
test_data = np.loadtxt(data_path + "mnist_test.csv", 
                       delimiter=",") 
#test_data[:10]

label = train_data[:,0]
raw_tdata = train_data[:,1:]
ts_data = test_data[:,1:]
ts_label = test_data[:,0] 

#%%

import models
from models import model

importlib.reload(models)
#importlib.reload(verf)

#%%
datalen = 10000
RN = 0.000005
#CFtres = 0.001
alpha = 0.1
beta = 0.2
#CF = 0
SPlim = 2000
gamma = 4
delta = 2
ep = 0.2
nnodes = 30
#capa = 4
stepL = 500
#MPsize = [28,28]
MPdia = 4
MPstride = 2

inMD = 'sample'
reSL = 1.4

TP = 0.3


T_label = label[:datalen]




#%%

#NN(nnodes(required),RN,alpha,beta,stepL,inMD,reSL,TP)
#NN_MP(nnodes,MPsize,MPdia,MPstride(required),RN,alpha,beta,stepL,inMD,reSL,TP,SPlim)

flow_size = [datalen,[28,28]]

A = model(flow_size,alpha,beta,gamma,delta,ep,RN,stepL,inMD,reSL,TP,SPlim)

A.add('Bipolar',2,0.5,'abs')
#A.add('Bipolar',2,0.5,'abs')
#A.add('Bipolar',2,0.5,'abs')
A.add('Padding',1,'constant')


A.add('Conv2D',3,1,1)
#A.add('Conv2D',3,1,2)
A.add('NNmp',32,2,2,stepL=300,alpha=0.1,beta=0.1,ep=0.8,inMD='sample',SPlim=300)
#A.add('Conv2D',3,1,2)
#A.add('Flat',1,2)
#A.add('oneHot',0)
A.add('Conv2D',3,1,2)
A.add('Flat',1,2)
A.add('NNmp',64,2,2,stepL=300,alpha=0.1,beta=0.1,ep=0.8,inMD='sample',SPlim=500)
#A.add('oneHot',0)
A.add('Flat',1,2)
A.add('NN',128,alpha=0.1,beta=0.05,ep=0.7,stepL=300,inMD='sample')
#A.add('GG',nnodes,RN,stepL,inMD,reSL)

#%%
#NN(nnodes(required),RN,alpha,beta,stepL,inMD,reSL,TP)
#NN_MP(nnodes,MPsize,MPdia,MPstride(required),RN,alpha,beta,stepL,inMD,reSL,TP,SPlim)

#flow_size = [datalen,[28,28]]
#A = model(flow_size,alpha,beta,gamma,delta,ep,RN,stepL,inMD,reSL,TP,SPlim)
#A.add('NN',200,alpha=0.1,beta=0.1,stepL=600,inMD='sample')




#%%
T_data = raw_tdata[:datalen]
outdata = A.init_train(T_data)
#outdata = A.sv_train(bsdata,t_label)

#frrt = np.dot(raw_tdata[20000:40000],A.frame[0].bsmx.T)
#bmu = np.argmax(frrt,axis=-1)
#np.unique(bmu,return_counts=True)
#%%
#outdata = A.us_train(bsdata)
U_data = raw_tdata[20000:30000]
Tlist = [1,1,1,1,1,1,1,1,1,3]
outdata = A.us_train(U_data,Tlist)


#%%
#LLsv_train(indata,label,nlb(required),stepL,delta)
A.frame[-3].alpha = 0
A.frame[-3].beta = 0
A.frame[-3].gamma = 0
A.frame[-3].delta = 2
A.frame[-3].ep = 0.8
A.frame[-3].TP = 0.05

A.frame[-1].alpha = 0.2
A.frame[-1].beta = 0.2
A.frame[-1].gamma = 0
A.frame[-1].delta = 10
A.frame[-1].ep = 0.9
#%%
bsdata = raw_tdata[10000:30000]
t_label = label[10000:30000]
outdata = A.LLsv_train(bsdata,t_label,nlb,400)

#%%
import matplotlib.pyplot as plt 
bs4 = A.frame[-3].bsmx
for i in range(10):
    plt.matshow(bs4[i].reshape(5,5))



#%%
A.LLtestTrain(raw_tdata[30000:40000],label[30000:40000],ts_data,ts_label,nlb)
#t_label = label[datalen:datalen*2]|

#p_label,Elist = A.ap_cacu(bsdata,t_label)

#%%
bsdata = raw_tdata[datalen:datalen*2]
t_label = label[datalen:datalen*2]

A.mpap_train(bsdata,t_label)

#%%
outdata = A.forward(bsdata,t_label)

#%%
import matplotlib.pyplot as plt 

for i in range(20):
    plt.matshow(np.reshape(A.frame[0].bsmx[i],(28,28)))


#%%

    


#%%




#%%



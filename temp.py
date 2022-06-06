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
datalen = 12000
RN = 0.000005
#CFtres = 0.001
alpha = 0.1
beta = 0.2
#CF = 0
SPlim = 20000
delta = 0.2
nnodes = 30
#capa = 4
stepL = 500
#MPsize = [28,28]
MPdia = 4
MPstride = 2

inMD = 'TR'
lp = 3
reSL = 1.3

TP = 0.3

T_data = raw_tdata[:datalen]
T_label = label[:datalen]


#bsdata = raw_tdata[:datalen]

#from layers import flat,relu,normalize,convolution,padding,GG_MP,bipolar,pl_index


#%%

#NN(nnodes(required),RN,alpha,beta,stepL,inMD,reSL,TP)
#NN_MP(nnodes,MPsize,MPdia,MPstride(required),RN,alpha,beta,stepL,inMD,reSL,TP,SPlim)

flow_size = [datalen,[28,28]]

A = model(flow_size,alpha,beta,delta,RN,stepL,inMD,reSL,TP,SPlim)

#A.add('Bipolar',2,0.5,n_mode='abs')

A.add('Padding',1,'constant')
#A.add('Bipolar',2,0.5,'abs')

A.add('Conv2D',7,2,1)
#A.add('Conv2D',3,1,2)
A.add('NN_MP',30,5,2,beta=1)
A.add('Flat',1,2)
#A.add('oneHot')
A.add('NN',50,beta=0.5,stepL=300,inMD='sample')
#A.add('GG',nnodes,RN,stepL,inMD,reSL)





#%%

outdata = A.init_train(T_data)
#outdata = A.sv_train(bsdata,t_label)

#frrt = np.dot(raw_tdata[20000:40000],A.frame[0].bsmx.T)
#bmu = np.argmax(frrt,axis=-1)
#np.unique(bmu,return_counts=True)
#%%
#outdata = A.us_train(bsdata)
outdata = A.us_train(T_data)


#%%
#LLsv_train(indata,label,nlb(required),stepL,delta)
bsdata = raw_tdata[datalen:datalen*2]
t_label = label[datalen:datalen*2]
outdata = A.LLsv_train(bsdata,t_label,nlb,200,delta)

#%%
A.LLtestTrain(raw_tdata[30000:40000],label[30000:40000],ts_data,ts_label,nlb)
#t_label = label[datalen:datalen*2]

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
r_data = bsdata[Elist!=1]
r_label = t_label[Elist!=1]

for i in range(lp):
    bsdata = raw_tdata[datalen*(i+1):datalen*(i+2)]
    t_label = label[datalen*(i+1):datalen*(i+2)]
    bsdata_p = bsdata+r_data
    label_p = t_label+r_label
    Elist = A.forward(bsdata_p)
    r_data = bsdata_p[Elist!=1]
    r_label = label_p[Elist!=1]
    


#%%




#%%
outdata = np.dot(bsdata,A.frame[0].bsmx.T)
bmu = np.argmax(outdata,axis=-1)
print(len(np.unique(bmu)))
apmx = np.zeros((nnodes,nlb))
for i in range(datalen):
    apmx[bmu[i],int(t_label[i])] += outdata[i,bmu[i]]*0.2

aplb = np.argmax(apmx,axis=-1)
pr_label = aplb[bmu]
Elist = np.equal(pr_label,t_label)+0
print(sum(Elist))
#%%

Eindex = np.where(Elist==0)[1]
Edata = bsdata[Eindex]
Elabel = t_label[Eindex]
bsdata = np.concatenate((Edata,bsdata))
t_label = np.concatenate((Elabel,t_label))
Elist = A.train(bsdata,t_label)



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

Rlabel = train_data[:,0]
Rdata = train_data[:,1:]
ts_data = test_data[:,1:]
ts_label = test_data[:,0]



#%%
Adatalen = 50000

tRdata = Rdata[:Adatalen]
tRlabel = Rlabel[:Adatalen]

#data augmentation
from scipy.ndimage.interpolation import shift

def shift_image(image, dx, dy):
    image = image.reshape((28, 28))
    shifted_image = shift(image, [dy, dx], cval=0, mode="constant")
    return shifted_image.reshape([-1])

AtRdata = [image for image in tRdata]
AtRlabel = [image for image in tRlabel]

for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
     for image, label in zip(tRdata, tRlabel):
             AtRdata.append(shift_image(image, dx, dy))
             AtRlabel.append(label)

#AtRdata = np.array(AtRdata)
#AtRlabel = np.array(AtRlabel)


shuffle_idx = np.random.permutation(len(AtRdata))
AtRdata = np.array(AtRdata)[shuffle_idx]
AtRlabel = np.array(AtRlabel)[shuffle_idx]

print(AtRdata.shape)

#%%

import models
from models import model

importlib.reload(models)
#importlib.reload(verf)

#%%

datalen = 50000

flow_size = [datalen,[28,28]]

A = model(flow_size)

#A.add('padding',1,'constant')



#A.add('onoff','0.01')
A.add('norm','UV')

A.add('CNP',20,4,1,1,2,2,'no',alpha=1,beta=4,gamma=1.4,delta=0.3,ep=0.1,stepL=1000,reSL=0.8,TP=(0.2,2),SPlim=6)
#A.add('deleuze',0)
A.add('salience',2)
#A.add('oneHot',0.1,'drop')
#A.add('oneHot',0.8,'relu')
A.add('oneHot',7,'delete')
#A.add('oneHot',0.8,'thres')
#A.add('EP',0.95)

#A.add('Flat',1,2)
A.add('CNP',40,4,1,2,3,2,'no',alpha=1,beta=4,gamma=1.6,delta=0.3,ep=0,stepL=1000,reSL=0.7,TP=(0.1,900),SPlim=3)
#A.add('salience',2)
#A.add('oneHot',0.1,'relu')
#A.add('oneHot',0.1,'drop')
A.add('oneHot',15,'delete')
#A.add('oneHot',0.2,'thres')
A.add('flat',1,2)
A.add('NN',1200,'no',alpha=1,beta=4,gamma=1.8,delta=1.9,ep=0,stepL=5000,inMD=0,reSL=0.5)


#%%
#A.frame[-1].delta=0.25
import timeit


start = timeit.default_timer()


#Tlist = [1,1,1,1,1,1,1,1,0]
Tdata = AtRdata[:datalen]
Tlabel = AtRlabel[:datalen]
#A.SoL_train(Tdata,Tlabel,ts_data,ts_label,10,10,1)
A.So_train(Tdata,Tlabel,ts_data,ts_label,10,2,1,2)

stop = timeit.default_timer()

print('Time: ', stop - start) 

#outdata = A.sv_train(bsdata,t_label)

#frrt = np.dot(raw_tdata[20000:40000],A.frame[0].bsmx.T)
#bmu = np.argmax(frrt,axis=-1)
#np.unique(bmu,return_counts=True)

#B2L = A.find_B2L(A.frame[-1].bsmx,10)
#outdata = A.Tforward(ts_data)
#A.So_test(outdata,ts_label,B2L)

#%%
outdata = A.forward(Tdata)
            #self.So_test(outdata,tlabel,B2L)
A.Mtest(outdata,A.frame[-1].bsmx,10,Tlabel,2)

#%%
B2L = A.find_B2L(A.frame[-1].bsmx,10)
outdata = A.forward(Tdata)
p_label = A.So_test(outdata,Tlabel,B2L)


#%%


#NN(nnodes(required),RN,alpha,beta,stepL,inMD,reSL,TP)
#NN_MP(nnodes,MPsize,MPdia,MPstride(required),RN,alpha,beta,stepL,inMD,reSL,TP,SPlim)

flow_size = [datalen,[28,28]]

A = model(flow_size)

#A.add('bipolar',2,0.5,'abs')
#A.add('bipolar',2,0.5,'abs')
#A.add('bipolar',2,0.5,'abs')
#A.add('deleuze',0)
A.add('padding',1,'constant')

A.add('norm','UV')
A.add('conv2D',5,1,1)
#A.add('Conv2D',3,1,2)
#A.add('salience',0.01)
A.add('NNmp',25,2,2,alpha=0.5,beta=4,gamma=1.5,delta=0.1,stepL=500,inMD='sample',TP=0.45,SPlim=100000)
#A.add('deleuze',0)
A.add('salience',2)
A.add('oneHot',0.1,'drop')
#A.add('oneHot',0.1,'relu')
A.add('oneHot',7,'delete')
#A.add('oneHot',0.2,'thres')
#A.add('EP',0.95)
A.add('conv2D',4,1,2)
A.add('flat',1,2)
#A.add('oneHot',0)
#A.add('Conv2D',3,1,2)
#A.add('Flat',1,2)
A.add('NNmp',40,2,1,alpha=1,beta=4,gamma=1.5,delta=0.1,stepL=500,inMD='sample',TP=0.05,SPlim=50000)
A.add('salience',2)
#A.add('oneHot',0.1,'relu')
A.add('oneHot',0.1,'drop')
A.add('oneHot',10,'delete')
#A.add('oneHot',0.2,'thres')
A.add('conv2D',4,1,2)
A.add('flat',1,2)
A.add('NNmp',64,2,1,alpha=1,beta=4,gamma=1.5,delta=0.2,stepL=500,inMD='sample',TP=0.05,SPlim=50000)
A.add('salience',2)
#A.add('oneHot',0.1,'relu')
#A.add('oneHot',0.1,'drop')
A.add('oneHot',12,'delete')
A.add('flat',1,2)
A.add('NN',800,alpha=1,beta=5,gamma=1.3,delta=0.35,stepL=1000,inMD='sample')
#A.add('deleuze',0)
#A.add('GG',nnodes,RN,stepL,inMD,reSL)



#%%

flow_size = [datalen,[28,28]]
A = model(flow_size,alpha,beta,gamma,delta,ep,RN,stepL,inMD,reSL,TP,SPlim)
#A.add('Norm','UVp')
#A.add('bipolar',2,0.5,'abs')
#A.add('Bipolar',2,0.5,'abs')
#A.add('Bipolar',2,0.5,'abs')
A.add('padding',1,'constant')

A.add('norm','UV')
A.add('conv2D',5,1,1)
#A.add('Conv2D',3,1,2)
#A.add('salience',0.01)
A.add('NNmp',32,2,2,alpha=0.8,beta=5,gamma=1.5,delta=0.1,stepL=500,inMD='sample',TP=0,SPlim=10000)
#A.add('deleuze',0)
A.add('oneHot',5)
A.add('flat',1,2)
A.add('NN',400,alpha=0.8,beta=5,gamma=1.5,delta=0.1,stepL=500,inMD='sample')


#%%

flow_size = [datalen,[28,28]]
A = model(flow_size,alpha,beta,gamma,delta,ep,RN,stepL,inMD,reSL,TP,SPlim)
#A.add('Norm','UVp')
A.add('NN',400,alpha=0.2,beta=4,gamma=3,stepL=2000,inMD='sample')

#%%
datalen = 10000
RN = 0.000005
#CFtres = 0.001
alpha = 0.7
beta = 5
#CF = 0
SPlim = 20000
gamma = 1.5
delta = 0.3
ep = 0
nnodes = 30
#capa = 4
stepL = 500
#MPsize = [28,28]
#MPdia = 4
#MPstride = 2

inMD = 'sample'
reSL = 4

TP = 0.05


#T_label = label[:datalen]



#%%
T_data = raw_tdata[:datalen]
T_label = label[:datalen]

A.frame[-1].delta=0.7

outdata = A.init_train(T_data,T_label,10)

#%%
A.frame[-8].alpha = 0.8
A.frame[-8].beta = 5
A.frame[-8].gamma = 1.5
A.frame[-8].delta = 0.05

A.frame[-4].alpha = 0.8
A.frame[-4].beta = 5
A.frame[-4].gamma = 1.5
A.frame[-4].delta = 0.05

A.frame[-1].alpha = 0.8
A.frame[-1].beta = 5
A.frame[-1].gamma = 1.5
A.frame[-1].delta = 0.35

#%%
#outdata = A.us_train(bsdata)
U_data = raw_tdata[20000:30000]
U_label = label[20000:30000]
Tlist = [1]
#Tlist = [1,1]
outdata = A.us_train(U_data,U_label,10,Tlist)


#%%
A.frame[-1].alpha = 0.2
A.frame[-1].beta = 0.15
Tdata = raw_tdata[10000:40000]
Tlabel = label[10000:40000]
A.So_train(Tdata,Tlabel,10,1,1,0.25)

#%%


A.frame[-1].alpha = 0.2
A.frame[-1].beta = 0.1
Tdata = raw_tdata[30000:40000]
Tlabel = label[30000:40000]
A.So_train(Tdata,Tlabel,10,1,1,0.3)


#%%
#LLsv_train(indata,label,nlb(required),stepL,delta)
A.frame[-3].alpha = 0
A.frame[-3].beta = 0
A.frame[-3].gamma = 0
A.frame[-3].delta = 2
A.frame[-3].ep = 0.8
A.frame[-3].TP = 0.05
#%%
A.frame[-1].alpha = 0.2
A.frame[-1].beta = 0.05
A.frame[-1].gamma = 0
A.frame[-1].delta = 1
A.frame[-1].ep = 0
#%%
bsdata = raw_tdata[10000:30000]
t_label = label[10000:30000]
outdata = A.LLsv_train(bsdata,t_label,nlb,400)

#%%
import matplotlib.pyplot as plt 
bsmx = A.frame[-1].bsmx[:,:-10]
plt.matshow(bsmx[3].reshape(28,28))
#%%
bs4 = A.frame[-1].bsmx
for i in range(10):
    plt.matshow(bs4[i].reshape(5,5))



#%%
out = A.LLtestTrain(raw_tdata[30000:40000],label[30000:40000],ts_data,ts_label,nlb)
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
    
    def normalize(data,norm_mode='UV'):
        #universal normalization tool, now works for hi-dim
        #UV mode: Unit Vector, default, keep dot(A,A)=1
        if norm_mode=='Z':  #make mean=0,std=1
            mean = np.mean(data,axis=-1,keepdims=True)
            mean = np.tile(mean,data.shape[-1])
            st = np.std(data,axis=-1,keepdims=True)
            newdata = (data-mean)/st
        elif norm_mode=='MinMax':   #make min=0,max=1
            mi = np.min(data,axis=-1,keepdims=True)
            mi = np.tile(mi,data.shape[-1])
            ma = np.max(data,axis=-1,keepdims=True)
            ma = np.tile(ma,data.shape[-1])
            newdata = (data-mi)/(ma-mi+0.0000001)
        elif norm_mode=='UV':   #keep dot(A,A)=1
            nr = np.linalg.norm(data,axis=-1,keepdims=True)
            newdata = data/(nr+0.0000001)
        elif norm_mode=='MMZ':   #keep mean=0, max-min=1
            mean = np.mean(data,axis=-1,keepdims=True)
            mean = np.tile(mean,data.shape[-1])
            mi = np.min(data,axis=-1,keepdims=True)
            ma = np.max(data,axis=-1,keepdims=True)
            newdata = (data-mean)/(ma-mi+0.000001)
        elif norm_mode=='OneMean':
            mean = np.mean(data,axis=-1,keepdims=True)
            mean = np.tile(mean,data.shape[-1])
            newdata = data/mean 
            
        return newdata


#%%
        
nnodes = 128

Tdata = raw_tdata[:11000]
Tdata = normalize(Tdata)
bsmx = Tdata[-nnodes:]
Tdata = Tdata[:10000]
    
frRT = np.dot(Tdata,bsmx.T).T

#%%
topN = 60
bbu = np.argpartition(frRT, -topN,axis = -1)[:,-topN:]
uni,s_freq = np.unique(bbu,return_counts=True)
Tlist = uni[np.where(s_freq>1)]

Ulist = np.zeros(nnodes)
for i in range(nnodes):
    Ulist[i] = len(np.intersect1d(bbu[i],Tlist))

#%%
bmu = np.zeros(10000)

for i in range(10000):
    hit = np.where(bbu==i)[0]
    if len(hit) == 0:
        bmu[i] = np.NaN
    elif len(hit) == 1:
        bmu[i] = hit
    elif len(hit)>1:
        bmu[i] = hit[np.argmax(Ulist[hit])]
        
#%%
        
def find_bmu(frRT,topN):
    nnodes = frRT.shape[0]
    dlen = frRT.shape[1]
    bbu = np.argpartition(frRT, -topN,axis = -1)[:,-topN:]
    uni,s_freq = np.unique(bbu,return_counts=True)
    Tlist = uni[np.where(s_freq>1)]

    Ulist = np.zeros(nnodes)
    for i in range(nnodes):
        Ulist[i] = len(np.intersect1d(bbu[i],Tlist))
        
    for i in range(dlen):
        hit = np.where(bbu==i)[0]
        hitP = np.append(hit,nnodes)
        UlistP = np.append(Ulist,0)
        bmu[i] = hitP[np.argmax(UlistP[hitP])]
    return bmu
        
#%%

    def find_bmu(frRT,topN):   #limited firing number bmu method
        frRT = frRT.T
        nnodes = frRT.shape[0]
        dlen = frRT.shape[1]
        
        bbu = np.argpartition(frRT, -topN,axis = -1)[:,-topN:]   #sample No. of topN frRT for each node 
        uni,s_freq = np.unique(bbu,return_counts=True)    #match number for each sample
        Tlist = uni[np.where(s_freq>1)]  #what samples have mutiple matches
    
        Ulist = np.zeros(nnodes)
        for i in range(nnodes):
            Ulist[i] = len(np.intersect1d(bbu[i],Tlist))  #how many mutiple-match samples for each nodes
        print(topN,Ulist) 
        bmu = np.zeros(dlen)        
        for i in range(dlen):
            hit = np.where(bbu==i)[0]
            hitP = np.append(hit,nnodes)
            UlistP = np.append(Ulist,0)
            bmu[i] = hitP[np.argmax(UlistP[hitP])]   #for multiple match, the largest Ulist nodes be the bmu
        bmu = bmu.astype(int)   #for indexing
        return bmu
    
#%%
Tdata = raw_tdata[:10000]
bsmx = raw_tdata[10000:10128]

#%%
def find_Ebmu1(frRT,av):  #bmu with equality
    
    MfrRT = np.pad(frRT,((0,0),(1,0)),'constant')  #add 0s as node #0, for argmax of 0s column = 0 
    
    bmu = np.zeros(len(frRT))   #creat empty bmu list to fill
    
    for i in range(frRT.shape[-1]):      
        Tbmu = np.argmax(MfrRT,axis=-1)  #calculate bmu every loop 
        uni,s_freq = np.unique(Tbmu,return_counts=True)
        MXnode = uni[np.argmax(s_freq[1:])+1] #find out nodes that fires most
        #print(np.argmax(s_freq[1:]))
        if np.max(s_freq[1:])>av:  #first node(0s) not included
            MXf = MfrRT[:,MXnode]   #select its frRT
            bmS = np.argpartition(MXf, -av)[-av:]  #findout top samples 
            bmu[bmS] = MXnode  #fill to bmu list
            MfrRT[bmS] = 0    #clear frRT row & column, will not apear again in bmu
            MfrRT[:,MXnode]= 0
         
        else:
            bmu[Tbmu>0]=Tbmu[Tbmu>0]   #fill rest of bmus
            break #stop until no hit freq>av
    bmu = bmu-1   #remove 0s column
    return bmu
#%%
    def find_Ebmu2(Tdata,bsmx,av):  #bmu with equality
        Ebsmx = bsmx+0
        ETdata = Tdata+0
        nnodes = bsmx.shape[0]
        dW = bsmx.shape[1]
        
        for i in range(nnodes):
            frRT = np.dot(ETdata,Ebsmx.T)
            Tbmu = np.argmax(frRT,axis=-1) #calculate bmu every loop 
            uni,s_freq = np.unique(Tbmu,return_counts=True)
            MXnode = uni[np.argmax(s_freq)] #nodes that fires most
            #print(MXnode)
            if np.max(s_freq)>av:  #if its fire freq>av
                maxUF = frRT[:,MXnode]   #select its frRT
                bmS = np.argpartition(maxUF, -av)[-av:]  #findout av samples # that most intensive fired(to keep)
                
                Rmx = np.zeros(dW)  #make onehot matrix,replace selected samples and bsmx(make them not bmu again)
                Rmx[MXnode] = 1
                Ebsmx[MXnode]= Rmx
                ETdata[bmS] = Rmx
            else:
                break #stop until no hit freq>av
        return Tbmu

#%%
    def find_Ebmu3(frRT,av1):  #bmu with equality

        
        for i in range(frRT.shape[1]):
            Tbmu = np.argmax(frRT,axis=-1) #calculate bmu every loop 
            uni,s_freq = np.unique(Tbmu,return_counts=True)
            MXnode = uni[np.argmax(s_freq)] #nodes that fires most
            #print(MXnode)
            if np.max(s_freq)>av1:  #if its fire freq>av
                #print(np.max(s_freq),av1)
                maxUF = frRT[:,MXnode]   #select its frRT
                bmS = np.argpartition(maxUF, -av1)[-av1:]  #findout av samples # that most intensive fired(to keep)
                frRT[bmS] = 0    #clear frRT row & column, will not apear again in bmu
                frRT[:,MXnode]= 0
                frRT[bmS,MXnode]=1
            else:
                #print(np.max(s_freq),av1)
                break #stop until no hit freq>av
        return Tbmu
#%%    


frRT=np.dot(Tdata,bsmx.T)
start = timeit.default_timer()

E1 = find_Ebmu1(frRT,120)
stop = timeit.default_timer()
print('Time: ', stop - start) 

start = timeit.default_timer()
E2 = find_Ebmu2(Tdata,bsmx,120)
stop = timeit.default_timer()
print('Time: ', stop - start) 


start = timeit.default_timer()
E3 = find_Ebmu3(frRT,120)
stop = timeit.default_timer()
print('Time: ', stop - start) 

#%%

def zipper(bsdata,label,nlb,Lweight):    #zip bsdata and labels, norm included
        #bsdata = self.normalize(bsdata,'UV')   #prepare 1D training data
    dlen = len(bsdata)
        
    apmx = np.zeros((dlen,nlb))     #prepare onehot label matrix     
    Tlabel = label.astype(int)
    apmx[range(dlen),Tlabel] = Lweight     
        
    duodata = np.hstack([bsdata,apmx])    #zip
    return duodata
        
#%%
start = timeit.default_timer()
G = zipper(raw_tdata,label,10,0.4)
stop = timeit.default_timer()
print('Time: ', stop - start) 
#%%
start = timeit.default_timer()
#outdata = np.random.permutation(Tdata)[:3700]
index = np.random.choice(10000,3700,replace=False)
outdata = Tdata[index]
stop = timeit.default_timer()
print('Time: ', stop - start) 

#%%
mx = np.random.rand(90,32)
NE = 9
mxS = np.sort(mx,axis=-1)
thres = mxS[...,-NE]
thresMX = np.repeat(thres[...,np.newaxis],mx.shape[-1],axis=-1)
newmx = mx+0
newmx = newmx-thresMX       
newmx[newmx<0]=0

#%%

import timeit


start = timeit.default_timer()
D2 = np.dot(D1,mx.T)

stop = timeit.default_timer()
print('Time: ', stop - start)


#%%
def drop(mx,NE):    
    abLen = np.prod(mx.shape)
    dropNum = int(abLen*NE)
    index = np.random.choice(abLen,dropNum,replace=False)
    newmx = mx.reshape(abLen)
    newmx[index] = 0
    newmx = newmx.reshape(mx.shape)
    return newmx

#%%
from scipy.ndimage.interpolation import shift

def shift_image(image, dx, dy):
    image = image.reshape((28, 28))
    shifted_image = shift(image, [dy, dx], cval=0, mode="constant")
    return shifted_image.reshape([-1])

#%%
   def pl_index(vsize,hsize,dia,stride):
        #convo index generator
        #no padding. Do padding somewhere else
        #dia: convo window diameter
        #noc: number of cuts
        #stride: >=1
        v_noc = -(-(vsize-(dia-1))//stride)
        h_noc = -(-(hsize-(dia-1))//stride)
    
        index=[]
        for i in range(v_noc):
            for j in range(h_noc):
                dx = np.arange(dia**2)
                dx = i*hsize*stride+j*stride+(dx%dia)+hsize*(dx//dia)
                index.append(dx)
        return np.array(index),v_noc,h_noc
    
#%%
        def DD(mx,NE):
            mxS = np.sort(mx,axis=-1)
            thres = mxS[...,NE]
            thresMX = np.repeat(thres[...,np.newaxis],mx.shape[-1],axis=-1)
            newmx = mx+0
            newmx[newmx<thresMX]=0
            return newmx

    
    
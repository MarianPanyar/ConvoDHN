#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 10:00:28 2021

@author: MarianPanya


"""



import numpy as np
import scipy
import matplotlib.pyplot as plt 

#from scipy import ndimage


class Layer(object):

    def __init__(self,csize):
        self.csize=csize  #input size, a list
    #forward: forward，including train and predict
    #init_train: forward and , layer by layer
    #train: train
    
    def forward(self,indata):
        raise NotImplementedError()
        
    def init_train(self,indata):   #(if needed) build matrices and train in layer itself
        outdata = self.forward(indata)
        return outdata
        #initial and training in each layer. Should be overwrite when init_train is really needed.
    
    def us_train(self,indata): #for layers not trainable
        outdata = self.forward(indata)
        return outdata
    
    def SEGsv_train(self,indata,label): #for layers not trainable
        outdata = self.us_train(indata)
        return outdata

        
class flat(Layer):
    #2D(or more) to 1D
    def __init__(self,W2,FD):
        #W2: where to flat, the lower dimention
        #FD: how many dimentions to flat
        self.W2=W2
        self.FD=FD
        
    def forward(self,indata):
        S1 = list(indata.shape)
        if self.W2==1:
            outdata = indata.reshape((S1[:-self.W2-self.FD+1]+[np.prod(S1[-self.W2-self.FD+1:])]))
        else:
            outdata = indata.reshape((S1[:-self.W2-self.FD+1]+[np.prod(S1[-self.W2-self.FD+1:-self.W2+1])]+S1[-self.W2+1:]))
        return outdata


class bipolar(Layer):
    #center-on or center-off neurons. sharpen the image 
    def __init__(self,csize,radius,beth,mode='on'):
        #creat a on-off center filter
        #csize: size of 2D data
        #radius: radius of peripheral depression
        #beth: peripheral haro weight
        #mode: on-center, off-center, on+off center
        self.csize=csize
        self.radius=radius
        self.beth=beth
        self.mode=mode
        
    def forward(self,data):
        sqdata = np.reshape(data,(len(data),self.csize[0],self.csize[1]))
        g_data = scipy.ndimage.gaussian_filter1d(sqdata,sigma=self.radius,axis=1)
        g_data = scipy.ndimage.gaussian_filter1d(g_data,sigma=self.radius,axis=2)
        on_center = sqdata-self.beth*g_data
        if self.mode=='on':
            osqdata=on_center
        elif self.mode=='off':
            osqdata=-on_center
        elif self.mode=='abs':
            osqdata=abs(on_center)
        outdata = np.reshape(osqdata,(len(data),self.csize[0]*self.csize[1]))
        return outdata

class oneHot(Layer):   #Last dimension one hot
    #only the max-firing pixel remain(in a column)
    def __init__(self,mode=0):
        self.mode = mode   #OneZero mode convert all max element to 1
        
    def forward(self,mx):
        #one hot bmu*rate array(convert non-max rate unit to 0). 
        #works for hi-dim
        mxT = mx.T
        mxT[mxT != mxT.max(0)] = 0
        if self.mode == 'OneZero':
            mxT[mxT == mxT.max(0)] = 1
        return mxT.T


class CaP(Layer):
    #pooling & convolution 
    #csize: size of oringinal square, should be like [x,y]
    #diameter: size of pooling diameter
    #stride: stride center to center. 
    def __init__(self,csize,diameter,stride=1,W2=1):
        self.csize=csize
        self.diameter=diameter
        self.stride=stride
        self.W2=W2
        self.out_index,v_noc,h_noc = pl_index(csize[0],csize[1],self.diameter,stride)

class convolution(CaP):
    #cut one sample into squares
    def __init__(self,csize,dia,stride=1,W2=1):
        #csize: original size [x,y]
        #radius: radius of kernel, diameter = 2*radius+1
        #print(csize)
        super().__init__(csize,dia,stride)
        self.W2=W2

    def forward(self,data):
        #output all tiles as 1D array, for training or forwarding
        if self.W2 ==1:
            tile_data = data[...,self.out_index]
        elif self.W2==2:
            tile_data = data[...,self.out_index,:]
        elif self.W2==3:
            tile_data = data[...,self.out_index,:,:]
        return tile_data
    
    
class padding(Layer):
    #padding
    def __init__(self,pd_size,margin,mode):
        self.pd_size = pd_size  #padding input size
        self.margin = margin
        self.mode = mode
        
    def forward(self,indata):
        sqdata = indata.reshape(list(np.shape(indata)[:-1])+self.pd_size)
        in_dim = len(np.shape(indata))
        sqoutdata = np.pad(sqdata,((0,0),)*(in_dim-1)+((self.margin,self.margin),(self.margin,self.margin),),self.mode)
        outdata = sqoutdata.reshape(sqoutdata.shape[:-2]+(-1,))
        return outdata

#%%
#neural layers
        
class NN(Layer):
    #normal neural gas layer, no pooling, no apmx. yes RN.
    #211119 new version: both supervised and unsupervised. apmx, delta & prediction outside in Models
    def __init__(self,nnodes,RN,alpha,beta,stepL,inMD,reSL,TP):
        self.nnodes = nnodes     #number of nodes in matrix
        self.RN = RN             #random noise level
        self.alpha = alpha       #bsmx learning rate
        self.beta = beta         #metmx learning rate
        self.stepL = stepL       #training step length
        self.inMD = inMD         #init-matrix mode, 'sample' or 'TR'
        self.reSL = reSL         #Metric matrix's learning step length para
        self.TP = TP         #firing threshold potential
    
    
    def init_train(self,bsdata):   #train basal matrix (bsmx) without supervise
        bsdata = self.normalize(bsdata,'UV')   #prepare 1D training data
        
        self.bsmx = self.mxinit(bsdata,self.nnodes,self.inMD)  #creat basal matrix
        self.metmx = np.zeros(self.nnodes)  #creat metric matrix, all 0s
        
        self.BUKtrain(bsdata,self.stepL,self.beta)      #train matrix
        outdata = self.firing(self.bsmx,bsdata,self.metmx)   #output
        return outdata
    
        
    def normalize(self,data,norm_mode='UV'):  #universal normalization tool, now works for hi-dim
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
            nr = np.linalg.norm(data,axis=-1,keepdims=True)/np.shape(data)[-1]
            nr = np.tile(nr,data.shape[-1])*data.shape[-1]
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
    
    
    def mxinit(self,indata,nnodes,initMod):  #bsmx init
        if initMod == 'sample':         #choose random sample from indata
            flatdata = indata.reshape(np.prod(np.shape(indata)[:-1]),np.shape(indata)[-1])
            index = np.random.randint(len(flatdata),size=nnodes)
            matrix = flatdata[index]
        elif initMod == 'TR':            #TR: Tabula Rasa, random noise value
            matrix = np.random.rand(nnodes,np.shape(indata)[-1])
        matrix = self.normalize(matrix)
        return matrix
    
    
    def us_train(self,HDdata): #unsupervised train bsmx  
        HDdata = self.normalize(HDdata,'UV')  
        self.BUKtrain(HDdata,self.stepL,self.beta)  #train bsmx and metmx
        outdata = self.firing(self.bsmx,HDdata,self.metmx)  #output
        return outdata
    

    def BUKtrain(self,bsdata,stepL,Sbeta):  #unsupervised training module
        #split data in nos parts
        #beta(metmx learning rate) can change
        bsdataT = bsdata+0
        np.random.shuffle(bsdataT) #!!! do NOT output frRT from BUKtrain !!!

        ct = 0  #sample counting
        nos = len(bsdataT)//stepL  #number of steps

        for i in range(nos):
            trRN = self.RN*(1-i/nos)  #RN decrease every step, Simulated Annealing
            beta = Sbeta*(1-i/nos)  #metmx learning rate, SA
            alpha = self.alpha*(1-i/nos)
            segdata = bsdataT[ct:ct+stepL] #cut segdata
            
            self.SEGtrain(segdata,trRN,alpha,beta) #train bsmx  
            ct = ct + stepL
        return self.bsmx #not return frRT, because shuffled frRT is meaningless

    
    def SEGtrain(self,segdata,trRN,alpha,beta):  #training in one segament
        frRT = self.firing(self.bsmx,segdata,self.metmx)  #do not call self.forward cause it might be rewriten
        
        #MfrRT = frRT[np.max(frRT,axis=-1)>self.TP]   #only choose large enough frRT
        #Msegdata = segdata[np.max(frRT,axis=-1)>self.TP]
        #print(MfrRT.shape)
        #Mbmu = self.find_bmu(MfrRT)   
        
        self.bsmx = self.BSMXtrain(frRT,segdata,trRN,alpha)  #train bsmx
        self.metmx = self.metTrain(self.metmx,frRT,beta)
        #show_dia = int(len(self.bsmx[0])**(1/2))
        #plt.matshow(np.reshape(self.bsmx[0],(show_dia,show_dia)))
        return self.metmx  #whatever
    
    def BSMXtrain(self,frRT,segdata,trRN,alpha):   #train bsmx only
        bmu = self.find_bmu(frRT)
        for i in range(len(segdata)):
            self.bsmx[bmu[i]] += frRT[i][bmu[i]]*segdata[i]*alpha
            #core of training. If frRT[a,b]==0, bsmx no change
        self.bsmx += trRN*self.inRN(self.bsmx)*len(segdata)   #use random trRN
        #print(self.bsmx)
        self.bsmx = self.normalize(self.bsmx)  #normalize after learning and RNing
        return self.bsmx
    
    def metTrain(self,metmx,frRT,beta):  #UNsupervised metMX training
        bmu = self.find_bmu(frRT)
        freq = self.bmu_freq(bmu,self.nnodes)  #freq: firing frequency
        capa = len(frRT)/self.nnodes #goal for freq: avg firing freq
        metmx += ((freq-capa)/capa)*beta  #freq high, metmx +
        # -∞ < metmx < +∞
        print('fire freq = ',freq)
        print('metmx = ',self.metmx)
        print('metmx avg = ',np.sum(self.metmx)/self.nnodes)
        return metmx
        
    def firing(self,mx,data,metmx):   #frRT: fire rate with metricMX
        dtrt = np.dot(data,mx.T)   #dtrt: dot fire rate, not final fire rate
        frRT = dtrt**(self.reSL**metmx)   #use self.reSL here, to get a better curve
        #!!!do NOT normalize result   !!!but must normlize input
        return frRT
        
    def find_bmu(self,frRT):   #best match unit, works even for high dimention 
        bmu = np.argmax(frRT,axis=-1)
        return bmu
    
    def inRN(self,matrix):    #generate random noise mx, the same size as matrix
        ms = np.shape(matrix)
        RNmx = np.random.rand(ms[0],ms[1])
        return RNmx
        
    def bmu_freq(self,bmu,nnodes): #return frequency of mx's BMUs (with some tricks) 
        s_bmu = np.concatenate((bmu,np.arange(nnodes)))
        uni,s_freq = np.unique(s_bmu,return_counts=True)
        freq = s_freq-1
        return freq
    
    def forward(self,indata):   #forward without training, output to next layer
        indata = self.normalize(indata,'UV')
        outdata = self.firing(self.bsmx,indata,self.metmx)
        #outdata = self.rt2oh(bsrt) #one hot
        return outdata
    
    def SmetTrain(self,metmx,frRT,UE_frRT,delta,beta):  #supervied train metmx
        metmx = self.metTrain(metmx,frRT,beta*delta)    #us train with all samples
        metmx = self.metTrain(metmx,UE_frRT,beta)
        #train with unequal samples, more important than normal samples
        return metmx
    
    
class NN_MP(NN):   #training & maxpooling layer
    def __init__(self,nnodes,MPsize,MPdia,MPstride,RN,alpha,beta,stepL,inMD,reSL,TP,SPlim):
        super().__init__(nnodes,RN,alpha,beta,stepL,inMD,reSL,TP)
        self.SPlim = SPlim    #bsdata training sample limit
        self.MPsize = MPsize     #maxpooling 2D size like [a,b]
        self.MPdia = MPdia        #maxpooling tiles diameter
        self.MPstride = MPstride    #maxpooling stride
        
    def init_train(self,HDdata):   #build and unsupervised train
        #HDdata: data blocks made by convolution, shape be like (dlen,14*14,5*5)
        HDdata = self.normalize(HDdata,'UV')   #prepare 1D training data
        self.bsmx = self.mxinit(HDdata[0],self.nnodes,'TR') 
        self.metmx = np.zeros(self.nnodes)  #creat metric matrix, all 0s
        #frRT = self.firing(self.bsmx,HDdata,self.metmx)
        #print(np.sum(frRT[160],axis=-1),frRT.shape)
        self.BUKtrain(HDdata,self.stepL,self.beta)  #train bsmx/metmx from bsindata but donot use outdata 
        outdata = self.forward(HDdata) #use HDdata to output outdata
        return outdata
    
    def BUKtrain(self,HDdata,stepL,Sbeta):  #unsupervised training module
        #split HDdata in nos parts
        #beta(metmx learning rate) can change
        HDdataT = HDdata+0
        np.random.shuffle(HDdataT) #!!! do NOT output frRT from BUKtrain !!!

        ct = 0  #sample counting
        nos = len(HDdataT)//stepL  #number of steps

        for i in range(nos):
            trRN = self.RN*(1-i/nos)  #RN decrease every step, Simulated Annealing
            beta = Sbeta*(1-i/nos)  #metmx learning rate, SA
            alpha = self.alpha*(1-i/nos)
            segHDdata = HDdataT[ct:ct+stepL] #cut segdata
            #frRT = self.firing(self.bsmx,segHDdata,self.metmx)
            #print(np.max(frRT[350],axis=-1),frRT.shape)
            self.SEGtrain(segHDdata,trRN,alpha,beta) #train bsmx  
            ct = ct + stepL
        return self.bsmx #not return frRT, because shuffled frRT is meaningless

    
    def SEGtrain(self,HDdata,trRN,alpha,beta):  #training in one segament
        #HDdata = self.normalize(HDdata,'UV') 
        bsinFRdata = self.MPin(HDdata)
        frRT = self.firing(self.bsmx,bsinFRdata,self.metmx)  #do not call self.forward cause it might be rewriten
        #print(bsinFRdata[:10],frRT[:10])
        MfrRT,MbsinFRdata = self.TPfilter(frRT,bsinFRdata,self.TP)
        #print(MbsinFRdata[20:50])
        print(frRT.shape,MfrRT.shape)
        MfrRT = MfrRT[:self.SPlim]
        MbsinFRdata = MbsinFRdata[:self.SPlim]
        self.bsmx = self.BSMXtrain(MfrRT,MbsinFRdata,trRN,alpha)  #train bsmx
        self.metmx = self.metTrain(self.metmx,MfrRT,beta)
        #show_dia = int(len(self.bsmx[0])**(1/2))
        #plt.matshow(np.reshape(self.bsmx[0],(show_dia,show_dia)))
        return self.metmx  #whatever
    
    def TPfilter(self,frRT,indata,TP):   #frRT & indata must have same shape
        MfrRT = frRT[np.max(frRT,axis=-1)>TP]
        Mindata = indata[np.max(frRT,axis=-1)>TP]
        return MfrRT,Mindata
    
    def us_train(self,HDdata):  #unsupervised training
        HDdata = self.normalize(HDdata)
        self.BUKtrain(HDdata,self.stepL,self.beta)   #train with NN.BUKtrain(not NN.us_train to avoid normalization)
        outdata = self.forward(HDdata)   #but output from forward
        return outdata
    
    def MPin(self,HDdata):    #firing tile data
        HDfrRT = super().forward(HDdata)     #NN.forward, max pooling not included
        #HDfrRT = self.firing(self.bsmx,HDdata,self.metmx)
        #e.g. (dlen,23*23,5*5) to (dlen,23*23,nnodes)
        #print(np.sum(HDdata[200],axis=-1),HDdata.shape,np.sum(HDfrRT[200],axis=-1),HDfrRT.shape)
        tile_frRT,tile_index = self.MPtiles(HDfrRT)

        TmaxfrRT = np.max(tile_frRT,axis=-1)    #max firerate of every tile (dispite channel)
        #e.g. (dlen,12*12,5*5)
        MTindex = np.argmax(TmaxfrRT,axis=-1)  #max tile index, which tile has max frRT
        #e.g. (dlen,12*12),value:(0,5*5-1)
        HDindex = tile_index[np.arange(len(tile_index)),MTindex]  #max tile index in HDdata
        #e.g. (dlen,12*12),value:(0,23*23-1)
        inFRdata = self.find_MPindata(HDdata,HDindex)   #fired tiles in HDdata input
        #e.g. (dlen,12*12,5*5)
        bsinFRdata = self.left_flat(inFRdata)   #flat tiles to train
        #e.g. (dlen*12*12,5*5)
        #print(bsinFRdata[4:10])
        return bsinFRdata
    
    def MPtiles(self,HDfrRT):
        tile_index,v_noc,h_noc = pl_index(self.MPsize[0],self.MPsize[1],self.MPdia,self.MPstride)
        #e.g. tile_index.S=(12*12,5*5),v_noc=12,h_noc=12
        tile_frRT = HDfrRT[...,tile_index,:]  #fire rate of every node, every tile
        #e.g. (dlen,23*23,nnodes) to (dlen,12*12,5*5,nnodes)
        return tile_frRT,tile_index
    
    def find_MPindata(self,HDdata,HDindex): #findout MP indata causes fire in HDdata tiles
        dlen = len(HDdata)
        HDiter = (HDdata[x][HDindex[x]] for x in range(dlen))
        MPindata = np.array(list(HDiter))
        return MPindata
    
    def forward(self,HDdata):  #forward with maxpooling
        HDfrRT = super().forward(HDdata)   
        tile_frRT,tile_index = self.MPtiles(HDfrRT)
        outdata = np.max(tile_frRT,axis=-2)  #max fire rate of every node in every tile
        #e.g. (dlen,12*12,nnodes)
        return outdata
    
    def left_flat(self,HDdata): # make matrix (a1,a2,a3...,b) to (a1*a2*a3...,b)
        flat_data = HDdata.reshape(np.prod(HDdata.shape[:-1]),HDdata.shape[-1])
        return flat_data        

    

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






 
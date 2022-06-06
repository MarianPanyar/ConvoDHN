#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 10:00:28 2021

@author: MarianPanyar


"""



import numpy as np
#import scipy.ndimage
#import matplotlib.pyplot as plt 


class Layer(object):    #P class
    def __init__(self,csize):
        self.csize=csize  #input size as list
    
    def forward(self,indata):   #fire (and predict) without training
        raise NotImplementedError()
        
    def Tforward(self,indata):   #special forward for testing
        outdata = self.forward(indata)   
        return outdata
        
    def init_train(self,indata,label,nlb):   #(if needed) build matrices and train layer by layer
        outdata = self.forward(indata)   #for layers not trainable. Should be overwrite when needed.
        return outdata
    
    def us_train(self,indata,label,nlb):    #unsupervised train, layer by layer
        outdata = self.forward(indata)     #for layers not trainable. Should be overwrite when needed.
        return outdata

        
class flat(Layer):    #2D(or more) to 1D
    def __init__(self,W2,FD):
        self.W2=W2   #W2: where to flat, the lower dimention
        self.FD=FD   #FD: how many dimentions to flat
        
    def forward(self,indata):
        S1 = list(indata.shape)
        if self.W2==1:
            outdata = indata.reshape((S1[:-self.W2-self.FD+1]+[np.prod(S1[-self.W2-self.FD+1:])]))
        else:
            outdata = indata.reshape((S1[:-self.W2-self.FD+1]+[np.prod(S1[-self.W2-self.FD+1:-self.W2+1])]+S1[-self.W2+1:]))
        return outdata


class oneHot(Layer):   #all elements to 0 except max NE elements, for last dimension
    #works for hi-dim
    def __init__(self,NE=0,Mode='delete'):
        self.NE = NE   #numbers of elite
        self.Mode = Mode
        
    def forward(self,mx):
        if self.Mode == 'sig':
            newmx = 1/(1+np.exp(self.NE/mx-1/(1-mx)))
            
        elif self.Mode == 'drop':
            abLen = np.prod(mx.shape)
            dropNum = int(abLen*self.NE)
            index = np.random.choice(abLen,dropNum,replace=False)
            newmx = mx.reshape(abLen)
            newmx[index] = 0
            newmx = newmx.reshape(mx.shape)
            
        elif self.Mode == 'delueze':
            dim = len(mx.shape)
            d2a = tuple(np.arange(dim-1))
            avgdata = np.average(mx,axis=d2a)
            newmx = mx-avgdata
            
        elif self.Mode == 'relu':
            newmx = mx-self.NE
            newmx[newmx<0] = 0
            
        elif self.Mode == 'thres':
            newmx = mx+0
            newmx[newmx<self.NE] = 0
            
        else:
            mxS = np.sort(mx,axis=-1)
            thres = mxS[...,-self.NE]
            thresMX = np.repeat(thres[...,np.newaxis],mx.shape[-1],axis=-1)
            newmx = mx+0
            if self.Mode == 'delete':
                newmx[newmx<thresMX]=0
            elif self.Mode == 'minus':
                newmx = newmx-thresMX
                newmx[newmx<0]=0
            else:
                print('sorry, wrong Mode!')
        return newmx
    
    def Tforward(self,mx):
        if self.Mode == 'drop':
            newmx = mx
        else:
            newmx = self.forward(mx)
        return newmx

class CaP(Layer):    #pooling & convolution 
    def __init__(self,csize,diameter,stride=1):
        self.csize=csize   #size of square to pool or convo, should be like [x,y]
        self.diameter=diameter   #diameter: size of pooling diameter
        self.stride=stride   #stride center to center. 
        self.out_index,v_noc,h_noc = pl_index(csize[0],csize[1],self.diameter,stride)

class convolution(CaP):
    #cut one sample into squares
    def __init__(self,csize,dia,stride=1,W2=1):
        super().__init__(csize,dia,stride)
        self.W2=W2   #where 2 convo

    def forward(self,data):
        #output all tiles as 1D array, for training or forwarding
        if self.W2 ==1:
            tile_data = data[...,self.out_index]
        elif self.W2==2:
            tile_data = data[...,self.out_index,:]
        elif self.W2==3:
            tile_data = data[...,self.out_index,:,:]
        return tile_data
    
    
class padding(Layer):  #padding
    def __init__(self,pd_size,margin,mode):
        self.pd_size = pd_size  #padding input size
        self.margin = margin  
        self.mode = mode   #padding mode: constant,edge,minimum,reflect
        
    def forward(self,indata):
        sqdata = indata.reshape(list(np.shape(indata)[:-1])+self.pd_size)
        in_dim = len(np.shape(indata))
        sqoutdata = np.pad(sqdata,((0,0),)*(in_dim-1)+((self.margin,self.margin),(self.margin,self.margin),),self.mode)
        outdata = sqoutdata.reshape(sqoutdata.shape[:-2]+(-1,))
        return outdata
    
class norm(Layer): #normalization   
    def __init__(self,mode):
        self.mode = mode
    
    def forward(self,data):  #universal normalization tool, now works for hi-dim
        #UV mode: Unit Vector, default, keep dot(A,A)=1
        newdata = self.normalize(data,self.mode)
        return newdata
        
    def normalize(self,data,norm_mode='UV'):
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


class salience(Layer):    # sum<TP, tiles to 0
    def __init__(self,PW):
        self.PW = PW
    
    def forward(self,data):
        newdata = data**self.PW
        return newdata
    
        
class NN(Layer):
    #normal neural gas layer, no pooling, no apmx. yes RN.
    #211119 new version: supervised and unsupervised inside layer; apmx, delta & prediction outside in Models
    def __init__(self,nnodes,RN,alpha,beta,gamma,delta,ep,stepL,inMD,reSL):
        self.nnodes = nnodes     #number of nodes in matrix
        self.RN = RN             #random noise level
        self.alpha = alpha       #bsmx learning rate
        self.beta = beta         #hit freq factor, beginning, should >1
        self.gamma = gamma       #hit freq factor, ending
        self.delta = delta       #sample/label weight ratio, Lweight
        self.ep = ep             #unsuperviesd learning times
        self.stepL = stepL       #training step length
        self.inMD = inMD         #init-matrix mode, 'sample' or 'TR'
        self.reSL = reSL         #N/A
        #self.TP = TP         #firing threshold potential
    
    
    def init_train(self,bsdata,label,nlb):   #train basal matrix (bsmx) without supervise
        duodata = self.zipper(bsdata,label,nlb,self.delta)  #zip bsdata and label, normalization included
        
        self.bsmx = self.mxinit(duodata,self.nnodes,self.inMD)  #creat basal matrix
        
        self.BUKtrain(duodata,self.stepL)      #train matrix
        
        outdata = self.Pfiring(self.bsmx,bsdata)   #output
        #bmu = self.find_bmu(outdata)
        #uni,s_freq = np.unique(bmu,return_counts=True)
        #print(uni,s_freq)
        return outdata
    
    def zipper(self,bsdata,label,nlb,Lweight):    #zip bsdata and labels, norm included
        bsdata = self.normalize(bsdata,'UV')   #prepare 1D training data
        dlen = len(bsdata)
        
        apmx = np.zeros((dlen,nlb))     #prepare onehot label matrix     
        Tlabel = label.astype(int)
        apmx[range(dlen),Tlabel] = Lweight     
        
        duodata = np.hstack([bsdata,apmx])    #zip
        duodata = self.normalize(duodata,'UV')
        return duodata
        
        
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
    
    
    def mxinit(self,indata,nnodes,initMod):  #bsmx init
        if initMod == 'sample':         #choose random sample from indata
            flatdata = indata.reshape(np.prod(np.shape(indata)[:-1]),np.shape(indata)[-1]) #in case HDdata
            index = np.random.randint(len(flatdata),size=nnodes)
            matrix = flatdata[index]
        elif initMod == 'TR':            #TR: Tabula Rasa, random noise value
            matrix = np.random.rand(nnodes,np.shape(indata)[-1])
        matrix = self.normalize(matrix)
        return matrix
    
    
    def us_train(self,bsdata,label,nlb): #unsupervised train bsmx  
        duodata = self.zipper(bsdata,label,nlb,self.delta) 
        self.BUKtrain(duodata,self.stepL)  #train bsmx and metmx
        outdata = self.Pfiring(self.bsmx,bsdata)  #output
        #bmu = self.find_bmu(outdata)
        #uni,s_freq = np.unique(bmu,return_counts=True)
        #print(uni,s_freq)
        return outdata
    

    def BUKtrain(self,bsdata,stepL):  #unsupervised training module
        #split data in nos parts
        #beta(metmx learning rate) can change
        bsdataT = self.normalize(bsdata,'UV')  
        np.random.shuffle(bsdataT) #!!! do NOT output frRT from BUKtrain !!!

        ct = 0  #sample counting
        nos = len(bsdataT)//stepL  #number of steps
        print(nos)
        for i in range(nos):
            trRN = self.RN*(1-i/nos)  #RN decrease every step, Simulated Annealing
            alpha = self.alpha*(1-i/nos)
            segdata = bsdataT[ct:ct+stepL] #cut segdata
            reSL = (self.beta-self.gamma)*(i/(nos-1))+self.gamma
            
            self.SEGtrain(segdata,trRN,alpha,reSL) #train bsmx  
            ct = ct + stepL
        return self.bsmx #not return frRT, because shuffled frRT is meaningless

    
    def SEGtrain(self,segdata,trRN,alpha,reSL):  #training in one segament

        dlen = len(segdata)
        av = int(dlen*reSL//self.nnodes)  #max limit of fire freq 
        frRT = self.firing(self.bsmx,segdata)

        Ebmu = self.find_Ebmu(frRT,av)
        #print(Ebmu.shape,max(Ebmu),bsmxp.shape,frRTp.shape)
        for i in range(dlen):
            self.bsmx[Ebmu[i]] += frRT[i][Ebmu[i]]*segdata[i]*alpha
            #core of training. If frRT[a,b]==0, bsmx no change
        self.bsmx += trRN*self.inRN(self.bsmx)*dlen   #use random trRN
        #print(self.bsmx)
        self.bsmx = self.normalize(self.bsmx)  #normalize after learning and RNing
        return self.bsmx

    
    def Pfiring(self,mx,data):   #when width of data smaller than mx, fill with 0s
        nlb = mx.shape[-1] - data.shape[-1]
        Pmx = mx[:,:-nlb]
        #NO NEED more normalize cause only add 0s
        import timeit
        
        start = timeit.default_timer()
        frRT = self.firing(Pmx,data)
        stop = timeit.default_timer()
        print('FR Time: ', stop - start)
        return frRT

        
    def firing(self,mx,data):   #frRT: fire rate with metricMX
        frRT = np.dot(data,mx.T)  #dtrt: dot fire rate, not final fire rate
        #!!!do NOT normalize result   !!!but must normlize input
        return frRT
    
        
    def find_bmu(self,frRT):   #best match unit, works even for high dimention 
        bmu = np.argmax(frRT,axis=-1)
        return bmu

    def find_Ebmu(self,frRT,av):  #bmu with equality 
        MfrRT = np.pad(frRT,((0,0),(1,0)),'constant')  #add 0s as node #0, for argmax of 0s column = 0 
        bmu = np.zeros(len(frRT))   #creat empty bmu list to fill
        
        for i in range(frRT.shape[-1]):      
            Tbmu = np.argmax(MfrRT,axis=-1)  #calculate bmu every loop 
            uni,s_freq = np.unique(Tbmu,return_counts=True)
            MXnode = uni[np.argmax(s_freq[1:])+1] #find out nodes that fires most
            
            if np.max(s_freq[1:])>av:  #first node(0s) not included
                MXf = MfrRT[:,MXnode]   #select its frRT
                bmS = np.argpartition(MXf, -av)[-av:]  #findout top samples 
                bmu[bmS] = MXnode  #fill to bmu list
                MfrRT[bmS] = 0    #clear frRT row & column, will not apear again in bmu
                MfrRT[:,MXnode]= 0
             
            else:
                bmu[Tbmu>0]=Tbmu[Tbmu>0]   #fill rest of bmus
                break #stop until no hit freq>av
        bmu = (bmu-1).astype(int)   #remove 0s column
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

        outdata = self.Pfiring(self.bsmx,indata)
        #outdata = self.rt2oh(bsrt) #one hot
        return outdata
    
    
class NN_MP(NN):   #training & maxpooling layer. Only init_train and us_train. sv_train in model.
    def __init__(self,nnodes,MPsize,MPdia,MPstride,RN,alpha,beta,gamma,delta,ep,stepL,inMD,reSL,TP,SPlim):
        super().__init__(nnodes,RN,alpha,beta,gamma,delta,ep,stepL,inMD,reSL)
        self.SPlim = SPlim    #bsdata training sample limit
        self.TP = TP      #TP filter 
        self.MPsize = MPsize     #maxpooling 2D size like [a,b]
        self.MPdia = MPdia        #maxpooling tiles diameter
        self.MPstride = MPstride    #maxpooling stride
        
    def init_train(self,HDdata,label,nlb):   #build and unsupervised train
        #HDdata: data blocks made by convolution, shape be like (dlen,14*14,5*5)
        #HDdata = self.normalize(HDdata)
        #HD normalization for TPfilter
        bsdata,bslabel = self.bsPre(HDdata,label,self.TP)
        bsdata,bslabel = self.SPfilter(bsdata,bslabel,self.SPlim)

        duodata = self.zipper(bsdata,bslabel,nlb,self.delta)  #zip bsdata and label, normalization included
        self.bsmx = self.mxinit(duodata,self.nnodes,'sample') 
        #frRT = self.firing(self.bsmx,HDdata,self.metmx)
        #print(np.sum(frRT[160],axis=-1),frRT.shape)
        super().BUKtrain(duodata,self.stepL)  #train bsmx/metmx from bsindata but donot use outdata 

        outdata = self.forward(HDdata) #use HDdata for output outdata
        return outdata
        
    def bsPre(self,HDdata,label,TP):  #hiDimentional data to 1D, and delete non_saliency samples
        bsdata = self.left_flat(HDdata)
        bslabel = np.repeat(label,HDdata.shape[-2])
        outbsdata = self.TPfilter(bsdata,bsdata,TP)  #delete weak tiles
        outbslabel = self.TPfilter(bslabel,bsdata,TP)
        #bsdata = self.normalize(bsdata,'UV')   #prepare 1D training data
        return outbsdata,outbslabel
        
    def TPfilter(self,fEdata,sDdata,TP):   #filter, delete data when sDdata tiles' sum greater than TP
        #fEdata: data to be filtered; sDdata: data as filter standard
        fdata = fEdata[np.sum(sDdata,axis=-1)>TP]
        print(len(fEdata),len(fdata),TP)
        return fdata
    
    def SPfilter(self,data1,data2,SPlim):  #randomly choose SPlim samples from data
        if len(data1>=SPlim):
            index = np.random.choice(len(data1),SPlim,replace=False)
            outdata1 = data1[index]
            outdata2 = data2[index]
        else:
            outdata1 = data1
            outdata2 = data2
            print('sorry, Splim > dataLen!')
        return outdata1,outdata2
    
    def forward(self,HDdata):  #forward with maxpooling
        HDdata[np.sum(HDdata,axis=-1)<self.TP]=0  #TP fliter
        HDfrRT = super().forward(HDdata)

        tile_frRT,tile_index = self.MPtiles(HDfrRT)  #maxpooling index
        outdata = np.max(tile_frRT,axis=-2)  #max fire rate of every node in every tile
        #e.g. (dlen,12*12,nnodes)
        return outdata
    
    def Tforward(self,HDdata):  #forward with maxpooling
        #HDdata[np.sum(HDdata,axis=-1)<self.TP]=0  #TP fliter
        HDfrRT = super().forward(HDdata)

        tile_frRT,tile_index = self.MPtiles(HDfrRT)  #maxpooling index
        outdata = np.max(tile_frRT,axis=-2)  #max fire rate of every node in every tile
        #e.g. (dlen,12*12,nnodes)
        return outdata
    
    def us_train(self,HDdata,label,nlb):  #unsupervised training
        bsdata,bslabel = self.bsPre(HDdata,label,self.TP)
        bsdata,bslabel = self.SPfilter(bsdata,bslabel,self.SPlim)

        duodata = self.zipper(bsdata,bslabel,nlb,self.delta)  #zip bsdata and label, normalization included
        super().BUKtrain(duodata,self.stepL)  #train bsmx/metmx from bsindata but donot use outdata 

        outdata = self.forward(HDdata) #use HDdata for output outdata
        return outdata
    
    def MPtiles(self,HDfrRT):
        tile_index,v_noc,h_noc = pl_index(self.MPsize[0],self.MPsize[1],self.MPdia,self.MPstride)
        #e.g. tile_index.S=(12*12,5*5),v_noc=12,h_noc=12
        tile_frRT = HDfrRT[...,tile_index,:]  #fire rate of every node, every tile
        #e.g. (dlen,23*23,nnodes) to (dlen,12*12,5*5,nnodes)
        return tile_frRT,tile_index
    
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






 
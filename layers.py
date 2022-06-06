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
    def __init__(self,nnodes,oomode,RN,alpha,beta,gamma,delta,ep,stepL,inMD,reSL):
        self.nnodes = nnodes     #number of nodes in matrix
        self.oomode = oomode
        self.RN = RN             #random noise level
        self.alpha = alpha       #bsmx learning rate
        self.beta = beta         #hit freq factor, beginning, should >1
        self.gamma = gamma       #hit freq factor, ending
        self.delta = delta       #sample/label weight ratio, Lweight
        self.ep = ep             #alpha curve power
        self.stepL = stepL       #training step length
        self.inMD = inMD         #us training times
        self.reSL = reSL         #av curve power
        #self.TP = TP         #firing threshold potential
    
    
    def init_train(self,bsdata,label,nlb):   #train basal matrix (bsmx) without supervise
        oodata = self.onoff(bsdata,self.oomode)
        #onoffdata = self.normalize(onoffdata,'UV')   #prepare 1D training data
        duodata = self.zipper(oodata,label,nlb,self.delta)  #zip bsdata and label, normalization included
        
        self.bsmx = self.mxinit(duodata,self.nnodes,'sample')  #creat basal matrix
        
        self.BUKtrain(duodata,self.stepL)      #train matrix
        
        outdata = self.BUKforward(bsdata,self.stepL)   #output
        #bmu = self.find_bmu(outdata)
        #uni,s_freq = np.unique(bmu,return_counts=True)
        #print(uni,s_freq)
        return outdata
    
    def onoff(self,indata,oomode):
        Tdim = tuple(np.arange(indata.ndim))
        mean = np.mean(indata,axis=Tdim[:-1])
        
        ONdata = indata-mean
        OFFdata = 0-ONdata
        ONdata[ONdata<0]=0
        
        if oomode=='on':
            outdata = ONdata
        elif oomode=='onoff':
            OFFdata[OFFdata<0]=0
            outdata = np.concatenate((ONdata,OFFdata),axis=-1)
        else:
            outdata = indata
            
        outdata = self.normalize(outdata)
        return outdata
    
    def zipper(self,bsdata,label,nlb,Lweight):    #zip bsdata and labels, norm included
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
    
    
    def mxinit(self,indata,nnodes,initMod='sample'):  #bsmx init
        if initMod == 'sample':         #choose random sample from indata
            flatdata = indata.reshape(np.prod(np.shape(indata)[:-1]),np.shape(indata)[-1]) #in case HDdata
            index = np.random.randint(len(flatdata),size=nnodes)
            matrix = flatdata[index]
        elif initMod == 'TR':            #TR: Tabula Rasa, random noise value
            matrix = np.random.rand(nnodes,np.shape(indata)[-1])
        matrix = self.normalize(matrix)
        return matrix
    
    
    def us_train(self,bsdata,label,nlb): #unsupervised train bsmx
        if self.oomode == 'onoff' or 'on':
            bsdata = self.onoff(bsdata,self.oomode)
            
        bsdata = self.normalize(bsdata,'UV')   #prepare 1D training data
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
            alpha = self.alpha*((1-i/nos)**2)
            segdata = bsdataT[ct:ct+stepL] #cut segdata
            reSL = (self.beta-self.gamma)*((1-i/nos)**self.reSL)+self.gamma
            #print(reSL)
            self.SEGtrain(segdata,trRN,alpha,reSL) #train bsmx  
            ct = ct + stepL
        return self.bsmx #not return frRT, because shuffled frRT is meaningless

    
    def SEGtrain(self,segdata,trRN,alpha,reSL):  #training in one segament

        dlen = len(segdata)
        av = int(dlen*reSL//self.nnodes)  #max limit of fire freq 
        #print('av = ',av)
        frRT = self.firing(self.bsmx,segdata)

        Ebmu = self.find_Ebmu(frRT,av)
        #print(Ebmu.shape,max(Ebmu),bsmxp.shape,frRTp.shape)
        self.bsmx = self.SOMtrain(dlen,self.bsmx,Ebmu,segdata,alpha,frRT,self.ep)
        #for i in range(dlen):
            #self.bsmx[Ebmu[i]] += segdata[i]*alpha*(frRT[i][Ebmu[i]])
            #core of training. If frRT[a,b]==0, bsmx no change
        self.bsmx += trRN*self.inRN(self.bsmx)*dlen   #use random trRN
        #print(self.bsmx)
        self.bsmx = self.normalize(self.bsmx)  #normalize after learning and RNing
        return self.bsmx
    
    def SOMtrain(self,dlen,bsmx,Ebmu,segdata,alpha,frRT,ep):
        nnodes = bsmx.shape[0]
        for i in range(dlen):
            bsmx[(Ebmu[i]-2)%nnodes] += segdata[i]*alpha*(frRT[i][(Ebmu[i]-2)%nnodes])*(ep**2)
            bsmx[(Ebmu[i]-1)%nnodes] += segdata[i]*alpha*(frRT[i][(Ebmu[i]-1)%nnodes])*ep
            bsmx[Ebmu[i]] += segdata[i]*alpha*(frRT[i][Ebmu[i]])
            bsmx[(Ebmu[i]+1)%nnodes] += segdata[i]*alpha*(frRT[i][(Ebmu[i]+1)%nnodes])*ep
            bsmx[(Ebmu[i]+2)%nnodes] += segdata[i]*alpha*(frRT[i][(Ebmu[i]+2)%nnodes])*(ep**2)
        return bsmx


        
        

    
    def Pfiring(self,mx,data):   #when width of data smaller than mx, fill with 0s
        nlb = mx.shape[-1] - data.shape[-1]
        Pmx = mx[:,:-nlb]
        Pmx = self.normalize(Pmx,norm_mode='UV')
        frRT = self.firing(Pmx,data)
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
        #print(av)
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
                #print('Frn = ',s_freq[0])
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
        #indata = self.normalize(indata,'UV')
        indata = self.onoff(indata,self.oomode)
        #print(indata.shape)
        outdata = self.Pfiring(self.bsmx,indata)
        #print(len(outdata))
        #outdata = self.rt2oh(bsrt) #one hot
        return outdata
    
    def BUKforward(self,indata,stepL):
        outdata = BUKfuc(indata,stepL,self.forward)
                
        return outdata
    
    

    
    
class CNP(NN):
    def __init__(self,nnodes,Csize,CVdia,CVstride,CVw2,MPdia,MPstride,oomode,RN,alpha,beta,gamma,delta,ep,stepL,inMD,reSL,TP,SPlim):
        super().__init__(nnodes,oomode,RN,alpha,beta,gamma,delta,ep,stepL,inMD,reSL)
        self.SPlim = SPlim    #bsdata training sample limit, for every sample
        self.TP = TP      #TP filter 
        self.Csize = Csize     #input 2D size like [a,b]
        self.CVdia = CVdia
        self.CVstride = CVstride
        self.CVw2 = CVw2
        self.MPdia = MPdia        #maxpooling tiles diameter
        self.MPstride = MPstride    #maxpooling stride
        
    def init_train(self,indata,label,nlb):   #build and unsupervised train
        duodata = self.tile_c(indata,label,nlb)
        #print(duodata.shape)
        self.BSin_train(duodata,self.nnodes,'sample',self.stepL)
        #print(self.bsmx.shape)
        outdata = self.forward(indata)
        return outdata
        
    def tile_c(self,indata,label,nlb):  #tile collect for training
        datalen = len(indata)
        tileIDX = self.indGen(self.Csize,self.CVdia,self.SPlim,datalen)  #random tile index
        HDdata = np.array([indata[x,tileIDX[x]] for x in range(datalen)])   #tiles
        bsdata = HDdata.reshape(np.prod(HDdata.shape[:-self.CVw2]),np.prod(HDdata.shape[-self.CVw2:]))  #reshape to 1D tiles
        bslabel = np.repeat(label,self.SPlim)
        bsdata,bslabel = self.TPfilter(bsdata,bslabel,self.TP)  #delete weak tiles
        bsdata = self.normalize(bsdata,'UV')   #prepare 1D training data
        
        if self.oomode == 'onoff' or 'on':
            oodata = self.onoff(bsdata,self.oomode)
        else:
            oodata = bsdata
        duodata = self.zipper(oodata,bslabel,nlb,self.delta)  #zip bsdata and label, normalization included
        return duodata
    
    def BSin_train(self,duodata,nnodes,inMD,stepL):
        self.bsmx = self.mxinit(duodata,nnodes,inMD) 
        super().BUKtrain(duodata,stepL)  #train bsmx/metmx from bsindata but donot use outdata 
        return self.bsmx
  
    def TPfilter(self,data,label,TP):   #filter, delete data when sDdata tiles' sum greater than TP
        #fEdata: data to be filtered; sDdata: data as filter standard
        TileSum = np.sum(data,axis=-1)
        index = np.where((TileSum>TP[0])&(TileSum<TP[1]))
        fdata = data[index]
        flabel = label[index]
        print(len(data),len(fdata),TP)
        return fdata,flabel


    def conv2D(self,indata):
        CVindex,CVx,CVy = self.pl_index(self.Csize[0],self.Csize[1],self.CVdia,self.CVstride)
        
        if self.CVw2 ==1:
            HDdata = indata[...,CVindex]
        elif self.CVw2==2:
            HDdata = indata[...,CVindex,:]
            HDdata = HDdata.reshape(HDdata.shape[:-2]+(np.prod(HDdata.shape[-2:]),))
        elif self.CVw2==3:
            HDdata = indata[...,CVindex,:,:]
        return HDdata,CVx,CVy
    
    def pl_index(self,vsize,hsize,dia,stride):
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
    
    
    def forward(self,indata):  #forward with maxpooling
        #print(indata.shape)
        outdata = BUKfuc(indata,self.stepL,self.Tforward)        
        return outdata
    
    def Tforward(self,indata):  #forward with maxpooling
        HDdata,CVx,CVy = self.conv2D(indata)
        #print(HDdata.shape)
        HDfrRT = super().forward(HDdata)
        #print(HDdata.shape,HDfrRT.shape)
        tile_frRT,tile_index = self.MPtiles(HDfrRT,CVx,CVy)  #maxpooling index
        outdata = np.max(tile_frRT,axis=-2)  #max fire rate of every node in every tile
        #e.g. (dlen,12*12,nnodes)
        return outdata
    
    def us_train(self,indata,label,nlb):  #unsupervised training
        duodata = self.tile_c(indata,label,nlb)
        super().BUKtrain(duodata,self.stepL)
        outdata = self.forward(indata)
        return outdata
    
    def MPtiles(self,HDfrRT,CVx,CVy):
        tile_index,v_noc,h_noc = self.pl_index(CVx,CVy,self.MPdia,self.MPstride)
        #e.g. tile_index.S=(12*12,5*5),v_noc=12,h_noc=12
        tile_frRT = HDfrRT[...,tile_index,:]  #fire rate of every node, every tile
        #e.g. (dlen,23*23,nnodes) to (dlen,12*12,5*5,nnodes)
        return tile_frRT,tile_index
    
    def indGen(self,Csize,dia,SPlim,slen):
        x1 = np.arange(dia*dia)
        x1 = x1+(x1//dia)*(Csize[0]-dia)
        Sx = np.random.randint(Csize[0]-dia,size = SPlim*slen)
        Sy = np.random.randint(Csize[1]-dia,size = SPlim*slen)
        Sidx = Sy*Csize[0]+Sx
        aSidx = np.array([x+x1 for x in Sidx])
        aSidx = aSidx.reshape(slen,SPlim,dia*dia)
        return aSidx
     


def BUKfuc(indata,stepL,fuc):
    ct = 0
    nos = len(indata)//stepL
    rem = len(indata)%stepL
        
    if rem > 0:
        firstL = rem
    else:
        firstL = stepL
        nos = nos-1
    firstindata = indata[:firstL]
        
    outdata = fuc(firstindata)
                            
    for i in range(nos):
        segindata = indata[firstL+ct:firstL+ct+stepL] #cut segdata
        ct = ct+stepL
            
        segoutdata = fuc(segindata)
        outdata = np.concatenate((outdata,segoutdata),axis=0)
                
    return outdata





 
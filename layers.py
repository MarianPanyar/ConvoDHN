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
        self.csize=csize

    #forward: forward，including train and predict
    #init_train: forward and (if needed) build matrices and train in layer itself, layer by layer
    #train: train
    
    def forward(self,indata):
        raise NotImplementedError()
        
    def init_train(self,indata):
        outdata = self.forward(indata)
        return outdata
        #initial and training in each layer. Should be overwrite when init_train is really needed.
    
    def bs_train(self,indata):
        outdata = self.forward(indata)
        return outdata
    
    def ap_cacu(self,indata,label):
        outdata = self.forward(indata)
        return outdata
    
    def mpap_train(self,indata,label):
        outdata = self.forward(indata)
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
        self.inshape=S1
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

class oneHot(Layer):
    #only the max-firing pixel remain(in a column)
    def __init__(self,rate):
        self.rate = rate
        
    def forward(self,data):
        oh_filter = (data == data.max(axis=-1)[...,None]).astype(int)
        oh_matrix = self.rate*oh_filter*data+(1-self.rate)*data
        return oh_matrix



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
        CaP.__init__(self,csize,dia,stride)
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
        self.pd_size = pd_size
        self.margin = margin
        self.mode = mode
        
    def forward(self,indata):
        sqdata = indata.reshape(list(np.shape(indata)[:-1])+self.pd_size)
        in_dim = len(np.shape(indata))
        sqoutdata = np.pad(sqdata,((0,0),)*(in_dim-1)+((self.margin,self.margin),(self.margin,self.margin),),self.mode)
        outdata = sqoutdata.reshape(sqoutdata.shape[:-2]+(-1,))
        return outdata

class pooling(CaP):
    #pooling(not in use now)
    #csize: size of oringinal square, should be like [x,y]
    #diameter: size of pooling diameter
    #stride: stride center to center. if diameter=stride, then non-overlapping
    def __init__(self,csize,dia,stride=1,W2=2):
        #csize: original size [x,y]
        #radius: radius of kernel, diameter = 2*radius+1
        CaP.__init__(self,csize,dia,stride)
        self.W2=W2
    
    def forward(self,data):
        #output all tiles as 1D array, for training or forwarding
        #if self.W2 ==1:
        #    tile_data = data[...,self.out_index]
        if self.W2==2:
            tile_data = data[...,self.out_index,:]
        else:
            print('W2 must be 2')
        #elif self.W2==3:
        #    tile_data = data[...,self.out_index,:,:]
        out_data = np.max(tile_data,axis=-self.W2)
        self.max_index = np.argmax(tile_data,axis=-self.W2)
        self.bp_ne = np.zeros(data.shape)
        #mxZ = (out_data == out_data.max(axis=-1, keepdims=True)).astype(int)
        #out_data = out_data*mxZ
        return out_data


#%%
#neural layers
        
class NN(Layer):
    #normal neural gas layer, no pooling, no apmx. RN yes.
    #typical first neural layer 
    def __init__(self,nnodes,RN,alpha,gamma,capa,stepL,relu,inMD,refd):
        self.nnodes = nnodes     #number of nodes in matrix
        self.RN = RN             #random noise level
        self.alpha = alpha       #bsmx learning rate
        #self.beta = beta
        self.gamma = gamma       #RN's decreasing rate 
        self.capa = capa         #monopoly pruning threshold
        self.stepL = stepL       #training step length
        self.relu = relu         #firing treshold
        self.inMD = inMD         #init-matrix mode, 'sample' or 'TR'
        self.refd = refd         #refund parameter, for monopoly martix adjusting
    
    
    def init_train(self,HDdata):
        #train basal matrix (bsmx) without supervise
        
        HDdata = self.normalize(HDdata,'UV')   #prepare 1D training data
        self.bsmx = self.mxinit(HDdata,self.nnodes,self.inMD)  #creat basal matrix
        self.monopoly = np.ones(self.nnodes)  #creat monopoly matrix

        self.BUKtrain(HDdata)      #train matrix
        
        frrt = self.firing(self.bsmx,HDdata,self.monopoly)   #see output
        outdata = self.rt2oh(frrt)    #one-hot output
        #outdata = frrt
        return outdata
    
        
    def normalize(self,data,norm_mode='UV'):
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
    
    
    def mxinit(self,indata,nnodes,init):
        #bsmx init
        if init == 'sample':         #choose random sample from indata
            flatdata = indata.reshape(np.prod(np.shape(indata)[:-1]),np.shape(indata)[-1])
            index = np.random.randint(len(flatdata),size=nnodes)
            matrix = flatdata[index]
        elif init == 'TR':            #TR: Tabula Rasa, random noise value
            matrix = np.random.rand(nnodes,np.shape(indata)[-1])
        matrix = self.normalize(matrix)
        
        return matrix
    
    
    def bs_train(self,HDdata): #train bsmx without supervise    
        HDdata = self.normalize(HDdata,'UV')
        data = HDdata+0   #because data will shuffle in BUKtrain!!

        self.BUKtrain(data)  #train matrix
        
        frrt = self.firing(self.bsmx,HDdata,self.monopoly)
        outdata = self.rt2oh(frrt)
        #output
        return outdata
    
    
    def BUKtrain(self,bsdata):  #unsupervised training module
        #split data in stepLs
        np.random.shuffle(bsdata) #!!! So, do NOT output frrt from BUKtrain !!!

        ct = 0  #sample counting
        nos = len(bsdata)//self.stepL  #number of steps
        trRN = self.RN+0  #random noise for training use

        for i in range(nos):
            segdata = bsdata[ct:ct+self.stepL] #cut segdata
            ct = ct + self.stepL
            
            frrt,bmu = self.SEGtrain(segdata,trRN) #train bsmx  
            trRN = trRN*(1-self.gamma)  #RN decrease every step, Simulated Annealing

        return self.bsmx
    
    def SEGtrain(self,segdata,trRN):  #training in one segament

        frrt = self.firing(self.bsmx,segdata,self.monopoly)
        bmu = self.find_bmu(frrt)
        
        for i in range(len(segdata)):
            self.bsmx[bmu[i]] += frrt[i][bmu[i]]*segdata[i]*(self.alpha/len(segdata))
            
        self.bsmx += trRN*self.inRN(self.bsmx)/len(segdata)   #use random trRN
        
        self.bsmx = self.normalize(self.bsmx)  #normalize after learning and RNing
        
        freq = self.bmu_freq(bmu,self.nnodes)  #freq: firing frequency of each step
        self.monopoly = self.MPtrain(freq,self.capa,self.stepL)
        print(freq,self.monopoly)
        plt.matshow(np.reshape(self.bsmx[7],(28,28)))
        return frrt,bmu
    
    def MPtrain(self,freq,capa,datalen):  #UNsupervised MP training
        indexIN = np.where(freq<1,1,0) #low fr_freq node, increase
        self.monopoly += indexIN*(self.refd/datalen)
        
        indexDE = np.where(freq>capa,1,0)   #high fr_freq node, decrease 
        self.monopoly += -3*indexDE*(self.refd/datalen)
        
        self.monopoly = self.normalize(self.monopoly,norm_mode='OneMean')
        #OneMean mode for MP, keep mean=1
        return self.monopoly
        
    def firing(self,mx,data,monopoly):   #frrt: fire rate, works on high dimention
        frrt = np.dot(data,mx.T)
        frrt = frrt*monopoly   #use MP
        frrt[frrt<self.relu] = 0   #relu: set unfire rate to 0 
        #do NOT normalize result!!    
        return frrt
        
    def find_bmu(self,frrt):   #best match unit, works even for high dimention 
        bmu = np.argmax(frrt,axis=-1)
        return bmu
    
    def inRN(self,matrix):    #generate random noise mx, the same size as matrix
        ms = np.shape(matrix)
        RNmx = np.random.rand(ms[0],ms[1])
        return RNmx
        
    def bmu_freq(self,bmu,nnodes): #return frequency of bmu in len(bsmx)=nnodes
        s_bmu = np.concatenate((bmu,np.arange(nnodes)))
        uni,s_freq = np.unique(s_bmu,return_counts=True)
        freq = s_freq-1
        return freq
    
    def forward(self,HDdata):   #forward without training, output to next layer
        #HDdata: high dimentional data for forward dataflow
        #self.HDdata = self.normalize(HDdata,'MinMax')
        self.HDdata = self.normalize(HDdata,'UV')

        bsrt = self.firing(self.bsmx,self.HDdata,self.monopoly)
        self.outbmu = self.find_bmu(bsrt)   #output bmu for MP supervised learning
        outdata = self.rt2oh(bsrt)
        #bmu = self.find_bmu(bsrt)
        print('outdata size = ',np.shape(outdata))
        return outdata
    
    def rt2oh(self,mx):
        #one hot bmu*rate array(convert non-max rate unit to 0). Recaculate bmu.
        #works for hi-dim
        mxT = mx.T
        #mxT[mxT == mxT.max(0)] = 1
        mxT[mxT != mxT.max(0)] = 0

        return mxT.T
    
class NN_MP(NN):
    def __init__(self,nnodes,RN,alpha,gamma,CFtrs,capa,seg,stepL,MPsize,MPdia,MPstride,relu,inMD):
        NN.__init__(self,nnodes,RN,alpha,gamma,capa,seg,stepL,relu,inMD)
        self.CFtrs = CFtrs
        #CFtres: tiles select filter tres
        self.MPsize = MPsize
        self.MPdia = MPdia
        #self.label = label
        self.MPstride = MPstride
        #self.relu = relu
        #print(MPstride)
        
    def init_train(self,HDdata):
        #train basal matrix without apical supervise
        #must seg*stepL<len(HDdata)
        
        #creat apmx here, and trained in FORWARD
        bsdata = self.data_prep(HDdata)
        ohrt = NN.init_train(self,bsdata)
        outdata = self.Mpooling(ohrt)

        return outdata
    
    def data_prep(self,HDdata):
        #convert high dimentional data(HDdata) to 1D (no change if 1D)
        #for matrix training
        HD = list(np.shape(HDdata))
        data = HDdata.reshape([np.prod(HD[:-1])]+[HD[-1]])
        data = self.Cfilter(data)
        np.random.shuffle(data)
        #ohlb = np.delete(ohlb,self.zr_index,axis=0)
        return data
    
    def Cfilter(self,data):
        #choose GOOD samples
        #delete samples whose center pixel<CFtres
        datawd = np.shape(data)[-1]
        print('all tiles: ',len(data))
        if datawd%2==0:
            s_center = int(datawd/2 - datawd**(1/2)/2)
        else:
            s_center = int(datawd//2)
        zr_index = np.where(data[:,s_center]<self.CFtrs)
        #print(self.zr_index)
        data = np.delete(data,zr_index,axis=0)
        print('good tiles: ',len(data))
        return data
    
    def Mpooling(self,data):
        #Max Pooling: output all tiles as 1D array, for training or forwarding
        
        out_index,v_noc,h_noc = pl_index(self.MPsize[0],self.MPsize[1],self.MPdia,self.MPstride)
        tile_data = data[...,out_index,:]
        out_data = np.max(tile_data,axis=-2)
        
        #self.bp_ne = np.zeros(data.shape)
        #mxZ = (out_data == out_data.max(axis=-1, keepdims=True)).astype(int)
        #out_data = out_data*mxZ
        return out_data
    
    def bs_train(self,HDdata):
        ohrt = NN.bs_train(self,HDdata)
        outdata = self.Mpooling(ohrt)
        return outdata
    
    def forward(self,HDdata):
        ohrt = NN.forward(self,HDdata)
        outdata = self.Mpooling(ohrt)
        return outdata

class GG(NN):
    #Single Column layer for learning and prediction
    def __init__(self,nnodes,RN,nlb,alpha,beta,gamma,seg,stepL,relu,inMD,redf):
        super().__init__(nnodes,RN,alpha,gamma,seg,stepL,relu,inMD,redf)
        self.nlb = nlb      #nlb: number of diff labels
        self.beta = beta    #beta: apmx learning rate

    
    def init_train(self,HDdata):  #init bsmx, MP & apmx
        outdata = super().init_train(HDdata)
        self.apmx = np.zeros((self.nnodes,self.nlb))
        return outdata
    
    def bs_train(self,HDdata):  #unsupervised training
        outdata = NN.bs_train(self,HDdata)        #print(len(np.unique(bmu)))
        return outdata
    
    def ap_cacu(self,HDdata,label):    #caculate apmx
        #only need label, do not need verify
        outdata = super().forward(HDdata)
        bmu = NN.find_bmu(self,outdata)

        for i in range(len(label)):
            self.apmx[bmu[i],int(label[i])] += outdata[i,bmu[i]]*self.beta
        self.apmx = self.normalize(self.apmx)
        self.pr_label = self.prophet(bmu)

        return outdata
    
    def prophet(self,bmu):
        #choose the strongest prophet
        aplb = np.argmax(self.apmx,axis=-1)
        pr_label = aplb[bmu]
        #aprt = np.max(self.apmx,axis=-1)
        #ap_frrt = aprt[bmu]
        return pr_label #ap_frrt
    
    
    def forward(self,HDdata):    #forward without training
        #HDdata: high dimentional data for forward dataflowß
        #self.HDdata = self.normalize(HDdata,'MinMax')
        outdata = super().forward(HDdata)
        self.bmu = np.argmax(outdata,axis=-1)
        self.pr_label = self.prophet(self.bmu)
        #save prd_label and max_frrt for voting
        #ohrt as max pooling input
        return outdata
    
    def mpap_train(self,HDdata,label):  #BUK train MP and bsmx, supervised 
        ct = 0  #sample counting
        nos = len(HDdata)//self.stepL  #number of steps
        
        for i in range(nos):
            segdata = HDdata[ct:ct+self.stepL]
            seglabel = label[ct:ct+self.stepL]#cut segdata
            ct = ct + self.stepL

            self.SEGmpap_train(segdata,seglabel) #train bsmx
            
            outdata = self.ap_cacu(HDdata,label)  #re-caculate apmx when bsmx changes
        return outdata
    
    def SEGmpap_train(self,bsdata,label): #train MP and bsmx, supervised 
        outdata = self.forward(bsdata)  #get pr_label
        
        Elist = np.equal(self.pr_label,label)+0
        Eindex = np.where(Elist==1)  #index of predict label = label
        UEindex = np.where(Elist==0)
        print('Elist = ',np.shape(Eindex))
        Edata = bsdata[Eindex]   #input data for predict_label = label
        
        self.bsmx = NN.SEGtrain(self,Edata,self.RN)  #train bsmx only with those right data
        
        #UE_bmu = self.bmu[UEindex]
        #self.monopoly = self.SMPtrain(UE_bmu,len(label))   #train MP
        return outdata
        
    def SMPtrain(self,bmu,datalen):
        freq = self.bmu_freq(bmu,self.nnodes)
        self.monopoly += -freq*(self.refd/datalen)
        self.monopoly = self.normalize(self.monopoly,norm_mode='OneMean')
        return self.monopoly

        
        

    

class GG_MP(GG):
    #generative neural gas layer AND Max Pooling layer
    #MERGE 2 layers, because forward learning happens in pooling layer
    
    #nnodes: inital number of nodes in matrix
    #nseeds: inital active (non-zero) number of nodes
    #alpha: bsmx learning rate
    #beta: apmx learning rate
    #gamma: alpha's decreasing rate 
    #CFtres: tiles select filter tres
    #ANtres: init_train's new nnodes born tres
    #seg and stepL: init_train segament, seg*stepL<len(HDdata)
    #MPsize: maxpooling 2D size like [a,b]
    #MPdia: maxpooling tiles diameter
    #MPstride: maxpooling stride
    #relu: node firing treshold
    

    def __init__(self,nnodes,RN,nlb,alpha,beta,gamma,CFtrs,seg,stepL,MPsize,MPdia,MPstride,relu,inMD,label):
        GG.__init__(self,nnodes,RN,nlb,alpha,beta,gamma,CFtrs,seg,stepL,relu,inMD,label)
        self.MPsize = MPsize
        self.MPdia = MPdia
        #self.label = label
        self.MPstride = MPstride
        #self.relu = relu
        #print(MPstride)
        
    def init_train(self,HDdata):
        #train basal matrix without apical supervise
        #must seg*stepL<len(HDdata)
        
        #creat apmx here, and trained in FORWARD

        ohrt = GG.init_train(self,HDdata)
        outdata = self.Mpooling(ohrt)

        return outdata
    
    def train(self,HDdata):
        #forward without training
        #HDdata: high dimentional data for forward dataflow
        #self.HDdata = self.normalize(HDdata,'MinMax')
        ohrt = GG.train(self,HDdata)
        outdata = self.Mpooling(ohrt)

        self.pr_label,self.pr_frrt = self.prophet(outdata)
        #self.pr_label,self.pr_frrt = self.prophet(bsrt)
        
        self.apmx_train(self.label,outdata)
        
        return outdata
    
    def apmx_train(self,label,frrt):
        #apmx train
        #only need label, do not need verify
        for i in range(len(label)):
            for j in frrt[i]:
                self.apmx[:,int(label[i])] += j*self.beta
        self.apmx = self.normalize(self.apmx)
        return self.apmx

    def Mpooling(self,data):
        #Max Pooling: output all tiles as 1D array, for training or forwarding
        
        out_index,v_noc,h_noc = pl_index(self.MPsize[0],self.MPsize[1],self.MPdia,self.MPstride)
        tile_data = data[...,out_index,:]
        out_data = np.max(tile_data,axis=-2)
        
        #self.bp_ne = np.zeros(data.shape)
        #mxZ = (out_data == out_data.max(axis=-1, keepdims=True)).astype(int)
        #out_data = out_data*mxZ
        return out_data
    
    def prophet(self,outdata):
        #choose the strongest prophet (for multicloumn layers)
        clm_apfrrt = np.dot(outdata,self.apmx)
        #print(np.shape(clm_apfrrt))
        #clm_frrt: every column's prediction rate for each label
        clmmax_apfrrt = np.max(clm_apfrrt,axis=-2)
        #clmmax_frrt: strongest predition rate for every label
        #(of all the columns in one layer. U don't need to know which column fires most)
        pr_label = np.argmax(clmmax_apfrrt,axis=-1)
        pr_frrt = np.max(clmmax_apfrrt,axis=-1)
        #strongest preditions of all labels, is the prophet from this layer
        return pr_label,pr_frrt

    def forward(self,HDdata):
        #forward without training
        #HDdata: high dimentional data for forward dataflow
        ohrt = GG.forward(self,HDdata)
        outdata = self.Mpooling(ohrt)

        self.pr_label,self.pr_frrt = self.prophet(outdata)
        return outdata    
    

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






 
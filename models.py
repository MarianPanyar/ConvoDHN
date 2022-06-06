#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 10:07:26 2021

@author: MarianPanya

LEARNING LOOP:
Add layers
init_train
us_train
LLsv_train
LLtest_train
..........


"""

import numpy as np
import importlib

import layers
from layers import flat,oneHot,convolution,padding,NN,NN_MP,bipolar,pl_index

#importlib.reload(GG_MP)
importlib.reload(layers)

class model(object):
    def __init__(self,flow_size,alpha=0.1,beta=0.2,delta=0.2,RN=0,stepL=300,inMD='TR',reSL=1.2,TP=0.3,SPlim=10000):
        #defaut value. u can change them when add layers
        self.alpha=alpha   #bsmx learning rate
        self.beta=beta    #metmx learning rate
        #self.gamma=gamma   #parameter annealing alpha & beta
        self.delta=delta   #supervised/unsupervised ratio
        self.RN=RN  #random noise level
        self.flow_size=flow_size   #outsize of previous layer, like: [300,[28,28],3],output like[300,[5,5],[4,4],3]
        self.stepL = stepL  #step length
        self.inMD = inMD  #init mode
        self.reSL = reSL #metmx learning steps
        self.TP = TP   #firing threshold potential
        self.SPlim = SPlim #samples limint
        self.frame=[]  #frame: a list of layers (class)
        
    def add(self,what,P1=0,P2=0,P3=0,P4=0,P5=0,RN=None,alpha=None,beta=None,stepL=None,inMD=None,reSL=None,TP=None,SPlim=None):
        #add a layer to frame 
        if RN is None:
            RN = self.RN
        if alpha is None:
            alpha = self.alpha
        if beta is None:
            beta = self.beta
        if stepL is None:
            stepL = self.stepL
        if inMD is None:
            inMD = self.inMD
        if reSL is None:
            reSL = self.reSL
        if TP is None:
            TP = self.TP
        if SPlim is None:
            SPlim = self.SPlim
        
        if what=='Padding':
            layer = padding(self.flow_size[-1],P1,P2)
            #self.flow_size[-1]=csize: size of mx for padding(last dimensions only)
            #P1=margin: margin size
            #P2=mode: padding mode, 'constant'=zero padding
            self.flow_size[-1] = [x+P1*2 for x in self.flow_size[-1]]
            print('Padding:',self.flow_size,',',P2,'Margin =',P1)  
            
        elif what=='oneHot':
            layer = oneHot(P1)  #P1=mode
            print('OneHot: ',P1)
            
        elif what=='Conv2D':  #convolution 2D
            size_in = self.flow_size[-P3]  #input 2D size
            #print(self.flow_size,P3)
            layer = convolution(size_in,P1,P2,P3)
            #P1 = diameter,P2=stride=1,P3=W2con(where to conv,1 = last layer)
            out_index,v_noc,h_noc = pl_index(size_in[0],size_in[1],P1,P2)
            self.flow_size[-P3]=[v_noc,h_noc]   #output 2D size
            self.flow_size.insert(len(self.flow_size)-P3+1,[P1,P1])
            print('Conv2D Layer:',self.flow_size,', Conv on the last',P3,'layer')

        elif what=='Flat':
            layer = flat(P1,P2)  #reshape 2 or more dimensionss into one
            #P1=W2: which dimension to flat,(the lower one)
            #P2=FD: flat dimensions, how many dims want to flatten(mostly 2)
            f_size = 1 #f_size: flated output size
            for i in range(P2):
                d_size = self.flow_size[len(self.flow_size)-P1] #shape of dimension to flat
                self.flow_size.pop(len(self.flow_size)-P1)  #delete it
                f_size = f_size*np.prod(d_size)           #use its product
            self.flow_size.insert(len(self.flow_size)-P1+1,f_size)    #put it back
            print('Flatten Layer: ',self.flow_size,', Flat',P2,'Dims from Dim',P1)
        
        elif what=='NN': #single-column layer
            layer = NN(P1,RN,alpha,beta,stepL,inMD,reSL,TP)   #P1=nnodes
            self.flow_size[-1] = P1
            print('NN: ',self.flow_size, 'Matrix size = ',P1)
            
        elif what=='NN_MP':  #multi-column layer
            MPsize = self.flow_size[-2]
            layer = NN_MP(P1,MPsize,P2,P3,RN,alpha,beta,stepL,inMD,reSL,TP,SPlim)
            #P1=nnodes,P2=MPdia,P3=MPstride

            out_index,v_noc,h_noc = pl_index(MPsize[0],MPsize[1],P2,P3) #vertical & horizontal size
            self.flow_size[-1] = P1             
            self.flow_size[-2]=[v_noc,h_noc]
            print('NN and MaxPooling: ',self.flow_size, 'Matrix size = ',P1)
            
        elif what=='Bipolar':  #enhance contract
            #P1 = radius, P2 = beth, P3 = mode
            layer = bipolar(self.flow_size[-1],P1,P2,P3)
            
        
        self.frame.append(layer)
        del layer
        return self.frame
    
    def init_train(self,indata):  #unsuperviesd train each layer one by one
        for i in self.frame:
            outdata = i.init_train(indata)
            indata = outdata
            print(type(i).__name__,np.shape(outdata))      
        return outdata
    
    def us_train(self,indata):   #unsupervised train after init
        for i in self.frame:
            outdata = i.us_train(indata)
            indata = outdata   #data flow only 
        return outdata
    
    def forward(self,indata): #pure forward without training or prediction
        for i in self.frame:
            outdata = i.forward(indata)
            indata = outdata
        return outdata
    
    def LLforward(self,indata,LLapmx):  #forward with prediction
        outdata = self.forward(indata)
        bmu = self.find_bmu(outdata)
        pr_label = self.LLprophet(bmu,LLapmx)
        return pr_label
    
    def LLsv_train(self,indata,label,nlb,stepL=None,delta=None):
        if stepL is None:
            stepL = self.stepL
        if delta is None:
            delta = self.delta
        indata = self.normalize(indata,'UV')
        self.LLsv_BUKtrain(indata,label,nlb,stepL,delta)
        return self.frame

    
    def LLsv_BUKtrain(self,indata,label,nlb,stepL,delta):  #last-layer-prediction supervised train.
        #last layer has apmx and only ONE column 

        ct = 0  #sample counting
        nos = len(indata)//stepL  #number of steps      
        
        for i in range(nos):
            trRN = self.RN*(1-i/nos)  #RN decrease every step, Simulated Annealing
            beta = self.beta*(1-i/nos)  #metmx learning rate, SA
            alpha = self.alpha*(1-i/nos)  
            
            segdata = indata[ct:ct+stepL]
            seglabel = label[ct:ct+stepL] #cut segdata
            ct = ct + stepL
            
            self.LLsv_SEGtrain(segdata,seglabel,nlb,trRN,alpha,beta,delta)

        return self.frame  #whatever(no apmx output needed. Generate apmx everytime.)
    
    def LLsv_SEGtrain(self,data,label,nlb,trRN,alpha,beta,delta):
        
        frRT = self.forward(data)
        LLapmx = self.LLapmxTrain(frRT,label,self.frame[-1].nnodes,nlb)  #generate/train apmx for LL
        LLbmu = self.find_bmu(frRT)
        pr_label = self.LLprophet(LLbmu,LLapmx)
        Elist = self.verf(pr_label,label)   #Elist is what u need for SmetTrain     
        U_data = data[np.where(Elist==0)] 
        
        Ldata = data  #layer input
        U_Ldata = U_data
        # *L*: bmu or data for certain layer (unlike segElist is for all layers)
        for i in self.frame:
            if isinstance(i,NN):
                Ldata = i.normalize(Ldata)
                U_Ldata = i.normalize(U_Ldata)
                    #segLbmu = self.find_bmu(segLoutdata)
                if isinstance(i,NN_MP):
                    bsLdataC = i.MPin(Ldata)
                    bsfrRT = i.firing(i.bsmx,bsLdataC,i.metmx)
                    U_bsLdataC = i.MPin(U_Ldata)
                    bsUE_frRT = i.firing(i.bsmx,U_bsLdataC,i.metmx)

                    MfrRT,Mdata = i.TPfilter(bsfrRT,bsLdataC,i.TP)
                    MU_frRT,MU_data = i.TPfilter(bsUE_frRT,U_bsLdataC,i.TP)

                    i.bsmx = i.BSMXtrain(MfrRT,Mdata,trRN,alpha)
                    i.metmx = i.SmetTrain(i.metmx,MfrRT,MU_frRT,delta,beta)
                else:
                    LfrRT = i.firing(i.bsmx,Ldata,i.metmx)
                    LU_frRT = i.firing(i.bsmx,U_Ldata,i.metmx)
                    i.bsmx = i.BSMXtrain(LfrRT,Ldata,trRN,alpha)
                    i.metmx = i.SmetTrain(i.metmx,LfrRT,LU_frRT,delta,beta)
            Loutdata = i.forward(Ldata)
            U_Loutdata = i.forward(U_Ldata)
            Ldata = Loutdata
            U_Ldata = U_Loutdata
        return Loutdata
    
    def LLtestTrain(self,train_data,train_label,test_data,test_label,nlb): #forward, no training, last-layer-prediction
        train_outdata = self.forward(train_data)
        LLapmx = self.LLapmxTrain(train_outdata,train_label,self.frame[-1].nnodes,nlb)
        
        test_outdata = self.forward(test_data)
        test_bmu = self.find_bmu(test_outdata)
        testP_label = self.LLprophet(test_bmu,LLapmx) #use last layer to predict 
        
        self.verf(testP_label,test_label)
        return LLapmx

    
    def verf(self,p_label,label):   #verify prediction label       
        Elist = np.equal(p_label,label)+0    #1 if equal,0 if unequal
        datalen = len(label)
        #nlb = max(label)+1
        print('score: ',np.sum(Elist)/datalen)     #prediction score
        return Elist
    
    def LLapmxTrain(self,frRT,label,nnodes,nlb):  #train/creat apmx(renew every time)
        LLapmx = np.zeros((nnodes,nlb))  #make new apmx 
        bmu = self.find_bmu(frRT)
        for i in range(len(label)):
            LLapmx[bmu[i],int(label[i])] += frRT[i,bmu[i]]    #add pRTs to certain apmx       
        LLapmx = self.normalize(LLapmx)
        return LLapmx
    
    def find_bmu(self,frrt):   #best match unit, works even for high dimensions 
        bmu = np.argmax(frrt,axis=-1)
        return bmu
    
    def LLprophet(self,bmu,apmx):  #predict label
        aplb = np.argmax(apmx,axis=-1)  #only care the max of apmx
        pr_label = aplb[bmu]
        return pr_label 
    
#    def LLprophet(self,outdata,apmx): #predict label for last layer(only one column)
#        OHoutdata = self.rt2oh(outdata)  #one hot
#        pLabel = np.argmax(pRT2D,axis=-1)   #predict label
#        pRT = np.max(pRT2D,axis=-1)    #prediction rate(how strong the prediction is)
#        pBMU = np.argmax(outdata,axis=-1)   #which node is using when making prediction
#        return pLabel,pRT,pBMU
        
    def ALprophet(self,outdata,apmx):  #(all layers) predict label and how strong the prediction is
        OHoutdata = self.rt2oh(outdata)  #one hot
        pRT3D = np.dot(OHoutdata,apmx)  #fire-rate of every column for every label
        
        pRT = np.max(pRT3D,axis=(-1,-2))   #predict rate, how strong every prediction is
        COLlist = np.argmax(np.max(pRT3D,axis=-1),axis=-1)   #which column has the strongest prediction

        COLpLabel = np.argmax(pRT3D,axis=-1)  #every column's predict label
        pLabel = COLpLabel[np.arange(len(COLlist)),COLlist]  
        
        COLbmu = np.argmax(OHoutdata,axis=-1)  #every column's bmu
        pBMU = COLbmu[np.arange(len(COLlist)),COLlist]  #strongest bmu who makes prediction
        return pLabel,pRT,pBMU
        
    def rt2oh(self,mx):    #one hot bmu*rate array(convert non-max rate unit to 0). 
        #works for hi-dim
        mxT = mx.T+0
        #mxT[mxT == mxT.max(0)] = 1
        mxT[mxT != mxT.max(0)] = 0
        return mxT.T
    
    def vote(self,lr_label,lr_frrt):
        #vote to decide predic label
        ele_layer = np.argmax(lr_frrt,axis=0)
        print(ele_layer)
        #ele_layer = np.array(ele_layer)
        #ele = ele_layer.astype(int)
        pr_label = lr_label[ele_layer]
        return pr_label
    
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
                
            







          


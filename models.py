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
from layers import flat,oneHot,convolution,padding,NN,NN_MP,bipolar,pl_index,Norm

#importlib.reload(GG_MP)
importlib.reload(layers)

class model(object):
    def __init__(self,flow_size,alpha=0.1,beta=0.2,gamma=0.1,delta=0.2,ep=0.2,RN=0,stepL=300,inMD='TR',reSL=1.2,TP=0.3,SPlim=10000):
        #defaut value. u can change them when add layers
        self.alpha=alpha   #bsmx learning rate
        self.beta=beta    #metmx learning rate
        self.gamma=gamma   #parameter annealing alpha & beta
        self.delta=delta   #supervised/unsupervised ratio
        self.ep=ep
        self.RN=RN  #random noise level
        self.flow_size=flow_size   #outsize of previous layer, like: [300,[28,28],3],output like[300,[5,5],[4,4],3]
        self.stepL = stepL  #step length
        self.inMD = inMD  #init mode
        self.reSL = reSL #metmx learning steps
        self.TP = TP   #firing threshold potential
        self.SPlim = SPlim #samples limint
        self.frame=[]  #frame: a list of layers (class)
        
    def add(self,what,P1=0,P2=0,P3=0,P4=0,P5=0,RN=None,alpha=None,beta=None,gamma=None,delta=None,ep=None,stepL=None,inMD=None,reSL=None,TP=None,SPlim=None):
        #add a layer to frame 
        if RN is None:
            RN = self.RN
        if alpha is None:
            alpha = self.alpha
        if beta is None:
            beta = self.beta
        if gamma is None:
            gamma = self.gamma
        if delta is None:
            delta = self.delta
        if ep is None:
            ep = self.ep
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
            layer = NN(P1,RN,alpha,beta,gamma,delta,ep,stepL,inMD,reSL)   #P1=nnodes
            self.flow_size[-1] = P1
            print('NN: ',self.flow_size, 'Matrix size = ',P1)
            
        elif what=='NNmp':  #multi-column layer
            MPsize = self.flow_size[-2]
            layer = NN_MP(P1,MPsize,P2,P3,RN,alpha,beta,gamma,delta,ep,stepL,inMD,reSL,TP,SPlim)
            #P1=nnodes,P2=MPdia,P3=MPstride

            out_index,v_noc,h_noc = pl_index(MPsize[0],MPsize[1],P2,P3) #vertical & horizontal size
            self.flow_size[-1] = P1             
            self.flow_size[-2]=[v_noc,h_noc]
            print('NN and MaxPooling: ',self.flow_size, 'Matrix size = ',P1)
            
        elif what=='Bipolar':  #enhance contract
            #P1 = radius, P2 = beth, P3 = mode
            layer = bipolar(self.flow_size[-1],P1,P2,P3)
            
        elif what=='Norm':
            layer = Norm(P1)  #P1 = mode
            
        self.frame.append(layer)
        del layer
        return self.frame
    
    def init_train(self,indata):  #unsuperviesd train each layer one by one
        Poutdata = []
        for i in self.frame:
            outdata = i.init_train(indata)
            indata = outdata
            Poutdata.append(outdata)
            print(type(i).__name__,np.shape(outdata))      
        return Poutdata
    
    def us_train(self,indata,Tlist):   #unsupervised train after init
        #Tlist:training cirles for each layer, like [1,1,1,3]
        ct=0
        for i in self.frame:
            for j in range(Tlist[ct]):
                outdata = i.us_train(indata)
            indata = outdata   #data flow only 
            ct=ct+1
        return outdata
    
    def forward(self,indata): #pure forward without training or prediction
        for i in self.frame:
            outdata = i.forward(indata)
            indata = outdata
        return outdata
    
    def LLforward(self,indata,LLapmx):  #forward with prediction
        outdata = self.forward(indata)
        bmu = self.find_bmu(outdata)
        pr_label,pr_it = self.LLprophet(bmu,LLapmx)
        return pr_label
    
    def Mforward(self,indata):  #multi output, return output of all layers in a list
        Moutdata = []
        for i in self.frame:
            outdata = i.forward(indata)
            indata = outdata
            Moutdata.append(outdata)
        return Moutdata
    
    def LLsv_train(self,indata,label,nlb,stepL=None):
        if stepL is None:   
            stepL = self.stepL
            
        indata = self.normalize(indata,'UV')
        self.LLsv_BUKtrain(indata,label,nlb,stepL)
        
        return self.frame

    
    def LLsv_BUKtrain(self,indata,label,nlb,stepL):  #last-layer-prediction supervised train
        #last layer has apmx and only ONE column 
        ct = 0  #sample counting
        nos = len(indata)//stepL  #number of steps      
        
        for i in range(nos):

            Anne = 1-i/nos  #Para decrease every step, Simulated Annealing
            trRN = self.RN*Anne
            
            segdata = indata[ct:ct+stepL]
            seglabel = label[ct:ct+stepL] #cut segdata
            ct = ct + stepL
            
            self.LLsv_SEGtrain(segdata,seglabel,nlb,trRN,Anne)

        return self.frame  #whatever(no apmx output needed. Generate apmx everytime.)
    
    def LLsv_SEGtrain(self,indata,label,nlb,trRN,Anne):      
        Moutdata = self.Mforward(indata)  #output of all layers as a list
        
        LLoutdata = Moutdata[-1]  #last layer outdata
        LLapmx = self.LLapmxTrain(LLoutdata,label,self.frame[-1].nnodes,nlb)  #generate/train apmx for LL(last layer)
        LLbmu = self.find_bmu(LLoutdata)   
        pr_label,pr_it = self.LLprophet(LLbmu,LLapmx)   #predict labels
        Elist = self.verf(pr_label,label)   #prediction 0-1 list 
        

        Lindata = indata  #this layer input data. first layer input data is indata
        #U_Lindata = indata[np.where(Elist==0)]
        E_Lindata = indata[np.where(Elist==1)]
        
        ct = 0
        
        for i in self.frame:
            # *L*: bmu or data for certain layer (unlike segElist is for all layers)
            Loutdata = Moutdata[ct] #this layer output. NO normalization for TPfilter
            U_Loutdata = Loutdata[np.where(Elist==0)]
            E_Loutdata = Loutdata[np.where(Elist==1)]
            
            if isinstance(i,NN):
                    #segLbmu = self.find_bmu(segLoutdata)
                if isinstance(i,NN_MP):            
                    bLoutdata = i.left_flat(Loutdata)    #all frRT and fired input
                    FbLoutdata = i.EPfilter(bLoutdata,bLoutdata,i.ep)
                    FbLoutdata = FbLoutdata[:i.SPlim]
                    
                    U_bLoutdata = i.left_flat(U_Loutdata)    #all frRT and fired input
                    U_FbLoutdata = i.EPfilter(U_bLoutdata,U_bLoutdata,i.ep)
                    
                    U_aLoutdata = np.tile(U_FbLoutdata,(i.delta,1))
                    metLoutdata = np.concatenate((FbLoutdata,U_aLoutdata),axis=0)
                    np.random.shuffle(metLoutdata)
                    i.metmx = i.metTrain(i.metmx,metLoutdata,i.beta*Anne)
                    
                    E_Lindata = i.bsPre(E_Lindata,i.TP)
                    E_bLindata = i.left_flat(E_Lindata)   #flated
                    print(E_bLindata[50:80])
                    #E_FbLindata = i.TPfilter(E_bLindata,E_bLindata,i.TP)
                    #E_FbLindata = E_FbLindata[:i.SPlim]
                    E_bLindata = i.normalize(E_bLindata)
                    #E_bLoutdata = i.firing(i.bsmx,E_FbLindata,i.metmx)    #all frRT and fired input
                    #E_FbLindata = i.EPfilter(E_FbLindata,E_bLoutdata,i.ep)                                        
                    #i.metmx = i.metTrain(i.metmx,FbLoutdata,i.beta*Anne)  #All outdata for metTrain
                    #i.metmx = i.metTrain(i.metmx,U_FbLoutdata,i.delta*Anne)  #Unpredicted outdata for metTrain
                    #i.bsmx = i.BSMXtrain(sDcLoutdata,sLindata,trRN,i.ep*Anne)   #Decaped outdata and all indata for BSMXtrain
                    i.bsmx = i.BSMXtrain(E_bLindata,trRN,i.alpha*Anne)  #All indata and outdata for BSMXtrain
                    #i.bsmx = i.BSMXtrain(U_sLoutdata,U_sLindata,trRN,i.gamma*Anne)  #Unpredicted indata and outdata for BSMXtrain
                    
                else:   #train NN layer
                    
                    
                    U_aLoutdata = np.tile(U_Loutdata,(i.delta,1))
                    metLoutdata = np.concatenate((Loutdata,U_aLoutdata),axis=0)
                    np.random.shuffle(metLoutdata)                    
                    i.metmx = i.metTrain(i.metmx,metLoutdata,i.beta*Anne)
                    #i.metmx = i.metTrain(i.metmx,U_Loutdata,i.delta*Anne)   #metmx train first
                    dcLoutdata = self.decap1D(Elist,Loutdata)
                    Lindata = i.normalize(Lindata)
                    i.bsmx = i.BSMXtrain(Lindata,trRN,i.alpha*Anne,dcLoutdata)
                    #i.bsmx = i.BSMXtrain(E_Lindata,trRN,i.alpha*Anne)                    
                    #i.bsmx = i.BSMXtrain(U_Loutdata,U_Lindata,trRN,i.gamma*Anne)
            Lindata = Loutdata
            #U_Lindata = U_Loutdata
            E_Lindata = E_Loutdata
            ct=ct+1
        return Loutdata

    def decap1D(self,Elist,frRT):   #delete bmu fire rate when unpredicted, leave change to 2nd bmu
        Uindex = np.where(Elist==0)
        dcfrRT = frRT+0
        dcfrRT[Uindex,dcfrRT[Uindex].argmax(-1)]=0
        return dcfrRT
    
    def decap2D(self,Elist,frRT):  #for 2d
        Uindex = np.where(Elist==0)[0]
        dcfrRT = frRT+0
        rUindex = np.repeat(Uindex,dcfrRT.shape[-2])
        d2 = np.tile(np.arange(dcfrRT.shape[-2]),len(Uindex))
        dcfrRT[rUindex,d2,dcfrRT[rUindex,d2].argmax(-1)]=0
        return dcfrRT

    def LLtestTrain(self,train_data,train_label,test_data,test_label,nlb): #forward, no training, last-layer-prediction
        train_outdata = self.forward(train_data)
        LLapmx = self.LLapmxTrain(train_outdata,train_label,self.frame[-1].nnodes,nlb)
        
        test_outdata = self.forward(test_data)
        test_bmu = self.find_bmu(test_outdata)
        testP_label,testP_it = self.LLprophet(test_bmu,LLapmx) #use last layer to predict 
        
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
        apit = np.max(apmx,axis=-1)
        pr_it = apit[bmu]   #how strong the prediction is
        return pr_label,pr_it 
    
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
                
            







          


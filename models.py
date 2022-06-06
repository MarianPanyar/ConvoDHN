#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 10:07:26 2021

@author: MarianPanyar

LEARNING LOOP:
Add layers
init_train
us_train

..........


"""

import numpy as np
import importlib

import layers
from layers import flat,oneHot,padding,NN,CNP,salience,norm

#importlib.reload(GG_MP)
importlib.reload(layers)

class model(object):
    def __init__(self,flow_size):
        self.flow_size=flow_size   #outsize of previous layer, like: [300,[28,28],3],output like[300,[5,5],[4,4],3]
        self.frame=[]  #frame: a list of layers (class)
        
    def add(self,what,P1=0,P2=0,P3=0,P4=0,P5=0,P6=0,P7=0,P8=0,P9=0,RN=0.000001,alpha=1,beta=5,gamma=1.5,delta=0.3,ep=0,stepL=500,inMD='sample',reSL=1.3,TP=0,SPlim=100000):
        #add a layer to frame 
        
        if what=='padding':
            layer = padding(self.flow_size[-1],P1,P2)
            #self.flow_size[-1]=csize: size of mx for padding(last dimensions only)
            #P1=margin: margin size
            #P2=mode: padding mode, 'constant'=zero padding
            self.flow_size[-1] = [x+P1*2 for x in self.flow_size[-1]]
            print('Padding:',self.flow_size,',',P2,'Margin =',P1)  
            
        elif what=='oneHot':
            layer = oneHot(P1,P2)  #P1=mode
            print('OneHot: keep top ',P1,' values, Mode ',P2)
            
        elif what=='flat':
            layer = flat(P1,P2)  #reshape 2 or more dimensionss into one
            #P1=W2: which dimension to flat,(the lower one)
            #P2=FD: flat dimensions, how many dims want to flatten(mostly 2)
            f_size = 1 #f_size: flated output size
            for i in range(P2):
                d_size = self.flow_size[len(self.flow_size)-P1] #shape of dimension to flat
                self.flow_size.pop(len(self.flow_size)-P1)  #delete it
                f_size = f_size*np.prod(d_size)           #use its product
            self.flow_size.insert(len(self.flow_size)-P1+1,f_size)    #put it back
            print('Flatten: ',self.flow_size,', Flat',P2,'Dims from Dim',P1)
        
        elif what=='NN': #single-column layer
            layer = NN(P1,RN,alpha,beta,gamma,delta,ep,stepL,inMD,reSL)   #P1=nnodes
            self.flow_size[-1] = P1
            print('NN: ',self.flow_size, 'Matrix size = ',P1)
            
        elif what=='CNP':  #convolution, neural and Maxpooling layer
            Csize = self.flow_size[-P4]  #dimention to be CV
            out_index,v_noc,h_noc = self.pl_index(Csize[0],Csize[1],P2,P3)
            self.flow_size[-P4]=[v_noc,h_noc]   #CV output 2D shape
            self.flow_size.insert(len(self.flow_size)-P4+1,[P2,P2])
            print('Conv2D: ',self.flow_size,', Conv on the last',P4,'layer')
            if P4 == 2:
                self.flow_size.pop(-P4)

            layer = CNP(P1,Csize,P2,P3,P4,P5,P6,RN,alpha,beta,gamma,delta,ep,stepL,inMD,reSL,TP,SPlim)
            #P1=nnodes,P2=CVdia,P3=CVstride,P4=CVw2,P5=MPdia,P6=MPstride
            MPsize = self.flow_size[-2]
            out_index,v_noc,h_noc = self.pl_index(MPsize[0],MPsize[1],P5,P6) #vertical & horizontal size
            self.flow_size[-1] = P1             
            self.flow_size[-2]=[v_noc,h_noc]
            print('NN and MaxPooling: ',self.flow_size, 'Matrix size = ',P1)

            
        elif what=='salience':
            layer = salience(P1)  #P1 = TP
            print('Salience: TP =',P1)
            
        elif what=='norm':
            layer = norm(P1)   #P1 = mode
            print('Normalization: ',P1)
            

        self.frame.append(layer)
        del layer
        return self.frame
    
    
    def init_train(self,indata,label,nlb):  #unsuperviesd train each layer one by one
        #import timeit
        for i in self.frame:
            #start = timeit.default_timer()

            outdata = i.init_train(indata,label,nlb)
            indata = outdata
            print(type(i).__name__,np.shape(outdata))  
            
            #stop = timeit.default_timer()
           #print('Time: ', stop - start) 
        return outdata

    
    def us_train(self,indata,label,nlb,Tlist):   #unsupervised train after init
        #Tlist:training cirles for each layer, like [1,1,1,3]
        ct=0
        for i in self.frame:
            for j in range(Tlist[ct]):
                outdata = i.us_train(indata,label,nlb)
            indata = outdata   #data flow only 
            ct=ct+1
        #outdata = self.forward(indata)
        return outdata
    
    def si_train(self,indata,label,nlb):
        for i in self.frame:
            #start = timeit.default_timer()
            outdata = i.init_train(indata,label,nlb)
            
            if type(i)==NN:
                for j in range(i.ep):  #if ep>0
                    outdata = i.us_train(indata,label,nlb)
                    print('us_train')
            indata = outdata
            print(type(i).__name__,np.shape(outdata))  
            
            #stop = timeit.default_timer()
           #print('Time: ', stop - start) 
        return outdata
    
    def So_train(self,indata,label,tdata,tlabel,nlb,nloops,Esp):  #Sol supervised training
        Adata = indata+0    
        Alabel = label+0
        
        #i.init_train(Adata,Alabel,nlb)

        for j in range(nloops):
            #self.frame[-1].delta=(delta2-delta1)*(j/(nloops-1))+delta1
            Aoutdata = self.si_train(Adata,Alabel,nlb)
            #self.us_train(Adata,Alabel,nlb,Tlist)
            
            #Aoutdata = self.forward(Adata)
            B2L = self.find_B2L(self.frame[-1].bsmx,nlb)
            p_label,Elist = self.So_test(Aoutdata,Alabel,B2L)
            print(sum(Elist))
            
            Sodata = Adata[np.where(Elist==0)]
            Solabel = Alabel[np.where(Elist==0)]
            Sodata,Solabel = self.piano(Sodata,Solabel,Esp)
            print(len(Sodata))
            #Adata = Adata[np.where(Elist==1)]
            #Alabel = Alabel[np.where(Elist==1)]
            Adata = np.concatenate((Adata,Sodata),axis=0)
            Alabel = np.concatenate((Alabel,Solabel),axis=0)
            print(len(Adata))
            
            B2L = self.find_B2L(self.frame[-1].bsmx,10)
            outdata = self.TBUKforward(tdata,self.frame[-1].stepL)
            self.So_test(outdata,tlabel,B2L)
            
            #Adata,Alabel = self.S_shuffle(Adata,Alabel)
            #shuffle in BUKtrain
        return Elist
    
    def Lus_train(self,indata,label,tdata,tlabel,nlb,LstepL):  #last layer train 
        #LstepL: long step length, longer than stepL
        ct = 0  #sample counting
        nos = len(indata)//LstepL  #number of steps
        print('steps = ',nos)
        
        for i in range(nos):
            #trRN = self.RN*(1-i/nos)  #RN decrease every step, Simulated Annealing
            #alpha = self.alpha*(1-i/nos)
            segdata = indata[ct:ct+LstepL] #cut segdata
            seglabel = label[ct:ct+LstepL]
            
            #reSL = (self.beta-self.gamma)*(i/(nos-1))+self.gamma
            
            self.segLus_train(segdata,seglabel,nlb) #train bsmx  
            ct = ct + LstepL
            
            B2L = self.find_B2L(self.frame[-1].bsmx,10)
            outdata = self.TBUKforward(tdata,self.frame[-1].stepL)
            p_label,Elist = self.So_test(outdata,tlabel,B2L)
        return Elist
        
    
    def segLus_train(self,indata,label,nlb):
        c=1
        for i in self.frame:
            outdata = i.us_train(indata,label,nlb)
            indata = outdata
            c+=1                
        return outdata
            

    def SoL_train(self,indata,label,tdata,tlabel,nlb,nloops,Esp):  #Sol supervised training
        
        Sdatalen = len(indata)//3
        Sdata = indata[:(Sdatalen*3)].reshape(3,Sdatalen,indata.shape[-1])
        Slabel = label[:(Sdatalen*3)].reshape(3,Sdatalen)        
        #i.init_train(Adata,Alabel,nlb)
        Sodata = np.array([], dtype=np.int64).reshape(0,indata.shape[-1])
        Solabel = np.array([], dtype=np.int64).reshape(0)
        for j in range(nloops):
            #self.frame[-1].delta=(delta2-delta1)*(j/(nloops-1))+delta1
            dataA = Sdata[(j%3)]
            labelA = Slabel[(j%3)]
            #dataB = Sdata[((j+1)%3)]
            #labelB = Slabel[((j+1)%3)]
            dataC = Sdata[((j+2)%3)]
            labelC = Slabel[((j+2)%3)]
            
            Adata = np.concatenate((dataA,Sodata),axis=0)
            Alabel = np.concatenate((labelA,Solabel),axis=0)
            print(len(Adata))
            
            self.si_train(Adata,Alabel,nlb)
            #self.us_train(Adata,Alabel,nlb,Tlist)
            
            outdataC = self.forward(dataC)
            B2L = self.find_B2L(self.frame[-1].bsmx,nlb)
            p_label,Elist = self.So_test(outdataC,labelC,B2L)
            print(sum(Elist))
            
            Sodata = dataC[np.where(Elist==0)]
            Solabel = labelC[np.where(Elist==0)]
            Sodata,Solabel = self.piano(Sodata,Solabel,Esp)
            print(len(Sodata))
            #Adata = Adata[np.where(Elist==1)]
            #Alabel = Alabel[np.where(Elist==1)]

            
            B2L = self.find_B2L(self.frame[-1].bsmx,10)
            outdata = self.TBUKforward(tdata,self.frame[-1].stepL)
            self.So_test(outdata,tlabel,B2L)
            
            #Adata,Alabel = self.S_shuffle(Adata,Alabel)
            #shuffle in BUKtrain
        return Elist
    
    def find_B2L(self,bsmx,nlb):  #final NN's lower half = label
        tailmx = bsmx[:,-nlb:]    #tail = onthot label
        B2L = np.argmax(tailmx,axis=-1)    #label for each node in bsmx
        return B2L
    
    def piano(self,data,label,Esp):
        if Esp>=1:
            data = np.repeat(data,Esp,axis=0)
            label = np.repeat(label,Esp,axis=0)
        else:
            index = np.random.choice(len(data),int(len(data)*Esp),replace=False)
            data = data[index]
            label = label[index]
        return data,label
            
            
    def So_test(self,outdata,label,B2L):  #use output predict label
        bmu = self.find_bmu(outdata)
        p_label = B2L[bmu]
        Elist = self.verf(p_label,label)
        return p_label,Elist
        
            
    def S_shuffle(self,arrayA,arrayB):   #
        Sindex = np.random.permutation(len(arrayA))
        newA = arrayA[Sindex]
        newB = arrayB[Sindex]
        return newA,newB         


    
    def forward(self,indata): #pure forward without training or prediction
        for i in self.frame:
            outdata = i.forward(indata)
            indata = outdata
        return outdata
    
    def TBUKforward(self,indata,stepL): #forward for testing
        ct = 0
        nos = len(indata)//stepL
        rem = len(indata)%stepL
        
        if rem > 0:
            firstL = rem
        else:
            firstL = stepL
            nos = nos-1
        firstindata = indata[:firstL]
        
        for j in self.frame:
            firstoutdata = j.Tforward(firstindata)
            firstindata = firstoutdata
        outdata = firstoutdata
                            
        for i in range(nos):
            segindata = indata[firstL+ct:firstL+ct+stepL] #cut segdata
            ct = ct+stepL
            
            for j in self.frame:
                segoutdata = j.Tforward(segindata)
                segindata = segoutdata
            outdata = np.concatenate((outdata,segoutdata),axis=0)
                
        return outdata

    
    def verf(self,p_label,label):   #verify prediction label       
        Elist = np.equal(p_label,label)+0    #1 if equal,0 if unequal
        datalen = len(label)
        #nlb = max(label)+1
        print('score: ',np.sum(Elist)/datalen)     #prediction score
        return Elist

    
    def find_bmu(self,frrt):   #best match unit, works even for high dimensions 
        bmu = np.argmax(frrt,axis=-1)
        return bmu

        


    
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







          


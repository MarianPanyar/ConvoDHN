#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 10:07:26 2021

@author: MarianPanya

LEARNING LOOP:
init_train
train * n
ap_train
forward - p_label - Elist - newdata
train
ap_train
..........


"""

import numpy as np
import importlib

import layers
from layers import flat,padding,NN,NN_MP,GG,GG_MP,bipolar,pl_index
#import operator

#importlib.reload(GG_MP)
importlib.reload(layers)

class model(object):
    def __init__(self,nlb,alpha,beta,gamma,flow_size,label,glrf):
        #nlb: number of labels, for classification
        #alpha: forward learning rate
        #beta: backward learning rate
        #flow_size: data-flow size of new adding layers
        #frame: a list of layers (class)
        self.nlb=nlb
        self.alpha=alpha
        self.beta=beta
        self.gamma = gamma
        self.flow_size=flow_size
        self.label=label
        self.glrf = glrf
        self.frame=[]
        
    def add(self,what,P1=0,P2=0,P3=0,P4=0,P5=0,P6=0,P7=0,P8=0,P9=0,P10=0,P11=0):
        #construct, layer by layer 
        
        if what=='Padding':
            layer = padding(self.flow_size[-1],P1,P2)
            #P1: margin
            #P2: padding mode, 'constant'=zero padding
            self.flow_size[-1] = [x+P1*2 for x in self.flow_size[-1]]
            print('Padding:',self.flow_size,',',P2,'Margin=',P1)            
        
        elif what=='Flat':
            layer = flat(P1,P2)
            #reshape 2 or more dimentions into one
            #P1: which dimention to flat, the lower one
            #P2: flat dimention, FD, how many dims want to flatten(mostly 2)
            #f_size: flated output size
            f_size = 1
            for i in range(P2):
                d_size = self.flow_size[len(self.flow_size)-P1]
                self.flow_size.pop(len(self.flow_size)-P1)
                if np.ndim(d_size) > 0:
                    f_size = f_size*np.prod(d_size)
                elif np.ndim(d_size) == 0:
                    f_size = f_size*d_size
                    
            self.flow_size.insert(len(self.flow_size)-P1+1,f_size)
            print('Flatten Layer: ',self.flow_size,', Flat',P2,'dims from Dim',P1)
        
        elif what=='NN':
            layer = NN(P1,P2,self.alpha,self.gamma,P3,P4,P5,P6,P7)
            #P1=nnodes,P2=RN,P3=capa,P4=stepL,P5=relu,P6=inMD,P7=refd
            self.flow_size[-1] = P1
            print('NN: ',self.flow_size, 'Matrix size = ',P1)
            
        elif what=='NN_MP':
            #main learning layer
            MPsize = self.flow_size[-2]
            layer = NN_MP(P1,P2,self.alpha,self.gamma,P3,P4,P5,MPsize,P6,P7,P8,P9,P10)
            #P1=nnodes,P2=RN,P3=CFtres,P4=capa,P5=stepL,P6=MPdia,P7=MPstride,P8=relu,P9=inMD,P10=refd

            out_index,v_noc,h_noc = pl_index(MPsize[0],MPsize[1],P7,P8) #here
            self.flow_size[-1] = P1             
            self.flow_size[-2]=[v_noc,h_noc]
            print('NN and MaxPooling: ',self.flow_size, 'Matrix size = ',P1)
            
        elif what=='GG':
            layer = GG(P1,P2,self.nlb,self.alpha,self.beta,self.gamma,P3,P4,P5,P6,P7)
            #P1=nnodes,P2=RN,P3=capa,P4=stepL,P5=relu,P6=inMD,P7=redf

            self.flow_size[-1] = P1
            print('GG:',self.flow_size, 'Matrix size = ',P1)        
            
        elif what=='GG_MP':
            #main learning layer
            MPsize = self.flow_size[-2]
            layer = GG_MP(P1,P2,self.nlb,self.alpha,self.beta,self.gamma,P3,P4,P5,P6,MPsize,P7,P8,P9,P10)
            #P1=nnodes,P2=RN,P3=CFtres,P4=ANtrs,P5=seg,P6=stepL,P7=MPdia,,P8=MPstride,P9=relu,P10=inMD
            #layer = pooling(size_in,P1,P2,P3)
            out_index,v_noc,h_noc = pl_index(MPsize[0],MPsize[1],P7,P8) #here
            self.flow_size[-1] = P1             
            self.flow_size[-2]=[v_noc,h_noc]
            print('GG and MaxPooling: ',self.flow_size, 'Matrix size = ',P1)
        
            
        elif what=='Bipolar':
            #P1 = radius
            #P2 = beth
            #P3 = mode
            layer = bipolar(self.flow_size[-1],P1,P2,P3)
            
        
        self.frame.append(layer)
        del layer
        return self.frame
    
    
    
    def init_train(self,indata):
        for i in self.frame:
            outdata = i.init_train(indata)
            indata = outdata
            print(type(i).__name__,np.shape(outdata))      
        return outdata
    
    def forward(self,indata,label):
        for i in self.frame:
            outdata = i.forward(indata)
            indata = outdata
        p_label = self.frame[-1].pr_label  
        Elist = self.verf(p_label,label)
        return p_label,Elist
    
    def bs_train(self,indata):
        for i in self.frame:
            outdata = i.bs_train(indata)
            indata = outdata
            #data flow    
        return outdata
    
    def ap_cacu(self,indata,label): 
        for i in self.frame:
            outdata = i.ap_cacu(indata,label)
            indata = outdata
        p_label = self.frame[-1].pr_label  
        Elist = self.verf(p_label,label)
        return p_label,Elist
    
    def mpap_train(self,indata,label):
        for i in self.frame:
            outdata = i.mpap_train(indata,label)
            indata = outdata
        p_label = self.frame[-1].pr_label  
        Elist = self.verf(p_label,label)
        return p_label,Elist        
    
    def vote(self,lr_label,lr_frrt):
        #vote to decide predic label
        ele_layer = np.argmax(lr_frrt,axis=0)
        print(ele_layer)
        #ele_layer = np.array(ele_layer)
        #ele = ele_layer.astype(int)
        pr_label = lr_label[ele_layer]
        return pr_label
    
    def verf(self,p_label,label):
        #last layer for label verify, input p_label from previous layer, and output prediction result
        #nlb: number of labels
        #bp_ne: back propagation, when prediction is wrong, send -1 for wrong label
        #bp_e: when prediction is correct, send 1 
        datalen = len(label)
        #nlb = max(label)+1
        
        Elist = np.equal(p_label,label)+0
        #1 if equal,0 if unequal
        print('score: ',np.sum(Elist)/datalen)
        return Elist
    
#    def blame(self,lclm_frrt,lclm_bmu):
#        ft_layer = 
#        np.argmin(lclm_frrt,axis=0)
#        ft_bmu = lclm_bmu[np.arange(len(ft_layer)),ft_layer]
#        return ft_layer,ft_bmu
    
                
            
#%%
            
def rep_data(Elist,indata):
    r_data = indata[Elist!=1]
    return r_data







          


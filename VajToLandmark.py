# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 13:59:18 2021

@author: user
"""
V=[] #vowels
C=[] #consonents
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Simpler: just borders
def  VajToLandmark1(VajSequence):
    L=len(VajSequence)
    OutSequence=[]
    StateOrEvent=[]
    for i in range(L):
        # CV:
        if find(VajSequence[i],C): #if it is consonent
            if i+1<L:
                if find(VajSequence[i+1],V):
                    out=VajSequence[i]+VajSequence[i+1]
                    flag='e'                    
        # VC:
        if find(VajSequence[i],V): #if it is consonent
            if i+1<L:
                if find(VajSequence[i+1],C):
                    out=VajSequence[i]+VajSequence[i+1]
                    flag='e'
        # CC:
        if find(VajSequence[i],C): #if it is consonent
            if i+1<L:
                if find(VajSequence[i+1],C):
                    if i+3<L:
                        out=VajSequence[i+1]+VajSequence[i+2]
                        flag='v'    #vage: it is space or   
                    else:
                        out=VajSequence[i+1]+' '
                        flag='v'    #vage: it is space or   
        #CS: (C+space)
        if find(VajSequence[i],C): #if it is consonent
            if i+2<L:
                if find(VajSequence[i+1],' '):
                    out=VajSequence[i+1]+VajSequence[i+2]
                    flag='v'    #vage: it is space or  
                    
        
        OutSequence.append(out)
        StateOrEvent.append(flag)
    
    return OutSequence,StateOrEvent
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# harder: borders and events
def  VajToLandmark2(VajSequence):
    L=len(VajSequence)
    OutSequence=[]
    StateOrEvent=[]
    for i in range(L):
        # CV:
        if find(VajSequence[i],C): #if it is consonent
            if i+1<L:
                if find(VajSequence[i+1],V):
                    out=VajSequence[i]+VajSequence[i+1]
                    flag='e'                    
        # VC:
        if find(VajSequence[i],V): #if it is consonent
            if i+1<L:
                if find(VajSequence[i+1],C):
                    out=VajSequence[i]+VajSequence[i+1]
                    flag='e'
        # CC:
        if find(VajSequence[i],C): #if it is consonent
            if i+1<L:
                if find(VajSequence[i+1],C):
                    if i+3<L:
                        out=VajSequence[i+1]+VajSequence[i+2]
                        flag='v'    #vage: it is space or   
                    else:
                        out=VajSequence[i+1]+' '
                        flag='v'    #vage: it is space or   
        #CS: (C+space)
        if find(VajSequence[i],C): #if it is consonent
            if i+2<L:
                if find(VajSequence[i+1],' '):
                    out=VajSequence[i+1]+VajSequence[i+2]
                    flag='v'    #vage: it is space or  
                    
        # V
        if find(VajSequence[i],V):
            out=VajSequence[i]+VajSequence[i]
            flag='s'
        OutSequence.append(out)
        StateOrEvent.append(flag)
    
    return OutSequence,StateOrEvent
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
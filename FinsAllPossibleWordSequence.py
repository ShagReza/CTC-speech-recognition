# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 11:01:35 2021

@author: user
"""
 #Find All possible word sequence
 
 a=['a1','a2']
 b=['b1','b2']
 c=['c1','c2']
 
 A=a+b+c

'''
from itertools import permutations
perms = [' '.join(p) for p in permutations(A)]
'''

'''
Len=3
TopN=2
for i in range(Len):
    A=[]
    for j in range(TopN):
        A.append(a[j]*(2**(3-(i+1))))
        
 
'''



A=[]
Len=len(VocabIndex)
for i in range(Len):
    print(i)
    v=VocabIndex[i][0]
    B=[]
    for j in range(TopN):
        B=B+([v[j]]*(TopN**(Len-(i+1))))
    B=B*(TopN**(i))
    A.append(B)
    
C=A[0]
        
        
        
        
        
        
        
        
        
        
        

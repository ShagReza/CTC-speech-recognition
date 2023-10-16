# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 15:23:54 2021

@author: user
"""
listmain = ['car', 'animal', 'house', 'animation']
listtest=['ouse']
a,b=find_closet_match(listtest[0], listmain)
a
#test_str=listtest[0]
#list2check=listmain

import numpy as np
def find_closet_match(test_str, list2check):
    scores = {}
    for ii in list2check:
        cnt = 0
        if len(test_str)<=len(ii):
            str1, str2 = test_str, ii
        else:
            str1, str2 = ii, test_str
        for jj in range(len(str1)):
            cnt += 1 if str1[jj]==str2[jj] else 0
        scores[ii] = cnt
    scores_values        = np.array(list(scores.values()))
    closest_match_idx    = np.argsort(scores_values, axis=0, kind='quicksort')[-1]
    closest_match        = np.array(list(scores.keys()))[closest_match_idx]
    closest_match
    return closest_match, closest_match_idx




def levenshtein(a,b):
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a,b = b,a
        n,m = m,n

    current = list(range(n+1))
    for i in range(1,m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1,n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]
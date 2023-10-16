# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 13:54:14 2021

@author: user
"""
"""
LICENSE

This file is part of Speech recognition with CTC in Keras.
The project is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version.
The project is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this project.
If not, see http://www.gnu.org/licenses/.

"""
#-----------------------------------------------------------
from itertools import groupby

import numpy as np
from math import log
from numpy import array
from numpy import argmax
from utils.text_utils import int_to_text_sequence
from utils.wer_utils import wers, pers
import itertools
BeamSize=10
TopN=5 #!!!!!!!!!!!!!!!!!!!
#-----------------------------------------------------------
def WER_LM(test_func, data_gen,UniProb,BiProb,BiLefth,BiRight,vocab,words,vocab2):
    #methods:
    method=1 # apply just dictionary with one best word and one beamsize
    method=2# apply with Dic, beam size and LM
    out_true = []
    out_pred = []
    for batch in range(0, data_gen.__len__(), data_gen.batch_size):
        print(batch)
        input_data, _ = data_gen.__getitem__(batch)
        x_data = input_data.get("the_input")
        y_data = input_data.get("the_labels")

        for i in y_data:
            out_true.append("".join(int_to_text_sequence(i)))
            
        Out = beamWithLM_decode(method,test_func, x_data,UniProb,BiProb,BiLefth,BiRight,vocab,words,vocab2)
        
        if method==1:
            Out=" ".join(Out)
 
        out_pred.append(Out)
        #print("".join(int_to_text_sequence(i)))
        #print(Out)        
    out = wers(out_true, out_pred)
    return out
#-----------------------------------------------------------
import itertools
def beamWithLM_decode(method,test_func, x_data,UniProb,BiProb,BiLefth,BiRight,vocab,words,vocab2):
    y_pred = test_func([x_data])[0]
    decoded = []   
    for i in range(0,y_pred.shape[0]):
        DecSeq=y_pred[i,:,:]
        Decoded=beam_search_decoder(DecSeq, BeamSize)
        if method==1:
            out_pred=ApplyDicOneBest(Decoded,UniProb,BiProb,BiLefth,BiRight,vocab,words,vocab2) 
        if method==2:
            out_pred=ApplyDicLM(Decoded,UniProb,BiProb,BiLefth,BiRight,vocab,words,vocab2)  
    return out_pred
#-----------------------------------------------------------

def ApplyDicLM(Decoded,UniProb,BiProb,BiLefth,BiRight,vocab,words,vocab2):
    Prob=[0]*BeamSize
    BestString=[0]*BeamSize
    for i in range(BeamSize):
        Dec=[Decoded[i]]
        Dec = list(itertools.chain(*Dec))
        out_pred = []
        Prob[i]=Dec[-1]
        Dec[-1]=[]
        Dec = list(itertools.chain(*Dec))
        decoded_batch=Dec
        temp = [k for k, g in groupby(decoded_batch)]
        temp[:] = [x for x in temp if x != [30]]
        OutChar="".join(int_to_text_sequence(temp))
        OutChar = OutChar.split()
        # Find more probable words sequence from dictionary  
        VocabIndex=[]
        for c,b in enumerate(OutChar):
            Dist=[]
            for a in vocab2:
                Dist.append(levenshtein(a,b))
            idx=sorted(range(len(Dist)), key=lambda k: Dist[k])
            VocabIndex.append([idx[0:TopN]])
        # Apply Language model:
        PLM,BestWords=ApplyLanguageModel(VocabIndex,BiProb,BiLefth,BiRight,vocab2,words)  
        Prob[i]=Prob[i]+PLM
        BestString[i]=BestWords
    #--------    
    I=np.argmax(Prob)
    OutChar=BestString[I]
    #OutChar[c]=vocab2[idx[0]]
    #------
    return(OutChar)
#-----------------------------------------------------------
# Simplified Bigram without Unigrams 
def ApplyLanguageModel(VocabIndex,BiProb,BiLefth,BiRight,vocab2,words):    
    #---
    # Find All Possible Word Sequences
    A=[]
    Len=len(VocabIndex)
    for i in range(Len):
        v=VocabIndex[i][0]
        B=[]
        for j in range(TopN):
            B=B+([v[j]]*(TopN**(Len-(i+1))))
        B=B*(TopN**(i))
        A.append(B)
    #-------------------------
    BiAll=[]
    for i in range(len(BiProb)):
        BiAll.append(BiLefth[i]+' '+BiRight[i])
    #-------------------------
    # Compute AllPossibleSeqs probabiliies
    ProbBi=np.zeros(len(A[0]))
    for i in range(len(A[0])):
        for j in range(Len):
            if j==0:
                BiString='<s>'+' '+words[A[j][i]][0]
                try :
                    d=BiAll.index(BiString)
                except ValueError :
                    d=-1 
                if d!=-1:
                    ProbBi[i]=ProbBi[i]+BiProb[d]
                else:
                    ProbBi[i]=ProbBi[i]-5
            if j== (Len-1):
                BiString=words[A[j][i]][0]+' '+'</s>'
                try :
                    d=BiAll.index(BiString)
                except ValueError :
                    d=-1 
                if d!=-1:
                    ProbBi[i]=ProbBi[i]+BiProb[d]
                else:
                    ProbBi[i]=ProbBi[i]-5
            else:
                BiString=words[A[j-1][i]][0]+' '+words[A[j][i]][0]
                try :
                    d=BiAll.index(BiString)
                except ValueError :
                    d=-1 
                if d!=-1:
                    ProbBi[i]=ProbBi[i]+BiProb[d]
                else:
                    ProbBi[i]=ProbBi[i]-5
    #-------------------------
    i=np.argmax(ProbBi)
    BestWords=''
    for j in range(Len):
        BestWords=BestWords +" "+ vocab2[A[j][i]]        
    BestWords=BestWords[1:]
    return (ProbBi[i],BestWords)
#-----------------------------------------------------------



def ApplyDicOneBest(Decoded,UniProb,BiProb,BiLefth,BiRight,vocab,words,vocab2):
    Dec=[Decoded[0]]
    Dec = list(itertools.chain(*Dec))
    Dec[-1]=[]
    Dec = list(itertools.chain(*Dec))
    decoded_batch=Dec
    #decoded_batch = list(itertools.chain(*decoded_batch))
    temp = [k for k, g in groupby(decoded_batch)]
    temp[:] = [x for x in temp if x != [30]]
    OutChar="".join(int_to_text_sequence(temp))
    OutChar = OutChar.split()
    # Find more probable words sequence from dictionary   
    for c,b in enumerate(OutChar):
        Dist=[]
        for a in vocab2:
            Dist.append(levenshtein(a,b))
        #val, idx = min((val, idx) for (idx, val) in enumerate(Dist))
        idx=sorted(range(len(Dist)), key=lambda k: Dist[k])
        OutChar[c]=vocab2[idx[0]]
    return(OutChar)
#-----------------------------------------------------------
def predict_on_batch(data_gen, test_func, batch_index):
    """
    Produce a sample of predictions at given batch index from data in data_gen

    :param data_gen: DataGenerator to produce input data
    :param test_func: Keras function that takes preprocessed audio input and outputs network predictions
    :param batch_index: which batch to use as input data
    :return: List containing original transcripts and predictions
    """
    input_data, _ = data_gen.__getitem__(batch_index)

    x_data = input_data.get("the_input")
    y_data = input_data.get("the_labels")

    res = max_decode(test_func, x_data)
    predictions = []

    for i in range(y_data.shape[0]):
        original = "".join(int_to_text_sequence(y_data[i]))
        predicted = "".join(int_to_text_sequence(res[i]))
        predictions.append([original,predicted])

    return predictions
#-----------------------------------------------------------
def calc_wer(test_func, data_gen):
    """
    Calculate WER on all data from data_gen

    :param test_func: Keras function that takes preprocessed audio input and outputs network predictions
    :param data_gen: DataGenerator to produce input data
    :return: array containing [list of WERs from each batch, average WER for all batches]
    """
    out_true = []
    out_pred = []
    for batch in range(0, data_gen.__len__(), data_gen.batch_size):
        input_data, _ = data_gen.__getitem__(batch)
        x_data = input_data.get("the_input")
        y_data = input_data.get("the_labels")

        for i in y_data:
            out_true.append("".join(int_to_text_sequence(i)))

        decoded = max_decode(test_func, x_data)
        for i in decoded:
            out_pred.append("".join(int_to_text_sequence(i)))
    print(out_true)
    print(out_pred)
    out = wers(out_true, out_pred)

    return out
#-----------------------------------------------------------
def calc_wer_beam(test_func, data_gen):
    """
    Calculate WER on all data from data_gen

    :param test_func: Keras function that takes preprocessed audio input and outputs network predictions
    :param data_gen: DataGenerator to produce input data
    :return: array containing [list of WERs from each batch, average WER for all batches]
    """
    out_true = []
    out_pred = []
    for batch in range(0, data_gen.__len__(), data_gen.batch_size):
        input_data, _ = data_gen.__getitem__(batch)
        x_data = input_data.get("the_input")
        y_data = input_data.get("the_labels")

        for i in y_data:
            out_true.append("".join(int_to_text_sequence(i)))

        decoded = beam_decode(test_func, x_data)
        for i in decoded:
            out_pred.append("".join(int_to_text_sequence(i)))

    out = wers(out_true, out_pred)

    return out
#-----------------------------------------------------------
def PER(test_func, data_gen):
    """
    Calculate WER on all data from data_gen
    
    :param test_func: Keras function that takes preprocessed audio input and outputs network predictions
    :param data_gen: DataGenerator to produce input data
    :return: array containing [list of WERs from each batch, average WER for all batches]
    """
    
    out_true = []
    out_pred = []
    for batch in range(0, data_gen.__len__(), data_gen.batch_size):
        input_data, _ = data_gen.__getitem__(batch)
        x_data = input_data.get("the_input")
        y_data = input_data.get("the_labels")

        for i in y_data:
            out_true.append("".join(int_to_text_sequence(i)))

        decoded = max_decode(test_func, x_data)
        for i in decoded:
            out_pred.append("".join(int_to_text_sequence(i)))

    out = pers(out_true, out_pred)

    return out
#-----------------------------------------------------------
def max_decode(test_func, x_data):
    """
    Calculate network probabilities with test_func and decode with max decode/greedy decode

    :param test_func: Keras function that takes preprocessed audio input and outputs network predictions
    :param x_data: preprocessed audio data
    :return: decoded: max decoded network output
    """
    y_pred = test_func([x_data])[0]
    decoded = []
    for i in range(0,y_pred.shape[0]):
        decoded_batch = []
        for j in range(0,y_pred.shape[1]):
            decoded_batch.append(np.argmax(y_pred[i][j]))
        temp = [k for k, g in groupby(decoded_batch)]
        temp[:] = [x for x in temp if x != [30]]
        decoded.append(temp)
    return decoded
#-----------------------------------------------------------

def beam_decode(test_func, x_data):
    y_pred = test_func([x_data])[0]
    decoded = []   
    for i in range(0,y_pred.shape[0]):
        DecSeq=y_pred[i,:,:]
        Decoded=beam_search_decoder(DecSeq, BeamSize)
        Dec=[Decoded[0]] # 0: like max decode
        Dec = list(itertools.chain(*Dec))
        Dec[-1]=[]
        Dec = list(itertools.chain(*Dec))
        decoded_batch=Dec
        #decoded_batch = list(itertools.chain(*decoded_batch))
        temp = [k for k, g in groupby(decoded_batch)]
        temp[:] = [x for x in temp if x != [30]]
        decoded.append(temp)
        
    return decoded
#-----------------------------------------------------------

# beam search

def beam_search_decoder(data, k):
	sequences = [[list(), 0.0]]
	# walk over each step in sequence
	for row in data:
		all_candidates = list()
		# expand each current candidate
		for i in range(len(sequences)):
			seq, score = sequences[i]
			for j in range(len(row)):
				candidate = [seq + [j], score -row[j]]
				all_candidates.append(candidate)
		# order all candidates by score
		ordered = sorted(all_candidates, key=lambda tup:tup[1])
		# select k best
		sequences = ordered[:k]
        
	return sequences
#-----------------------------------------------------------
def beam_search_decoder_main(data, k):
	sequences = [[list(), 0.0]]
	# walk over each step in sequence
	for row in data:
		all_candidates = list()
		# expand each current candidate
		for i in range(len(sequences)):
			seq, score = sequences[i]
			for j in range(len(row)):
				candidate = [seq + [j], score - log(row[j])]
				all_candidates.append(candidate)
		# order all candidates by score
		ordered = sorted(all_candidates, key=lambda tup:tup[1])
		# select k best
		sequences = ordered[:k]
	return sequences
#-----------------------------------------------------------
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
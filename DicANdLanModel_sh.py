# -*- coding: utf-8 -*-
"""
vocab_file="D:\Shapar\ShaghayeghUni\AfterPropozal\Phase3-SpeechRecognition\Dic&LanModels\WordDict3k_sh.txt"
def load_vocab(vocab_file):
  vocab = []
  with open(vocab_file, 'r') as f:
    for line in f:
        vocab.append(line)
  #vocab.append('_')
  return vocab

vocab=load_vocab(vocab_file)


vocab_file="D:\Shapar\ShaghayeghUni\AfterPropozal\Phase3-SpeechRecognition\Dic&LanModels\WordDict3k_sh.txt"
def load_vocab(vocab_file):
  vocab = []
  words=[]
  with open(vocab_file, 'r') as f:
    for line in f:
        data = line.strip().split('\n')
        word = [item.split('\t')[0] for item in data if item]
        words.append(word)
        wanted_data= [item.split('\t')[1] for item in data if item]
        vocab.append(wanted_data)       
  return vocab,words
vocab,words=load_vocab(vocab_file)

"""






def load_vocab(vocab_file):   
    vocab = []
    vocab2=[]
    words=[]
    with open(vocab_file, 'r') as f:       
        for line in f:
           
            data = line.strip().split('\n')
            word = [item.split('\t')[0] for item in data if item]
            words.append(word)
            wanted_data= [item.split('\t')[1] for item in data if item]
            w=wanted_data
            w=[b for segments in w for b in segments.split()]
            w = [b.replace('zh', '}') for b in w]
            w = [b.replace('ch', '{') for b in w]
            w = [b.replace('gs', '\'') for b in w]
            #w = [b.replace('gh', 'q') for b in w]
            w = [b.replace('aa', 'w') for b in w]
            w = [b.replace('kh', 'x') for b in w]
            vocab.append(w) 
            vocab2.append("".join(w))
            
    return vocab,words,vocab2
  



"""
LanModel="D:\Shapar\ShaghayeghUni\AfterPropozal\Phase3-SpeechRecognition\Dic&LanModels\a"
def load_LanModel(LanModel):
  LM = []
  with open(vocab_file, 'r') as f:
    for line in f:
        LM.append(line)
  return LM
LM=load_LanModel(LanModel)
"""

def Load_LM(DicModel,LanModel):
    import pandas as pd    
    df = pd.read_csv(DicModel, sep="\t", header=None)
    UniProb1=df[0]
    UniWords=df[1]
    UniProb2=df[2]
    UniProb=df[0]+df[2]    
    
    df = pd.read_csv(LanModel, sep="\t", header=None)
    BiProb=df[0]
    BiLefth=df[1]
    BiRight=df[2]
    return(UniProb,BiProb,BiLefth,BiRight)







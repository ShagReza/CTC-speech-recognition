
#--------------------------------------------
from scipy import io
import numpy as np

def find_pair(seq, a, b):
    i = 1
    EventIndex=[]
    while i < len(seq):
        if seq[i] == b and seq[i - 1] == a: 
            EventIndex.append(i-1)
            EventIndex.append(i)
        i =i+2
    return (EventIndex)
#--------------------------------------------  
def FindEventsCore(seq, a, b):
    i = 1
    StateIndex=[]
    Index=[]
    k=0
    while i < len(seq):        
        if seq[i] == b and seq[i - 1] == a: 
            Index.append(i-1)
            k=1
        else:
            if k==1:
                mm=floor(len(Index)/2)
                StateIndex.append(Index[mm])
                StateIndex.append(Index[mm-1])
                StateIndex.append(Index[mm+1])
                k=0
                Index=[]                        
        i =i+2
    return (EventIndex)      
#--------------------------------------------        
def ExtractingSimpLandmarksAndTheirLabels(Events, States):
    load(Labels30)
    load(Feats)
    LenLabels=
    LE=len(Events)
    LS=len(States)
    Landmarks=[]
    LabLandmarks=[]
    for i in range(LenLabels):
        lab=Labels30[i]
        feat=Feats[i]
        for i in range(LE):
            EventIndex=find_pair(lab, Events[i][0], Events[i][1])
            SelectedFeat=feat[EventIndex,:]
            lab=[0 for i in range(60)]
            lab[Events[i][0]]=1
            lab[Events[i][1]]=1
            lab= #lab to ndaray
            labs= # repmat lab according to EventIndex length
            Landmarks.append(SelectedFeat)
            LabLandmarks.append(labs) 
        for i in range(LS):
            StateIndex=FindEventsCore(lab, States[i][0], States[i][1])
            SelectedFeat=feat[StateIndex,:]
            lab=[0 for i in range(60)]
            lab[Events[i][0]]=1
            lab[Events[i][1]]=1
            lab= #lab to ndaray
            labs= # repmat lab according to StateIndex length
            Landmarks.append(SelectedFeat)
            LabLandmarks.append(labs)   
    save(Landmarks)
    save (LabLandmarks)                     
    return
#--------------------------------------------
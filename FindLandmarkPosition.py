

Chars=[a,b,...] #characters representing Vajs

# Simple: 
# simplification1:forcely select any frame index which is better in recognizing that landmark
# simplification2: find the best value using 10 adjacent frames
# important: big silences should be emitted!!!!!!!!!!!!!!!!!!!!!!!
def  FindLandmarkPosition1(OutSequence,StateOrEvent,LandmarkRecognitionOut):
    L=len(OutSequence)
    L2=len(LandmarkRecognitionOut)
    j=0
    N=10 #adjacent frames
    Alignent=[]
    for i in range (L):
        if StateOrEvent[i]='e'
            Lefth=Chars.index(OutSequence[i,0])
            Right=Chars.index(OutSequence[i,1])
            if j<10:
                s=j
                e=j+10
            elif j+5>L2:
                e=L2
                s=L2-10
            else:
                e=j+5
                s=j+5
        List1=[s:e]
        for k1,k2 in enumerate(List1)
            p(k1)=LandmarkRecognitionOut[Lefth][k2]+LandmarkRecognitionOut[Right][k2]
        k=List1[argmax(p)] #index of landmark occurance
        Align[j:k]=OutSequence[i,0] #repeat the character from j to k
        Alignemtn.append(Align)
        j=k+1
        if StateOrEvent[i]='v'
            #if it is vage what we should do????????????????????????????
        
    # !!!!!!!!! Note that   we dont have blank in landmark alignment
    # but should have space (for silence or between two words!) !!!!!!!!         
    
    return(Alignment)
    
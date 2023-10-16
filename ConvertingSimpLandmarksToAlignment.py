
from VajToLandmark import VajToLandmark1
from VajToLandmark import VajToLandmark2
from FindLandmarkPosition import FindLandmarkPositio1
# Converting simplified landmarks to Alignment
def ConvertingSimpLandmarksToAlignment(LandmarkRecognitionOut,VajSequence):
    #------------
    OutSequence,StateOrEvent = VajToLandmark1(VajSequence)
    # OutSequence,StateOrEvent = VajToLandmark2(VajSequence, OutSequence)
    #------------
    Alignment=FindLandmarkPosition1(OutSequence,StateOrEvent,LandmarkRecognitionOut)
    #------
    return(Alignment)


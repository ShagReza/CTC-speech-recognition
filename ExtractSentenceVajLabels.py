        # Extract Sentence Vaj Labels

# Read label file of each file from small-farstdat
# Segment each label using big silences
# emit silence from the beginning and the end
# store them adding the name of the number of each farsdat sentence 
# input : vaj labels of each file, example: Z1.txt
# output: vaj labels of each sentence, example: S1-49.txt, S1-50.txt,..., S1-58.txt


def ExtractSentenceVajLabels():
    #-------------------------------------
    # Read label file of each file from small-farstdat
    #-------------------------------------
    # Segment each label using big silences
    #-------------------------------------
    # emit silence from the beginning and the end
    #-------------------------------------
    # reading each file meta data (number of red sentences)
    #-------------------------------------
    # store segmented labels with the name of the number of each farsdat sentence 
    #-------------------------------------
    print('Extracting Setences Vaj Labels done completely!')
    #-------------------------------------
    return
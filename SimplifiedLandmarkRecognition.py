# -*- coding: utf-8 -*-
"""
Shaghayegh Reza
"""
#from ExtractSentenceVajLabels import ExtractSentenceVajLabels
from Converting44LabelsTo30 import Converting44LabelsTo30
from SimplifiedLandmarks import SimplifiedLandmarks
from ExtractingLandmarksAndTheirLabels import ExtractingLandmarksAndTheirLabels
from TrainSimpLandmarkRecognition import TrainSimpLandmarkRecognition

#------------------------------------------------
# Converting 44 Labels to 30
Converting44LabelsTo30()

# Simplified Landmarks Definition
Events, States=SimplifiedLandmarks()

# Extractinr Features
FeatureExtraction()

# Extracting Landmarks And Their Labels
# outputs are Landmarks and LabLandmarks
ExtractingSimpLandmarksAndTheirLabels(Events, States) 

# Training Simplified Landmark Recognition
TrainSimpLandmarkRecognition()
#-----------------------------------------------------------------------------
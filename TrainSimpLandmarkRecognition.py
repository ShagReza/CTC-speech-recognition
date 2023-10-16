import argparse
from datetime import datetime

import keras.backend as K
import tensorflow as tf
#from keras import models
from keras.models import load_model
from keras.models import save_model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
#from keras.utils import multi_gpu_model


import models
from DataGenerator import DataGenerator
from LossCallback import LossCallback
from data import combine_all_wavs_and_trans_from_csvs



def TrainSimpLandmarkRecognition():
    load(Landmarks)
    load(LabLandmarks)
 
    frequency = 16                   
    cudnnlstm = False
    model_type='brnn' #brnn, blstm, deep_rnn, deep_lstm, cnn_blstm
    units=256
    feature_type='mfcc' # 'mfcc' or 'mels'
    output_dim=60
    dropout=0.2
    n_layers= 1
    mfcc_features=26
    n_mels=40
    
    if feature_type == 'mfcc':
        input_dim = mfcc_features #default :26
    else:
        input_dim = n_mels #default 0:
        
    model = models.model(model_type=model_type, units=units, input_dim=input_dim,
                                            output_dim=output_dim, dropout=dropout, cudnn=cudnnlstm, n_layers=n_layers)
    optimizer = Adam(lr=learning_rate, epsilon=1e-8, clipnorm=2.0)
    loss='mean_squared_error'
    model.compile(loss=loss, optimizer=optimizer)
    model.summary()
    model.fit(Landmarks, LabLandmarks,epochs=100,shuffle=True)
    model.save_weights('model_weights_landmarks.h5')
    
    return




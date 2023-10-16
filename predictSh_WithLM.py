
import argparse

import keras.backend as K
from keras import models

import models2
from DataGenerator import DataGenerator
from data import combine_all_wavs_and_trans_from_csvs
from utils.train_utils import predict_on_batch, calc_wer, PER,calc_wer_beam
from utils.train_utils_sh import WER_LM

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
from DicANdLanModel_sh import load_vocab, Load_LM


DicModel="D:\Shapar\ShaghayeghUni\AfterPropozal\Phase3-SpeechRecognition\Dic&LanModels\\WordBigram3k_Unigrams.txt"
LanModel="D:\Shapar\ShaghayeghUni\AfterPropozal\Phase3-SpeechRecognition\Dic&LanModels\\WordBigram3k_Bigrams.txt"
vocab_file="D:\Shapar\ShaghayeghUni\AfterPropozal\Phase3-SpeechRecognition\Dic&LanModels\WordDict3k_sh.txt"
UniProb,BiProb,BiLefth,BiRight=Load_LM(DicModel,LanModel)
vocab,words,vocab2=load_vocab(vocab_file)

#----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
    # Predict data params:
        
    #shaghayegh:
parser.add_argument('--audio_dir', type=str, default="data_dir/farsdat_small/F.csv",
                    help='Path to .csv file of audio to predict')
parser.add_argument('--batch_size', type=int, default=1,
                    help='Number of files to predict.')
parser.add_argument('--batch_index', type=int, default=1,
                    help='Index of batch in sorted .csv file to predict.')
                    

parser.add_argument('--calc_wer', action='store_true',
                    help='Calculate the word error rate on the data in audio_dir.')

# Only need to specify these if feature params are changed from default (different than 26 MFCC and 40 mels)
parser.add_argument('--feature_type', type=str,default='mfcc',
                    help='Feature extraction method: mfcc or spectrogram. '
                         'If none is specified it tries to detect feature type from input_shape.')
parser.add_argument('--mfccs', type=int, default=26,
                    help='Number of mfcc features per frame to extract.')
parser.add_argument('--mels', type=int, default=40,
                    help='Number of mels to use in feature extraction.')

# Model load params:
parser.add_argument('--model_load', type=str,default="D:\Shapar\ShaghayeghUni\AfterPropozal\Phase3-SpeechRecognition\CTC-tools\CTCsh",
                    help='Path of existing model to load.')
parser.add_argument('--load_multi', action='store_true',
                    help='Load multi gpu model saved during parallel GPU training.')

args = parser.parse_args()
#-----------------------------------------------------------------------------




if not args.model_load:
    raise ValueError()
audio_dir = args.audio_dir

print ("\nReading test data: ")
_, df = combine_all_wavs_and_trans_from_csvs(audio_dir)

batch_size = args.batch_size
batch_index = args.batch_index

mfcc_features = args.mfccs
n_mels = args.mels
frequency = 16           # Sampling rate of data in khz (LibriSpeech is 16khz)

# Training data_params:
model_load = args.model_load
load_multi = args.load_multi

# Sets the full dataset in audio_dir to be available through data_generator
# The data_generator doesn't actually load the audio files until they are requested through __get_item__()
epoch_length = 0

# Load trained model
# When loading custom objects, Keras needs to know where to find them.
# The CTC lambda is a dummy function
custom_objects = {'clipped_relu': models2.clipped_relu,
                  '<lambda>': lambda y_true, y_pred: y_pred}

model_type='brnn'
units=256
input_dim=26
output_dim = 31 
dropout=0.2
cudnnlstm = False
n_layers = 1
model = models.model(model_type=model_type, units=units, input_dim=input_dim,
                        output_dim=output_dim, dropout=dropout, cudnn=cudnnlstm, n_layers=n_layers)
print ("Creating new model: ", model_type)
model.load_weights('model_weights.h5')



# Dummy loss-function to compile model, actual CTC loss-function defined as a lambda layer in model
loss = {'ctc': lambda y_true, y_pred: y_pred}

model.compile(loss=loss, optimizer='Adam')

feature_shape = model.input_shape[0][2]

# Model feature type
if not args.feature_type:
    if feature_shape == 26:
        feature_type = 'mfcc'
    else:
        feature_type = 'spectrogram'
else:
    feature_type = args.feature_type

print ("Feature type: ", feature_type)

# Data generation parameters
data_params = {'feature_type': feature_type,
               'batch_size': batch_size,
               'frame_length': 20 * frequency,
               'hop_length': 10 * frequency,
               'mfcc_features': mfcc_features,
               'n_mels': n_mels,
               'epoch_length': epoch_length,
               'shuffle': False
               }

# Data generators for training, validation and testing data
data_generator = DataGenerator(df, **data_params)

# Print model summary
#model.summary()
#-------------------------------------------------
model.load_weights('model_weights.h5')
input_data = model.get_layer('the_input').input
y_pred = model.get_layer('ctc').input[0]
test_func = K.function([input_data], [y_pred])

'''
print ("\n - Calculation WER on ", audio_dir)
wer = calc_wer(test_func, data_generator)
print ("Average WER: ", wer[1])

print ("\n - Calculation PER on ", audio_dir)
wer = PER(test_func, data_generator)
print ("Average PER: ", wer[1])
'''
'''
print ("\n - Calculation WER (beam search) on ", audio_dir)
wer = calc_wer_beam(test_func, data_generator)
print ("Average WER: ", wer[1])
'''

print ("\n - Calculation WER (beam search with Dic and LanModel) on ", audio_dir)
wer = WER_LM(test_func, data_generator,UniProb,BiProb,BiLefth,BiRight,vocab,words,vocab2)
print ("Average WER: ", wer[1])



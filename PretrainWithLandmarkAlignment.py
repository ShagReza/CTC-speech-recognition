#------------------------------------------------------------------
from data import combine_all_wavs_and_trans_from_csvs

path = "data_dir/farsdat_small/farsdat_small_ph_3.csv"
print ("\nReading training data:")
_, input_dataframe = combine_all_wavs_and_trans_from_csvs(path)
  
LL=len(input_dataframe)
#-------------------------------------------------------------------






input_dim=26
output_dim=29
dropout=0.2
numb_of_dense=3
n_layers=1


"""
:param units: Hidden units per layer
:param input_dim: Size of input dimension (number of features), default=26
:param output_dim: Output dim of final layer of model (input to CTC layer), default=29
:param dropout: Dropout rate, default=0.2
:param numb_of_dense: Number of fully connected layers before recurrent, default=3
:param n_layers: Number of bidirectional recurrent layers, default=1
:return: network_model: brnn

Default model contains:
 1 layer of masking
 3 layers of fully connected clipped ReLu (DNN) with dropout 20 % between each layer
 1 layer of BRNN
 1 layers of fully connected clipped ReLu (DNN) with dropout 20 % between each layer
 1 layer of softmax
"""

# Input data type
dtype = 'float32'
# Kernel and bias initializers for fully connected dense layers
kernel_init_dense = 'random_normal'
bias_init_dense = 'random_normal'

# Kernel and bias initializers for recurrent layer
kernel_init_rnn = 'glorot_uniform'
bias_init_rnn = 'zeros'

# ---- Network model ----
# x_input layer, dim: (batch_size * x_seq_size * features)
input_data = Input(name='the_input',shape=(None, input_dim), dtype=dtype)

# Masking layer
x = Masking(mask_value=0., name='masking')(input_data)

# Default 3 fully connected layers DNN ReLu
# Default dropout rate 20 % at each FC layer
for i in range(0, numb_of_dense):
    x = TimeDistributed(Dense(units=units, kernel_initializer=kernel_init_dense, bias_initializer=bias_init_dense,
                              activation=clipped_relu), name='fc_'+str(i+1))(x)
    x = TimeDistributed(Dropout(dropout), name='dropout_'+str(i+1))(x)

# Bidirectional RNN (with ReLu)
for i in range(0, n_layers):
    x = Bidirectional(SimpleRNN(units, activation='relu', kernel_initializer=kernel_init_rnn, dropout=0.2,
                                bias_initializer=bias_init_rnn, return_sequences=True),
                      merge_mode='concat', name='bi_rnn'+str(i+1))(x)

# 1 fully connected layer DNN ReLu with default 20% dropout
x = TimeDistributed(Dense(units=units, kernel_initializer=kernel_init_dense, bias_initializer=bias_init_dense,
                          activation='relu'), name='fc_4')(x)
x = TimeDistributed(Dropout(dropout), name='dropout_4')(x)

# Output layer with softmax
y_pred = TimeDistributed(Dense(units=output_dim, kernel_initializer=kernel_init_dense,
                               bias_initializer=bias_init_dense, activation='softmax'), name='softmax')(x)


model=Model([input_data], [y_pred])
model.summary()
#------------------------------------------------------------------------------
optimizer = Adam(lr=learning_rate, epsilon=1e-8, clipnorm=2.0)
loss='CategoricalCrossentropy'
model.compile(loss=loss, optimizer=optimizer)
eopch=100
for i in range(epoch)
    for i in range(LL):
        wavfile=input_dataframe[i][0]
        labels=input_dataframe[i][1]
        feat=FeatExtract(wavfile)
        LandmarkRecognitionOut=ApplyLandrmarkRecognition(feat)
        VajSequence=labels
        Alignment=ConvertingSimpLandmarksToAlignment(LandmarkRecognitionOut,VajSequence)
        AlignmentlignedOut=AlignmentTodesiredOutput(Alignment) #تبدیل همردیفی به وان-هات برای برچسب 0 تا 31
        model.fit(feat, AlignmentlignedOut,epochs=1)




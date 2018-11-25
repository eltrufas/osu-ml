import pandas as pd
import pickle
import numpy as np

# initial loading of data
examples = pickle.load(open('examples.p', 'rb'))

def tp_to_tuple(tp):
    return (tp['offset'], tp['ms_per_beat'], tp['meter'], 0)

examples = [(beatmap, features) for (beatmap, features) in examples if features is not None]

# Transform timing points into numpy arrays

max_output_len = max(len(beatmap['timing_points']) for (beatmap, features) in examples)
max_input_len = max(len(features) for (beatmap, features) in examples if examples )


def beatmap_to_output(beatmap):
    tps = beatmap['timing_points'][:]
    prev_positive = tps[0]['ms_per_beat']
    
    features = [tp_to_tuple(tps[0])]
    
    for i in range(1, len(tps)):
        tp = dict(tps[i])
        if tp['ms_per_beat'] < 0:
            tp['ms_per_beat'] = prev_positive * (-tp['ms_per_beat'] / 100)
        features.append(tp_to_tuple(tp))
    
    arr = np.array(features)
    
    arr[:, 0] = np.ediff1d(np.pad(arr[:, 0], (1, 0), 'constant'))
    arr[-1, 3] = 1
    n_pad = max_output_len - arr.shape[0] 
    arr = np.pad(arr, ((0, n_pad), (0, 0)), mode='constant')
    return arr


def build_input(features):
    features = np.ediff1d(np.pad(features, (1, 0), 'constant'))
    zeros = np.zeros(features.shape)
    zeros[-1] = 1
    inp = np.dstack((features, zeros)).reshape(features.size, 2)
    
    n_pad = max_input_len - features.size
        
    return np.pad(inp, ((0, n_pad), (0, 0)), mode='constant')


def build_target(decoded):
    target = decoded[1:,:]
    return np.pad(target, ((0, 1), (0, 0)), mode='constant')


bare_examples = [(build_input(features),
                  beatmap_to_output(beatmap))
                 for (beatmap, features) in examples]

from keras.models import Model
from keras.layers import Input, LSTM, Dense, Bidirectional, Concatenate
from keras.callbacks import TensorBoard
from keras.utils import normalize, to_categorical
from sklearn import preprocessing

from time import time

# Process training sequences into a form that can be fed to the model

encoder_input_data, decoder_input_data = zip(*bare_examples)

def scale(X, scaler):
    return scaler.transform(X.reshape(-1, X.shape[2])).reshape(X.shape)    


encoder_input_data = np.array(encoder_input_data)
decoder_input_data = np.array(decoder_input_data)
decoder_target_data = np.pad(decoder_input_data[:,1:,:], ((0, 0), (0, 1), (0, 0)), mode='constant')

decoder_input_data_done = decoder_input_data[:,:,-1:]
decoder_input_data_meter = to_categorical(decoder_input_data[:,:,2:3])
decoder_input_data_tp = decoder_input_data[:,:,:-2]
decoder_target_data_done = decoder_target_data[:,:,-1:]
decoder_target_data_meter = to_categorical(decoder_target_data[:,:,2:3])
decoder_target_data_tp = decoder_target_data[:,:,:-2]


old_shape = decoder_input_data_tp.shape
decoder_input_data_tp = decoder_input_data_tp.reshape(-1, old_shape[2])
print(decoder_input_data_tp.shape)
scaler = preprocessing.StandardScaler().fit(decoder_input_data_tp)
decoder_input_data_tp = decoder_input_data_tp.reshape(old_shape)

decoder_input_data_tp = scale(decoder_input_data_tp, scaler)
decoder_target_data_tp = scale(decoder_target_data_tp, scaler)

print(encoder_input_data.shape)
print(decoder_input_data_done.shape)
print(decoder_input_data_meter.shape)
print(decoder_input_data_tp.shape)
print(decoder_target_data_done.shape)
print(decoder_target_data_meter.shape)
print(decoder_target_data_tp.shape)

# Build neural network architecture
latent_dim = 1000

encoder_inputs = Input(shape=(None, bare_examples[0][0].shape[1]))
encoder = Bidirectional(LSTM(latent_dim, return_state=True))
encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(encoder_inputs)

state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])

encoder_states = [state_h, state_c]

decoder_inputs_done = Input(shape=(None, decoder_input_data_done.shape[2]))
decoder_inputs_meter = Input(shape=(None, decoder_input_data_meter.shape[2]))
decoder_inputs_tp = Input(shape=(None, decoder_input_data_tp.shape[2]))

decoder_inputs = Concatenate()([decoder_inputs_tp, decoder_inputs_meter, decoder_inputs_done])

decoder_lstm = LSTM(latent_dim * 2, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)

decoder_dense1_tp_output = Dense(500)(decoder_outputs)
decoder_dense1_meter_output = Dense(500)(decoder_outputs)
decoder_dense1_done_output = Dense(500)(decoder_outputs)

decoder_dense2_tp = Dense(decoder_target_data_tp.shape[2], activation=None)
decoder_dense2_meter = Dense(decoder_target_data_meter.shape[2], activation='softmax')
decoder_dense2_done = Dense(decoder_target_data_done.shape[2], activation='sigmoid')

decoder_outputs_tp = decoder_dense2_tp(decoder_dense1_tp_output)
decoder_outputs_meter = decoder_dense2_meter(decoder_dense1_meter_output)
decoder_outputs_done = decoder_dense2_done(decoder_dense1_done_output)

model = Model([decoder_inputs_tp, decoder_inputs_meter, decoder_inputs_done, encoder_inputs],
              [decoder_outputs_tp, decoder_outputs_meter, decoder_outputs_done])

model.compile(optimizer='adam', loss=['mean_squared_error', 'categorical_crossentropy', 'binary_crossentropy'])

tensorboard = TensorBoard(log_dir='logs/{}'.format(time()), histogram_freq=0,
                          write_graph=True, write_images=False, update_freq='batch')

# Train

model.fit([decoder_input_data_tp, decoder_input_data_meter, decoder_input_data_done, encoder_input_data],
          [decoder_target_data_tp, decoder_target_data_meter, decoder_target_data_done],
          batch_size=100,
          epochs=7,
          validation_split=0.2,
          callbacks=[tensorboard])


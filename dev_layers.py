import keras.backend as K
from keras.layers import Lambda, Activation, Dropout, Embedding, SpatialDropout1D, Dense, concatenate, merge
from keras.layers import TimeDistributed  # This applies the model to every timestep in the input sequences
from keras.layers import Bidirectional, LSTM, GlobalMaxPooling1D, GlobalAveragePooling1D
# from keras.layers.advanced_activations import ELU
from keras.models import Sequential
from keras import regularizers
from keras.layers.normalization import BatchNormalization

class BiLSTM_Layer(object):
    """
    Encode the embedded words by using BiLSTM
    """

    def __init__(self, max_length, hidden_units, dropout=0.5):
        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(hidden_units, return_sequences=True, dropout=dropout, recurrent_dropout=dropout), input_shape=(max_length, hidden_units)))  # return_sequences: return the last output in the output sequence, or the full sequence.
        self.model.add(TimeDistributed(Dense(hidden_units, activation='relu', kernel_initializer='he_normal')))
        self.model.add(TimeDistributed(Dropout(dropout)))

    def __call__(self, embedded_words):
        return self.model(embedded_words)

class Pooling_Layer(object):

    def __init__(self, hidden_units, output_units, dropout=0.5, l2_weight_decay=0.0):
        self.model = Sequential()
        self.model.add(Dropout(dropout, input_shape=(hidden_units * 4,)))
        self.model.add(Dense(hidden_units, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(l2_weight_decay)))
        self.model.add(Activation('relu'))
        self.model.add(Dense(output_units, activation='softmax', kernel_initializer='zero', kernel_regularizer=regularizers.l2(l2_weight_decay)))

    def __call__(self, a, b):
        a_max = GlobalMaxPooling1D()(a)
        a_avg = GlobalAveragePooling1D()(a)

        b_max = GlobalMaxPooling1D()(b)
        b_avg = GlobalAveragePooling1D()(b)

        return self.model(concatenate([a_avg, a_max, b_avg, b_max]))   # shape=(batch_size, 4 * units)
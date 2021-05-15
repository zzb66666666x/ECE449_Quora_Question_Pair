import keras.backend as K
from keras.layers import Lambda, Activation, Dropout, Embedding, SpatialDropout1D, Dense, merge
from keras.layers import TimeDistributed  # This applies the model to every timestep in the input sequences
from keras.layers import Bidirectional, LSTM
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
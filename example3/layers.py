import keras.backend as K
from keras.layers import Lambda, Activation, Dropout, Embedding, SpatialDropout1D, Dense, merge
from keras.layers import TimeDistributed  # This applies the model to every timestep in the input sequences
from keras.layers import Bidirectional, LSTM
# from keras.layers.advanced_activations import ELU
from keras.models import Sequential
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling1D, GlobalMaxPooling1D


class EmbeddingLayer(object):

    def __init__(self, vocab_size, embedding_size, max_length, output_units, init_weights=None, nr_tune=1000, dropout=0.5):
        self.output_units = output_units
        self.max_length = max_length
        self.dropout = dropout

        self.embed = Embedding(
            vocab_size,
            embedding_size,
            input_length=max_length,
            weights=[init_weights],
            name='embedding',
            trainable=False
        )

        self.tune = Embedding(
            nr_tune,
            output_units,
            input_length=max_length,
            weights=None,
            name='tune',
            trainable=True,
        )

        self.mod_ids = Lambda(lambda sent: sent % (nr_tune - 1) + 1, output_shape=(self.max_length,))

        # Project the embedding vectors to lower dimensionality
        self.project = TimeDistributed(Dense(output_units, use_bias=False, name='project'))

    def __call__(self, sentence):
        mod_sent = self.mod_ids(sentence)

        tuning = SpatialDropout1D(self.dropout)(self.tune(mod_sent))  # SpatialDropout1D drops entire 1D feature maps instead of individual elements
        projected = self.project(self.embed(sentence))

        return merge([projected, tuning], mode='sum')


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


def attention(encoded):
    """
    INPUTS:
        encoded_a    shape=(batch_size, time_steps, num_units)
        encoded_b    shape=(batch_size, time_steps, num_units)
    """

    weights = K.batch_dot(x=encoded[0], y=K.permute_dimensions(encoded[1], pattern=(0, 2, 1)))
    return K.permute_dimensions(weights, (0, 2, 1))


def attention_output(encoded):
    input_shape = encoded[0]
    embed_size = input_shape[1]
    return (input_shape[0], embed_size, embed_size)


def attention_softmax3d(x):
    attention = x[0]
    sentence = x[1]

    # 3D softmax: calculate the subphrase in the sentence through attention
    exp = K.exp(attention - K.max(attention, axis=-1, keepdims=True))
    summation = K.sum(exp, axis=-1, keepdims=True)
    weights = exp / summation

    return K.batch_dot(weights, sentence)


def attention_softmax3d_output(x):
    attention_shape = x[0]
    sentence_shape = x[1]

    return (attention_shape[0], attention_shape[1], sentence_shape[2])


def substract(x):
    encode = x[0]
    align = x[1]

    return encode - align


def substract_output(x):
    return x[0]


def multiply(x):
    encode = x[0]
    align = x[1]

    return encode * align


def multiply_output(x):
    return x[0]


class Composition_Layer(object):

    def __init__(self, hidden_units, max_length, dropout=0.5):
        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(hidden_units, return_sequences=True, dropout=dropout, recurrent_dropout=dropout), input_shape=(max_length, 4 * hidden_units)))
        self.model.add(TimeDistributed(Dense(hidden_units, activation='relu', kernel_initializer='he_normal')))
        self.model.add(TimeDistributed(Dropout(dropout)))

    def __call__(self, _input):
        return self.model(_input)


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

        return self.model(merge([a_avg, a_max, b_avg, b_max], mode='concat'))   # shape=(batch_size, 4 * units)

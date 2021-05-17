import numpy as np
import keras
from keras.models import Model, Sequential
from keras.layers import Input, TimeDistributed, Dense, Lambda, concatenate, Dropout, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras import backend as K
from keras import regularizers

import dev_layers

# const
MAX_SEQUENCE_LENGTH = 30
EMBEDDING_DIM = 300
HIDDEN_DIM = 150
FEAT_DENSE_DIM = 50
DROPOUT_RNN = 0.25
DROPOUT_POOL = 0.1
DROPOUT_DENSE = 0.3
DROPOUT_FEATURE = 0.2
L2_WEIGHT_DECAY = 1e-4

def get_RNN(word_embedding_matrix, X_train_feat, nb_words):

    question1 = Input(shape=(MAX_SEQUENCE_LENGTH,))
    question2 = Input(shape=(MAX_SEQUENCE_LENGTH,))
    feat_input = Input(shape=(X_train_feat.shape[1],))

    feat_layer = Dense(FEAT_DENSE_DIM, activation='relu')(feat_input)
    feat_layer = Dropout(DROPOUT_FEATURE)(feat_layer)

    q1 = Embedding(nb_words + 1, 
                    EMBEDDING_DIM, 
                    weights=[word_embedding_matrix], 
                    input_length=MAX_SEQUENCE_LENGTH, 
                    trainable=False)(question1)
    q1 = TimeDistributed(Dense(HIDDEN_DIM, activation='relu'))(q1)

    q2 = Embedding(nb_words + 1, 
                    EMBEDDING_DIM, 
                    weights=[word_embedding_matrix], 
                    input_length=MAX_SEQUENCE_LENGTH, 
                    trainable=False)(question2)
    q2 = TimeDistributed(Dense(HIDDEN_DIM, activation='relu'))(q2)

    q1 = dev_layers.BiLSTM_Layer(MAX_SEQUENCE_LENGTH, HIDDEN_DIM, 1, DROPOUT_RNN)(q1)
    q2 = dev_layers.BiLSTM_Layer(MAX_SEQUENCE_LENGTH, HIDDEN_DIM, 1, DROPOUT_RNN)(q2)

    q1, q2 = dev_layers.Attention_Layer()(q1, q2)

    ## LSTM return sequence, then pooling
    # q1 = dev_layers.BiLSTM_Layer(MAX_SEQUENCE_LENGTH, HIDDEN_DIM, 4, DROPOUT_RNN)(q1)
    # q2 = dev_layers.BiLSTM_Layer(MAX_SEQUENCE_LENGTH, HIDDEN_DIM, 4, DROPOUT_RNN)(q2)
    # merged = dev_layers.Pooling_Layer(HIDDEN_DIM, 250, 4, DROPOUT_POOL, l2_weight_decay=L2_WEIGHT_DECAY)(q1, q2)
    # merged = concatenate([merged, feat_layer])

    ## the LSTM only returns final state, then concat with features
    q1 = dev_layers.BiLSTM_Layer(MAX_SEQUENCE_LENGTH, HIDDEN_DIM, 4, DROPOUT_RNN, ret_seq=False)(q1)
    q2 = dev_layers.BiLSTM_Layer(MAX_SEQUENCE_LENGTH, HIDDEN_DIM, 4, DROPOUT_RNN, ret_seq=False)(q2)
    merged = concatenate([q1, q2])
    merged = Dense(150, activation='relu')(merged)
    merged = Dropout(DROPOUT_DENSE)(merged)
    merged = BatchNormalization()(merged)
    merged = concatenate([merged, feat_layer])

    merged = Dense(150, activation='relu', kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY))(merged)
    merged = Dropout(DROPOUT_DENSE)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(150, activation='relu', kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY))(merged)
    merged = Dropout(DROPOUT_DENSE)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(100, activation='relu', kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY))(merged)
    merged = Dropout(DROPOUT_DENSE)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(10, activation='relu', kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY))(merged)
    merged = Dropout(DROPOUT_DENSE)(merged)
    merged = BatchNormalization()(merged)

    is_duplicate = Dense(1, activation='sigmoid')(merged)

    model = Model(inputs=[question1,question2,feat_input], outputs=is_duplicate)

    opt = keras.optimizers.Nadam(learning_rate=0.0004)

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model
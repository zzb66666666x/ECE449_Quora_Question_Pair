import numpy as np
import keras
from keras.models import Model, Sequential
from keras.layers import Input, TimeDistributed, Dense, Lambda, concatenate, Dropout, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras import backend as K
from keras import regularizers

import dev_layers

MAX_SEQUENCE_LENGTH = 30
FEAT_DENSE_DIM = 50
DROPOUT_FEATURE = 0.2
EMBEDDING_DIM = 300
HIDDEN_DIM = 150
DROPOUT_DENSE = 0.3 

def get_CNN(word_embedding_matrix, X_train_feat, nb_words):
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

    mergeq1=dev_layers.Convolution_layer(q1)
    mergeq2=dev_layers.Convolution_layer(q2)
    diff = Lambda(lambda x: K.abs(x[0] - x[1]), output_shape=(4*128+4*32,))([mergeq1, mergeq2])
    mul = Lambda(lambda x: x[0] * x[1], output_shape=(4*128+4*32,))([mergeq1, mergeq2])
    merge=concatenate([diff, mul])
    merge= Dropout(0.2)(merge)
    merge = BatchNormalization()(merge)
    merge = Dense(150, activation='relu')(merge)
    merge = Dropout(0.2)(merge)
    merge = BatchNormalization()(merge)
    merge=concatenate([merge, feat_layer])
    merge = Dense(100, activation='relu', kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY))(merge)
    merge = Dropout(DROPOUT_DENSE)(merge)
    merge = BatchNormalization()(merge)

    merge = Dense(10, activation='relu', kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY))(merge)
    merge = Dropout(DROPOUT_DENSE)(merge)
    merge = BatchNormalization()(merge)
    
    is_duplicate = Dense(1, activation='sigmoid')(merge)

    model = Model(inputs=[question1,question2,feat_input], outputs=is_duplicate)

    opt = keras.optimizers.Nadam(learning_rate=0.0004)

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model
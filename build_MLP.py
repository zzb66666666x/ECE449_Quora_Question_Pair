from keras.models import Model
from keras.layers import Input, TimeDistributed, Dense, Lambda, concatenate, Dropout, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras import backend as K

MAX_SEQUENCE_LENGTH = 30
EMBEDDING_DIM = 300
DROPOUT = 0.1

def get_MLP(word_embedding_matrix, X_train_feat, nb_words):

    question1 = Input(shape=(MAX_SEQUENCE_LENGTH,))
    question2 = Input(shape=(MAX_SEQUENCE_LENGTH,))

    q1 = Embedding(nb_words + 1, 
                    EMBEDDING_DIM, 
                    weights=[word_embedding_matrix], 
                    input_length=MAX_SEQUENCE_LENGTH, 
                    trainable=False)(question1)
    q1 = TimeDistributed(Dense(EMBEDDING_DIM, activation='relu'))(q1)
    q1 = Lambda(lambda x: K.max(x, axis=1), output_shape=(EMBEDDING_DIM, ))(q1)

    q2 = Embedding(nb_words + 1, 
                    EMBEDDING_DIM, 
                    weights=[word_embedding_matrix], 
                    input_length=MAX_SEQUENCE_LENGTH, 
                    trainable=False)(question2)
    q2 = TimeDistributed(Dense(EMBEDDING_DIM, activation='relu'))(q2)
    q2 = Lambda(lambda x: K.max(x, axis=1), output_shape=(EMBEDDING_DIM, ))(q2)

    merged = concatenate([q1,q2])
    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)
    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)
    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)
    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(DROPOUT)(merged)
    merged = BatchNormalization()(merged)

    is_duplicate = Dense(1, activation='sigmoid')(merged)

    model = Model(inputs=[question1,question2], outputs=is_duplicate)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

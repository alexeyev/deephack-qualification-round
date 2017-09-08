from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.layers import Input, Dense, Dropout, Flatten, Merge, Embedding, LSTM, SimpleRNN
from keras.layers.convolutional import Convolution1D, MaxPooling1D, Conv1D
import math


def build_stack(inputs, dense_outputs, maxlen, vocab_size):
    reduced_dim = vocab_size // 2
    print("Reduced_dim = ", reduced_dim)

    # emb = TimeDistributed(Dense(input_dim=vocab_size, output_dim=reduced_dim), input_shape=(maxlen, vocab_size))(inputs)

    lstm = LSTM(units=128, dropout=0.2, recurrent_dropout=0.2)(inputs)
    z = Dropout(0.2)(Dense(dense_outputs, activation='relu')(lstm))
    prefinal = Dropout(0.5)(Dense(dense_outputs // 2, activation='relu')(z))

    return prefinal


def model(filter_kernels, dense_outputs, maxlen_r, maxlen_c, vocab_size, nb_filter, mode='clas', cat_output=1,
          optimizer='rmsprop'):
    print("Constructing model with mode %s, optimizer %s" % (mode, optimizer))

    input_r = Input(shape=(maxlen_r, vocab_size), name='input_r', dtype='float32')
    input_c = Input(shape=(maxlen_c, vocab_size), name='input_c', dtype='float32')

    # Output dense layer with softmax activation
    # pred = Dense(cat_output, activation='softmax', name='output')(z)
    #
    z_r = build_stack(input_r, dense_outputs, maxlen_r, nb_filter)
    z_c = build_stack(input_c, dense_outputs, maxlen_c, nb_filter)

    concatted = Merge(mode="concat")([z_r, z_c])

    dense_after_concat = Dropout(0.3)(Dense(dense_outputs * 3 // 2, activation='relu')(concatted))

    # Default: output dense layer with boolean for binary classification
    pred = Dense(1, name='output')(dense_after_concat)
    model = Model(input=[input_r, input_c], output=[pred])
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

    print(model.get_config())

    return model

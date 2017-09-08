# from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.layers import Input, Dense, Dropout, Flatten, Merge
from keras.layers.convolutional import Convolution1D, MaxPooling1D, Conv1D


def build_stack(inputs, dense_outputs, nb_filter, maxlen, vocab_size, filter_kernels):

    conv = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[0],
                         border_mode='valid', activation='relu',
                         input_shape=(maxlen, vocab_size))(inputs)
    conv = MaxPooling1D(pool_length=3)(conv)

    conv1 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[1],
                          border_mode='valid', activation='relu')(conv)
    conv1 = MaxPooling1D(pool_length=3)(conv1)

    conv2 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[2],
                          border_mode='valid', activation='relu')(conv1)

    conv3 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[3],
                          border_mode='valid', activation='relu')(conv2)

    conv4 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[4],
                          border_mode='valid', activation='relu')(conv3)

    conv5 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[5],
                          border_mode='valid', activation='relu')(conv4)
    conv5 = MaxPooling1D(pool_length=3)(conv5)
    conv5 = Flatten()(conv5)

    zz = Dropout(0.5)(Dense(dense_outputs, activation='relu')(conv5))

    return zz


def model(filter_kernels, dense_outputs, maxlen_r, maxlen_c, vocab_size, nb_filter,
          mode='1mse', cat_output=1, optimizer='adam'):
    print("Constructing model with mode %s, optimizer %s" % (mode, optimizer))

    input_r = Input(shape=(maxlen_r, vocab_size), name='input_r', dtype='float32')
    input_c = Input(shape=(maxlen_c, vocab_size), name='input_c', dtype='float32')

    # Output dense layer with softmax activation
    # pred = Dense(cat_output, activation='softmax', name='output')(z)
    #
    z_r = build_stack(input_r, dense_outputs, nb_filter, maxlen_r, vocab_size, filter_kernels)
    z_c = build_stack(input_c, dense_outputs, nb_filter, maxlen_c, vocab_size, filter_kernels)

    concatted = Merge(mode="concat")([z_r, z_c])

    prepred = Dropout(0.5)(Dense(dense_outputs * 3 // 2, activation='relu')(concatted))

    # Default: output dense layer with boolean for binary classification
    pred = Dense(1, name='output', activation="softmax")(prepred)
    model = Model(input=[input_r, input_c], output=[pred])
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    print("Model compiled " + str(model.get_config()))

    return model

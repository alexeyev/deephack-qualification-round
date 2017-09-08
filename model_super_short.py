
from keras.layers import Input, Dense, Dropout, Flatten, Merge
from keras.layers.convolutional import Convolution1D, MaxPooling1D, Conv1D
from keras.models import Model
from keras.optimizers import SGD, Adam


def build_stack(inputs, dense_outputs, nb_filter, maxlen, vocab_size, filter_kernels):

    conv0 = Conv1D(filters=nb_filter, kernel_size=filter_kernels[0],
                   padding='valid', activation='relu',
                   input_shape=(maxlen, vocab_size))(inputs)
    conv0 = MaxPooling1D(pool_size=4)(conv0)

    conv1 = Conv1D(filters=nb_filter, kernel_size=filter_kernels[1],
                   padding='valid', activation='relu',
                   input_shape=(maxlen, vocab_size))(conv0)

    conv3 = Conv1D(filters=nb_filter, kernel_size=filter_kernels[2],
                   padding='valid', activation='relu',
                   input_shape=(maxlen, vocab_size))(conv1)
    conv3 = MaxPooling1D(pool_size=3)(conv3)

    convf = Flatten()(conv3)

    return convf


def model(filter_kernels, dense_outputs, maxlen_r, maxlen_c,
          vocab_size, nb_filter, mode='1mse', cat_output=1, optimizer='adam'):
    print("Constructing model with mode %s, optimizer %s" % (mode, optimizer))

    input_r = Input(shape=(maxlen_r, vocab_size), name='input_r', dtype='float32')
    input_c = Input(shape=(maxlen_c, vocab_size), name='input_c', dtype='float32')

    z_r = build_stack(input_r, dense_outputs, nb_filter, maxlen_r, vocab_size, filter_kernels)
    z_c = build_stack(input_c, dense_outputs, nb_filter, maxlen_c, vocab_size, filter_kernels)

    concatted = Merge(mode="concat")([z_r, z_c])

    concatted_upd = Dropout(0.5)(Dense(dense_outputs, activation='relu')(concatted))

    pred = Dense(1, name='output', activation="sigmoid")(concatted_upd)

    model = Model(input=[input_r, input_c], output=[pred])

    print(model.summary())

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    print("Model compiled " + str(model.get_config()))

    return model

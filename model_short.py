# from keras.layers.merge import concatenate
from keras.layers import Input, Dense, Dropout, Flatten, Merge
from keras.layers.convolutional import Convolution1D, MaxPooling1D, Conv1D
from keras.models import Model
from keras.optimizers import SGD, Adam


def build_stack(inputs, dense_outputs, nb_filter, maxlen, vocab_size, filter_kernels):

    conv = Conv1D(filters=nb_filter, kernel_size=filter_kernels[0],
                  padding='valid', activation='relu',
                  input_shape=(maxlen, vocab_size))(inputs)

    conv = MaxPooling1D(pool_size=3)(conv)

    conv1 = Conv1D(filters=nb_filter, kernel_size=filter_kernels[1],
                   padding='valid', activation='relu')(conv)

    conv5 = MaxPooling1D(pool_size=3)(conv1)
    conv5 = Flatten()(conv5)

    # Two dense layers with dropout of .5
    z = Dropout(0.2)(Dense(dense_outputs, activation='relu')(conv5))
    z = Dropout(0.5)(Dense(dense_outputs, activation='relu')(z))

    return z


def model(filter_kernels, dense_outputs, maxlen_r, maxlen_c, vocab_size, nb_filter, mode='1mse', cat_output=1,
          optimizer='adam'):
    print("Constructing model with mode %s, optimizer %s" % (mode, optimizer))

    input_r = Input(shape=(maxlen_r, vocab_size), name='input_r', dtype='float32')
    input_c = Input(shape=(maxlen_c, vocab_size), name='input_c', dtype='float32')

    # Output dense layer with softmax activation
    # pred = Dense(cat_output, activation='softmax', name='output')(z)
    #
    z_r = build_stack(input_r, dense_outputs, nb_filter, maxlen_r, vocab_size, filter_kernels)
    z_c = build_stack(input_c, dense_outputs, nb_filter, maxlen_c, vocab_size, filter_kernels)
    #
    # concatted = Merge([z_r, z_c])

    concatted = Merge(mode="concat")([z_r, z_c])

    # concatted = KTF.concat([z_r, z_c])

    # Default: output dense layer with boolean for binary classification
    pred = Dense(1, name='output', activation="sigmoid")(concatted)
    model = Model(input=[input_r, input_c], output=[pred])
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    print("Model compiled " + str(model.get_config()))

    return model

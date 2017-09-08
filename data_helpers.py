# coding: utf-8
import pandas as pd
import numpy as np
import string


def read_test_file(fname, cx_index=1, rp_index=2, normalize=True, binary=False, has_header=True, sep='\t'):
    print("Reading test data...")

    content = pd.read_csv(fname, index_col=False, sep=sep)
    content.dropna(inplace=True)
    content.reset_index(inplace=True, drop=True)

    cx = content.ix[:, cx_index]
    cx = np.array(cx)

    rx = content.ix[:, rp_index]
    rx = np.array(rx)

    print("Data  test read")

    return cx, rx


def read_data_file(fname, cx_index=0, rp_index=1, target_index=2, normalize=True, binary=False, has_header=True,
                   sep='\t'):
    print("Reading data...")

    content = pd.read_csv(fname, index_col=False, sep=sep)
    content.dropna(inplace=True)
    content.reset_index(inplace=True, drop=True)

    cx = content.ix[:, cx_index]
    cx = np.array(cx)

    rx = content.ix[:, rp_index]
    rx = np.array(rx)

    y = content.ix[:, target_index].values

    print(y)

    if normalize:
        max_y = np.max(np.abs(y))
        y = y / max_y

    if binary:
        vals = list(set(y))
        print(vals)
        if len(vals) > 2:
            raise Exception("Binary input data is not binary! Dataset %s, target_index=%d" % (fname, target_index))
        y = np.array([0 if a == vals[0] else 1 for a in y])

    print("Data read")

    return cx, rx, y

def load_val_data():
    val_data = read_data_file('/media/data/anton/deephack/data/validation_upd.txt', target_index=2, binary=True)
    return val_data


def load_data():
    train_data = read_data_file('/media/data/anton/deephack/data/train_upd.txt', target_index=2, binary=True)
    val_data = read_data_file('/media/data/anton/deephack/data/validation_upd.txt', target_index=2, binary=True)
    return train_data, val_data


def load_test_data():
    test_data = read_test_file('/media/data/anton/deephack/data/test_upd.txt') #test_upd.txt')
    return test_data


def mini_batch_generator(xr, xc, y, vocab, vocab_size, vocab_check, maxlen_r, maxlen_c, batch_size=128):
    for i in range(0, len(xr), batch_size):
        xr_sample = xr[i:i + batch_size]
        xc_sample = xc[i:i + batch_size]
        y_sample = y[i:i + batch_size]

        input_data_r = encode_data(xr_sample, maxlen_r, vocab, vocab_size, vocab_check)
        input_data_c = encode_data(xc_sample, maxlen_c, vocab, vocab_size, vocab_check)

        yield (input_data_r, input_data_c, y_sample)


def test_data_generator(xr, xc, vocab, vocab_size, check, maxlen_r, maxlen_c, batch_size=1000):
    for i in range(0, len(xr), batch_size):
        xr_sample = xr[i:i + batch_size]
        xc_sample = xc[i:i + batch_size]

        input_data_r = encode_data(xr_sample, maxlen_r, vocab, vocab_size, check)
        input_data_c = encode_data(xc_sample, maxlen_c, vocab, vocab_size, check)

        yield input_data_r, input_data_c


def encode_data(x, maxlen, vocab, vocab_size, check):
    # Iterate over the loaded data and create a matrix of size maxlen x vocabsize
    # In this case that will be 1014x69. This is then placed in a 3D matrix of size
    # data_samples x maxlen x vocab_size. Each character is encoded into a one-hot
    #  array. Chars not in the vocab are encoded into an all zero vector.

    input_data = np.zeros((len(x), maxlen, vocab_size))

    print(x.shape, len(x))

    for dix, sent in enumerate(x):

        counter = 0
        sent_array = np.zeros((maxlen, vocab_size))
        chars = list(sent.lower().replace(' ', ''))

        for c in chars[:min(len(chars), maxlen)]:
            char_array = np.zeros(vocab_size, dtype=np.int)
            if c in check:
                ix = vocab[c]
                char_array[ix] = 1
            sent_array[counter, :] = char_array
            counter += 1

        print(input_data.shape)
        print(dix, sent)
        input_data[dix, :, :] = sent_array

    return input_data


def enumerate_data(x, maxlen, vocab, vocab_size, check):
    # Iterate over the loaded data and create a matrix of size maxlen

    input_data = np.zeros((len(x), maxlen))
    input_data += -1

    for dix, sent in enumerate(x):
        sent_array = np.zeros((maxlen))
        chars = list(sent)

        for idx, c in enumerate(chars[:min(len(chars), maxlen)]):
            char_array = np.zeros(maxlen, dtype=np.int)
            if c in check:
                ix = vocab[c]
                char_array[idx] = ix
            sent_array[idx, :] = char_array

        input_data[dix, :] = sent_array

    return input_data


# def shuffle_matrix(x, y):
#     stacked = np.hstack((np.matrix(x).T, np.asmatrix(y).T))
#     np.random.shuffle(stacked)
#     xi = np.array(stacked[:, 0]).flatten()
#     yi = np.array(stacked[:, 1:])
#
#     return xi, yi


def create_vocab_set():
    # This alphabet is 69 chars vs. 70 reported in the paper since they include two
    # '-' characters. See https://github.com/zhangxiangxiao/Crepe#issues.

    alphabet = (
        list("qwertyuiopasdfghjklzxcvbnm<>") + list(string.digits) + list(string.punctuation) + ['\n'])
    # alphabet = (list(string.ascii_lowercase) + list(string.digits) +
    #             list(string.punctuation) + ['\n'])
    vocab_size = len(alphabet)
    check = set(alphabet)

    vocab = {}
    reverse_vocab = {}
    for ix, t in enumerate(alphabet):
        vocab[t] = ix
        reverse_vocab[ix] = t

    return vocab, reverse_vocab, vocab_size, check

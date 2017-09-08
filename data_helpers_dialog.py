# coding: utf-8
import pandas as pd
import numpy as np
import string
import data_helpers
# from deephack import data_helpers


# def encode_one_tweet(x, maxlen,vocab,vocab_size, check):
#     return encode_data(x, maxlen, vocab, vocab_size, check):


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

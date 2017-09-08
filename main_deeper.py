import datetime

import numpy as np
from sklearn.metrics.ranking import roc_auc_score

np.random.seed(123)  # for reproducibility

from deephack import data_helpers
from deephack.model import *

import json
import pandas as pd

pref = "long_cnn_data_"

import logging

logging.basicConfig(filename=pref + '_out.log',
                    format='[%(asctime)s] [%(levelname)s] %(message)s',
                    level=logging.DEBUG)

lg = logging.getLogger()
lg.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
ch.setFormatter(formatter)
lg.addHandler(ch)

# ======================================================================

# set parameters:
subset = None

# Maximum length. Longer gets chopped. Shorter gets padded.
maxlen_c = 1000
maxlen_r = 140

# Number of units in the dense layer
dense_outputs = 1024

# Conv layer kernel size
filter_kernels = [7, 4, 3, 3, 3 ,3]
# python
# Number of units in the final output layer. Number of classes.
cat_output = 1

# Compile/fit params
train_batch_size = 256
val_batch_size = 400
nb_epoch = 10

lg.info("Loading data")

(xtrr, xtrc, ytr), (xr_val, xc_val, y_val) = data_helpers.load_data()

lg.info('Creating vocab...')

vocab, reverse_vocab, vocab_size, check = data_helpers.create_vocab_set()

lg.info('Building model...')

model = model(filter_kernels, dense_outputs, maxlen_r, maxlen_c, vocab_size, nb_filter=256, mode='binary')

lg.info('Fitting model...')
initial = datetime.datetime.now()

for e in range(nb_epoch):

    lg.info("Epoch " + str(e) + "; shuffling...")

    shuffle_tr = np.random.permutation(xtrr.shape[0])
    shuffle_val = np.random.permutation(xc_val.shape[0])

    xi_r, xi_c, yi = xtrr[shuffle_tr], xtrc[shuffle_tr], ytr[shuffle_tr]
    xi_val_r, xi_val_c, yi_val = xr_val[shuffle_val], xc_val[shuffle_val], y_val[shuffle_val]

    lg.info("Generating batches...")

    if subset:
        batches = data_helpers.mini_batch_generator(xi_r[:subset], xi_c[:subset], yi[:subset],
                                                    vocab, vocab_size, check,
                                                    maxlen_r, maxlen_c,
                                                    batch_size=train_batch_size)
    else:
        batches = data_helpers.mini_batch_generator(xi_r, xi_c, yi, vocab, vocab_size,
                                                    check, maxlen_r, maxlen_c, batch_size=train_batch_size)

    val_batches = data_helpers.mini_batch_generator(xi_val_r, xi_val_c, yi_val, vocab,
                                                    vocab_size, check, maxlen_r, maxlen_c, batch_size=val_batch_size)

    accuracy = 0.0
    loss = 0.0
    step = 1
    start = datetime.datetime.now()
    lg.info('Epoch: {} (step-loss-accuracy-timeepoch-timetotal)'.format(e))

    for x_train_r, x_train_c, y_train in batches:

        f = model.train_on_batch([x_train_r, x_train_c], y_train)

        loss += f[0]
        loss_avg = loss / step
        accuracy += f[1]
        accuracy_avg = accuracy / step

        if step % 100 == 0:
            lg.info('{}\t{}\t{}\t{}\t{}'.format(step, loss_avg, accuracy_avg, (datetime.datetime.now() - start),
                                                (datetime.datetime.now() - initial)))
        if step % 1000 == 0:
            lg.info('Saving model with prefix %s.%02d.%02dK...' % (pref, e, step / 1000))
            model_name_path = '%s.%02d.step%02dK.json' % (pref, e, step / 1000)
            model_weights_path = '%s.%02d.step%02dK.h5' % (pref, e, step / 1000)
            json_string = model.to_json()
            with open(model_name_path, 'w') as f:
                json.dump(json_string, f)
            model.save_weights(model_weights_path)
        step += 1

    test_acc = 0.0
    test_loss = 0.0
    test_step = 1
    test_loss_avg = 0.0
    test_acc_avg = 0.0
    roc_auc_acc = 0.0
    roc_auc_avg = 0.0

    lg.info("Accuracy on testing batches...")

    countdown = 4

    for x_test_batch_r, x_test_batch_c, y_test_batch in val_batches:

        lg.info("Validation..." + str(countdown))
        f_ev = model.test_on_batch([x_test_batch_r, x_test_batch_c], y_test_batch)

        lg.info(str(f_ev))

        test_loss += f_ev[0]
        test_loss_avg = test_loss / test_step

        test_acc += f_ev[1]
        test_acc_avg = test_acc / test_step
        test_step += 1

        try:
            lg.info("Prediction...")
            pred = model.predict([x_test_batch_r, x_test_batch_c])
            roc_auc = roc_auc_score(y_test_batch, pred[:, 0])
            lg.info(str(pred.shape) + " ROC AUC " + str(roc_auc) + " Accuracy " + str(test_acc_avg))
            roc_auc_acc += roc_auc
            roc_auc_avg = roc_auc_acc / test_step
            lg.info(str(pred.shape) + " ROC AUC avg " + str(roc_auc_avg))

        except e as Exception:
            lg.exception("Can't predict")
            lg.error("Cant predict, " + str(e))
        countdown -= 1

        if countdown == 0:
            break

    stop = datetime.datetime.now()
    e_elap = stop - start
    t_elap = stop - initial
    lg.info('Epoch {}. Loss: {}. Accuracy: {}\nEpoch time: {}. Total time: {}\n'.format(e, test_loss_avg, test_acc_avg,
                                                                                        e_elap, t_elap))

    if pref:
        lg.info('Saving model with prefix %s.%02d...' % (pref, e))
        model_name_path = '%s.%02d.json' % (pref, e)
        model_weights_path = '%s.%02d.h5' % (pref, e)
        json_string = model.to_json()

        with open(model_name_path, 'w') as f:
            json.dump(json_string, f)

        model.save_weights(model_weights_path, overwrite=True)

        # try:
        #     lg.info("Generating data for generating submission file")
        #     (xt_r, xt_c) = data_helpers.load_test_data()
        #
        #     lg.info("Starting predictions...")
        #
        #     predictions = []
        #
        #     for x_test_r, x_test_c in data_helpers.test_data_generator(xt_r, xt_c, vocab, vocab_size, check, maxlen_r,
        #                                                                maxlen_c, batch_size=10000):
        #         lg.info(str(x_test_r.shape) + " " + str(len(predictions)))
        #         predictions.extend(list(model.predict([x_test_r, x_test_c])[:, 0]))
        #     df = pd.DataFrame({"id": list(range(len(predictions))), "human-generated": predictions})
        #     df.to_csv('%s.%02d-%s.csv' % (pref, e, str(roc_auc_avg * 1000 // 100)), index=None,
        #               columns=["id", "human-generated"])
        #     del df
        #
        # except Exception as ex:
        #     lg.exception("Can't predict on testset")
        #     lg.error(str(ex))

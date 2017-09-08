"""
    if pref and roc_auc_avg > 0.7:
        lg.info('Saving model with prefix %s.%02d...' % (pref, e))
        model_name_path = '%s.%02d.json' % (pref, e)
        model_weights_path = '%s.%02d.h5' % (pref, e)
        json_string = model.to_json()

        with open(model_name_path, 'w') as f:
            json.dump(json_string, f)

        model.save_weights(model_weights_path, overwrite=True)

        try:
            lg.info("Generating data for generating submission file")
            (xt_r, xt_c) = data_helpers.load_test_data()

            lg.info("Starting predictions...")

            predictions = []

            for x_test_r, x_test_c in data_helpers.test_data_generator(xt_r, xt_c, vocab, vocab_size, check, maxlen_r,
                                                                       maxlen_c, batch_size=10000):
                lg.info(str(x_test_r.shape) + " " + str(len(predictions)))
                predictions.extend(list(model.predict([x_test_r, x_test_c])[:, 0]))
            df = pd.DataFrame({"id": list(range(len(predictions))), "human-generated": predictions})
            df.to_csv('%s.%02d-%s.csv' % (pref, e, str(roc_auc_avg * 1000 // 100)), index=None,
                      columns=["id", "human-generated"])
            del df

        except Exception as ex:
            lg.exception("Can't predict on testset")
            lg.error(str(ex))

"""

from  deephack import data_helpers
import json

from keras.models import model_from_json
import pandas as pd

prefix = "short_cnn_huge_data_.00"
# prefix = "short_cnn_huge_data_.03.step14K"
# prefix = "shallow_cnn_data_.00"
# prefix = "shallow_cnn_data_.01.step17K"

name_json = prefix + ".json"
name_h5 = prefix + ".h5"

# set parameters:
subset = None

# Maximum length. Longer gets chopped. Shorter gets padded.
maxlen_c = 700
maxlen_r = 140

# ============= VOCAB =============

print('Creating vocab...')

vocab, reverse_vocab, vocab_size, check = data_helpers.create_vocab_set()

# ============= TEST DATA =============

print("Running test set predictions...")

try:
    print("Generating data for generating submission file")

    (xt_r, xt_c) = data_helpers.load_test_data()

    print("Loading model...")

    print("Compiling for the first time from " + prefix)
    json_file = open(name_json, 'r')
    model_as_json = json.load(json_file, encoding="UTF-8")
    json_file.close()

    model = model_from_json(model_as_json)
    model.load_weights(name_h5)
    print("Loaded model from disk...")

    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
    print("\tcompiled! now running with weights loaded from h5 files...")
    print("Starting predictions...")

    predictions = []

    for x_test_r, x_test_c in data_helpers.test_data_generator(xt_r, xt_c,
                                                               vocab, vocab_size,
                                                               check, maxlen_r, maxlen_c,
                                                               batch_size=10000):
        print(str(x_test_r.shape) + " " + str(len(predictions)))
        predictions.extend(list(model.predict([x_test_r, x_test_c])[:, 0]))

    df = pd.DataFrame({"id": list(range(len(predictions))), "human-generated": predictions})
    df.to_csv(prefix + "-2.csv", index=None, columns=["id", "human-generated"])
    del df

except Exception as ex:
    print("Can't predict on testset")
    print(str(ex))

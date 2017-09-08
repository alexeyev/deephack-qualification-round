

from  deephack import data_helpers
import json

from keras.models import model_from_json
import pandas as pd

import traceback
import prepare_df_2bots

prefix = "short_cnn_huge_data_.00"
# prefix = "shallow_cnn_data_.02"


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

    # (xt_r, xt_c) = data_helpers.load_test_data()
    (xt_c_alice, xt_r_alice, alice_ids, xt_c_bob, xt_r_bob, bob_ids) = prepare_df_2bots.load("input.json")

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


    # ALICE

    predictions = []

    # print(xt_r_alice, xt_r_bob)
    print(xt_c_alice.shape, xt_c_bob.shape)

    for x_test_r, x_test_c in data_helpers.test_data_generator(xt_r_alice, xt_c_alice,
                                                               vocab, vocab_size,
                                                               check, maxlen_r, maxlen_c,
                                                               batch_size=3):
        print(str(x_test_r.shape) + " " + str(len(predictions)))
        predictions.extend(list(1 - model.predict([x_test_r, x_test_c])[:, 0]))

    df_alice = pd.DataFrame({"dialogId": list(alice_ids), "Alice":  predictions})

    # BOB

    predictions = []

    for x_test_r, x_test_c in data_helpers.test_data_generator(xt_r_bob, xt_c_bob,
                                                               vocab, vocab_size,
                                                               check, maxlen_r, maxlen_c,
                                                               batch_size=3):
        print(str(x_test_r.shape) + " " + str(len(predictions)))
        predictions.extend(list(1 - model.predict([x_test_r, x_test_c])[:, 0]))

    df_bob = pd.DataFrame({"dialogId": list(bob_ids), "Bob": predictions})
    df = pd.merge(df_alice, df_bob, how='outer', on='dialogId').fillna(0)

    df.to_csv("___submission_2bots_shallow02.csv", index=None, columns=["dialogId", "Alice", "Bob"])



except Exception as ex:
    traceback.print_exc()
    print("Can't predict on testset")
    print(str(ex))

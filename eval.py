#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from tensorflow.contrib import learn
from input_helpers import InputHelper

# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_string("eval_filepath", "/Users/rudresh/git/deep-siamese-text-similarity/test/data/match_valid_copy.tsv",
                       "Evaluate on this data (Default: None)")
# tf.flags.DEFINE_string("eval_filepath", "/Users/rudresh/Downloads/kaggle-quora-nlp/test.csv",
#                        "Evaluate on this data (Default: None)")
tf.flags.DEFINE_string("vocab_filepath", "runs/1495015126/checkpoints/vocab",
                       "Load training time vocabulary (Default: None)")
tf.flags.DEFINE_string("model", "runs/1495015126/checkpoints/model-31000",
                       "Load trained model checkpoint (Default: None)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

if FLAGS.eval_filepath == None or FLAGS.vocab_filepath == None or FLAGS.model == None:
    print("Eval or Vocab filepaths are empty.")
    exit()

# load data and map id-transform based on training time vocabulary
inpH = InputHelper()
id_test, x1_test, x2_test = inpH.getTestDataSet(FLAGS.eval_filepath, FLAGS.vocab_filepath, 30)
print(x1_test)
exit(1)

# def func_(x):
#     return 1-x
#
# func_ = np.vectorize(func_)

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = FLAGS.model
print(checkpoint_file)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x1 = graph.get_operation_by_name("input_x1").outputs[0]
        input_x2 = graph.get_operation_by_name("input_x2").outputs[0]
        input_y = graph.get_operation_by_name("input_y").outputs[0]

        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/distance").outputs[0]

        accuracy = graph.get_operation_by_name("accuracy/accuracy").outputs[0]
        # emb = graph.get_operation_by_name("embedding/W").outputs[0]
        # embedded_chars = tf.nn.embedding_lookup(emb,input_x)
        # Generate batches for one epoch
        batches = inpH.batch_iter(list(zip(x1_test, x2_test)), 2 * FLAGS.batch_size, 1, shuffle=False)
        # Collect the predictions here
        all_predictions = []
        all_d = []
        all_id = []
        id_dev_count = 0
        count = 0
        for db in batches:
            x1_dev_b, x2_dev_b = zip(*db)
            id_dev = [int(id_test[i]) for i in range(id_dev_count, id_dev_count + len(x1_dev_b))]
            id_dev_count += len(x1_dev_b)
            batch_predictions, _ = sess.run([predictions, predictions],
                                            {input_x1: x1_dev_b, input_x2: x2_dev_b,
                                             dropout_keep_prob: 1.0})
            # print(len(all_predictions))
            # print(batch_predictions.shape)
            all_predictions = np.concatenate([all_predictions, batch_predictions])
            # print(batch_predictions)
            d = np.copy(batch_predictions)
            # print(d)
            # print(type(d))
            # d[d >= 0.5] = 999.0
            # d[d < 0.5] = 1
            # d[d > 1] = 0
            # d = func_(d)
            # print(d)
            # batch_acc = np.mean(y_dev_b == d)
            all_id = np.concatenate([all_id, id_dev])
            all_d = np.concatenate([all_d, d])
            # print("DEV acc {}".format(batch_acc))
            count += 2 * FLAGS.batch_size
            print("done for: ", count, "values")
        # for ex in all_predictions:
        #     print(ex)
        with open("result-46k.csv", 'w') as f:
            f.write("test_id,is_duplicate\n")
            for id, p in zip(all_id, all_d):
                id = int(id)
                # p = int(p)
                # if id % 1000 == 0:
                #     print("count", id)
                f.write(str(id) + "," + str(p) + "\n")
                # correct_predictions = float(np.mean(all_d == y_test))
                # print("Accuracy: {:g}".format(correct_predictions))
            # f.write("272848,0\n")
            # f.write("250131,0\n")
            # f.write("1944712,0\n")

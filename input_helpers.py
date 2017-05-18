import numpy as np
import re
import itertools
from collections import Counter
import numpy as np
import time
import gc, csv, codecs
from tensorflow.contrib import learn
from gensim.models.word2vec import Word2Vec
import gzip
from random import random
import sys

# reload(sys)
# sys.setdefaultencoding("utf-8")
from preprocess import MyVocabularyProcessor

delimiter = ","


class InputHelper(object):
    def clean_str(self, string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        # string = re.sub(r"\'s", " \'s", string)
        # string = re.sub(r"\'ve", " \'ve", string)
        # string = re.sub(r"n\'t", " n\'t", string)
        # string = re.sub(r"\'re", " \'re", string)
        # string = re.sub(r"\'d", " \'d", string)
        # string = re.sub(r"\'ll", " \'ll", string)
        # string = re.sub(r",", " , ", string)
        # string = re.sub(r"!", " ! ", string)
        # string = re.sub(r"\(", " \( ", string)
        # string = re.sub(r"\)", " \) ", string)
        # string = re.sub(r"\?", " \? ", string)
        # string = re.sub(r"\s{2,}", " ", string)

        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", string)
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"\-", " - ", text)
        text = re.sub(r"\=", " = ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" u s ", " american ", text)
        text = re.sub(r"\\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e - mail", "email", text)
        text = re.sub(r"j k", "jk", text)
        text = re.sub(r"\s{2,}", " ", text)

        return text.strip().lower()



    def getTsvData(self, filepath):
        print("Loading training data from " + filepath)
        x1 = []
        x2 = []
        y = []
        # positive samples from file
        for line in codecs.open(filepath, "r",encoding='utf-8', errors='ignore'):
            try:
                # line = bytes.decode(line)
                l = [line for line in csv.reader([line], skipinitialspace=True)][0]
                if len(l) != 3:
                    print(len(l), "Condition not satisfied>>>> ", line)
                    continue
                x1.append(self.clean_str(l[0].lower()))
                x2.append(self.clean_str(l[1].lower()))
                y.append(1 if (l[2].strip().lower() == 'y' or l[2].strip().lower() == '1') else 0)  # np.array([0,1]))
            except Exception as e:
                print(e)
        return np.asarray(x1), np.asarray(x2), np.asarray(y)

    def getTsvTestData(self, filepath):
        print("Loading testing/labelled data from " + filepath)
        id = []
        x1 = []
        x2 = []
        # y = []
        # positive samples from file
        for line in codecs.open(filepath, "r",encoding='utf-8', errors='ignore'):
            l = [line for line in csv.reader([line], skipinitialspace=True)][0]
            if len(l) != 3:
                print(len(l), "Condition not satisfied>>>> ", line)
                continue
            id.append(l[0].lower())
            x1.append(self.clean_str(l[1].lower()))
            x2.append(self.clean_str(l[2].lower()))
            # y.append(1 if (l[0].strip().lower() == 'y' or l[0].strip().lower() == '1') else 0)  # np.array([0,1]))
        return np.asarray(id), np.asarray(x1), np.asarray(x2)

    def batch_iter(self, data, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data = np.asarray(data)
        # print(data.shape)
        data_size = len(data)
        num_batches_per_epoch = int(len(data) / batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]

    def dumpValidation(self, x1_text, x2_text, y, shuffled_index, dev_idx, i):
        print("dumping validation " + str(i))
        x1_shuffled = x1_text[shuffled_index]
        x2_shuffled = x2_text[shuffled_index]
        y_shuffled = y[shuffled_index]
        x1_dev = x1_shuffled[dev_idx:]
        x2_dev = x2_shuffled[dev_idx:]
        y_dev = y_shuffled[dev_idx:]
        del x1_shuffled
        del y_shuffled
        with open('validation.txt' + str(i), 'w') as f:
            for text1, text2, label in zip(x1_dev, x2_dev, y_dev):
                f.write(str(label) + delimiter + text1 + delimiter + text2 + "\n")
            f.close()
        del x1_dev
        del y_dev

    # Data Preparatopn
    # ==================================================

    def getDataSets(self, training_paths, max_document_length, percent_dev, batch_size):
        x1_text, x2_text, y = self.getTsvData(training_paths)

        # Build vocabulary
        print("Building vocabulary")
        # vocab_processor = MyVocabularyProcessor(max_document_length, min_frequency=0)
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length, min_frequency=0)
        vocab_processor.fit_transform(np.concatenate((x2_text, x1_text), axis=0))
        print("Length of loaded vocabulary ={}".format(len(vocab_processor.vocabulary_)))
        i1 = 0
        train_set = []
        dev_set = []
        sum_no_of_batches = 0
        x1 = np.asarray(list(vocab_processor.transform(x1_text)))
        x2 = np.asarray(list(vocab_processor.transform(x2_text)))
        # Randomly shuffle data
        np.random.seed(131)
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x1_shuffled = x1[shuffle_indices]
        x2_shuffled = x2[shuffle_indices]
        y_shuffled = y[shuffle_indices]
        dev_idx = -1 * len(y_shuffled) * percent_dev // 100
        del x1
        del x2
        # Split train/test set
        self.dumpValidation(x1_text, x2_text, y, shuffle_indices, dev_idx, 0)
        # TODO: This is very crude, should use cross-validation
        x1_train, x1_dev = x1_shuffled[:dev_idx], x1_shuffled[dev_idx:]
        x2_train, x2_dev = x2_shuffled[:dev_idx], x2_shuffled[dev_idx:]
        y_train, y_dev = y_shuffled[:dev_idx], y_shuffled[dev_idx:]
        print("Train/Dev split for {}: {:d}/{:d}".format(training_paths, len(y_train), len(y_dev)))
        sum_no_of_batches = sum_no_of_batches + (len(y_train) // batch_size)
        train_set = (x1_train, x2_train, y_train)
        dev_set = (x1_dev, x2_dev, y_dev)
        gc.collect()
        return train_set, dev_set, vocab_processor, sum_no_of_batches

    def getTestDataSet(self, data_path, vocab_path, max_document_length):
        id_list, x1_temp, x2_temp = self.getTsvTestData(data_path)

        # Build vocabulary
        # vocab_processor = MyVocabularyProcessor(max_document_length, min_frequency=0)
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length, min_frequency=0)
        vocab_processor = vocab_processor.restore(vocab_path)
        # print(len(vocab_processor.vocabulary_))

        x1 = np.asarray(list(vocab_processor.transform(x1_temp)))
        x2 = np.asarray(list(vocab_processor.transform(x2_temp)))
        # Randomly shuffle data
        del vocab_processor
        gc.collect()
        return id_list, x1, x2

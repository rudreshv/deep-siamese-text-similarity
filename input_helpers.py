import numpy as np
import re
import itertools
from collections import Counter
import numpy as np
import time
import gc
from tensorflow.contrib import learn
from gensim.models.word2vec import Word2Vec
import gzip
from preprocess import MyVocabularyProcessor

class InputHelper(object):
    
    def getTsvData(self, filepath):
        print("Loading training data from "+filepath)
        x1=[]
        x2=[]
        y=[]
        # positive samples from file
        for line in open(filepath):
            l=line.strip().split("\t")
            if len(l)<2:
                continue
            x1.append(l[0])
            x2.append(l[1])
            y.append(1)#np.array([0,1]))
        # generate random negative samples
        combined = np.asarray(x1+x2)
        shuffle_indices = np.random.permutation(np.arange(len(combined)))
        combined_shuff = combined[shuffle_indices]
        for i in xrange(len(combined)):
            x1.append(combined[i])
            x2.append(combined_shuff[i])
            y.append(0) #np.array([1,0]))
        return np.asarray(x1),np.asarray(x2),np.asarray(y)
   
    def batch_iter(self, data, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data = np.asarray(data)
        print(data)
        print(data.shape)
        data_size = len(data)
        num_batches_per_epoch = int(len(data)/batch_size) + 1
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
                
    def dumpValidation(self,x_text,y,shuffled_index,dev_idx,i):
        print("dumping validation "+str(i))
        x_shuffled=x_text[shuffled_index]
        y_shuffled=y[shuffled_index]
        x_dev=x_shuffled[dev_idx:]
        y_dev=y_shuffled[dev_idx:]
        del x_shuffled
        del y_shuffled
        with open('validation.txt'+str(i),'w') as f:
            for text,label in zip(x_dev,y_dev):
                f.write(str(label)+"\t"+text+"\n")
            f.close()
        del x_dev
        del y_dev
    
    # Data Preparatopn
    # ==================================================
    
    
    def getDataSets(self, training_paths, max_document_length, percent_dev, batch_size):
        x1_text, x2_text, y=self.getTsvData(training_paths)
        
        # Build vocabulary
        print("Building vocabulary")
        vocab_processor = MyVocabularyProcessor(max_document_length,min_frequency=0)
        vocab_processor.fit_transform(np.concatenate((x2_text,x1_text),axis=0))
        print("Length of loaded vocabulary ={}".format( len(vocab_processor.vocabulary_)))
        i1=0
        train_set=[]
        dev_set=[]
        sum_no_of_batches = 0
        x1 = np.asarray(list(vocab_processor.transform(x1_text)))
        x2 = np.asarray(list(vocab_processor.transform(x2_text)))
        # Randomly shuffle data
        np.random.seed(131)
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x1_shuffled = x1[shuffle_indices]
        x2_shuffled = x2[shuffle_indices]
        y_shuffled = y[shuffle_indices]
        dev_idx = -1*len(y_shuffled)*percent_dev//100
        del x1
        del x2
        del y
        # Split train/test set
        # TODO: This is very crude, should use cross-validation
        x1_train, x1_dev = x1_shuffled[:dev_idx], x1_shuffled[dev_idx:]
        x2_train, x2_dev = x2_shuffled[:dev_idx], x2_shuffled[dev_idx:]
        y_train, y_dev = y_shuffled[:dev_idx], y_shuffled[dev_idx:]
        print("Train/Dev split for {}: {:d}/{:d}".format(training_paths, len(y_train), len(y_dev)))
        sum_no_of_batches = sum_no_of_batches+(len(y_train)//batch_size)
        train_set=(x1_train,x2_train,y_train)
        dev_set=(x1_dev,x2_dev,y_dev)
        gc.collect()
        return train_set,dev_set,vocab_processor,sum_no_of_batches
    
    def getTestDataSet(self, data_path, vocab_path, max_document_length, filter_h_pad):
        x_temp,y = self.getTsvData(data_path)

        # Build vocabulary
        vocab_processor = MyVocabularyProcessor(max_document_length,min_frequency=1)
        vocab_processor = vocab_processor.restore(vocab_path)
        print len(vocab_processor.vocabulary_)

        x = np.asarray(list(vocab_processor.transform(x_temp)))
        x = np.concatenate((np.zeros((len(x),filter_h_pad)),x),axis=1)
        # Randomly shuffle data
        del x_temp
        del vocab_processor
        gc.collect()
        return x, y

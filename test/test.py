# # # """
# # # @author : Rudresh
# # # @Created on: 24/04/17
# # # """
# # # # from gensim.models.keyedvectors import KeyedVectors
# # # #
# # # # model = KeyedVectors.load_word2vec_format('/Users/rudresh/git/Siamese-LSTM/GoogleNews-vectors-negative300.bin', binary=True)
# # # # model.save_word2vec_format('/Users/rudresh/git/Siamese-LSTM/GoogleNews-vectors-negative300.txt', binary=False)
# # # # exit(1)
# # # import numpy as np
# # # import time
# # # initW = []
# # # with open("/Users/rudresh/git/Siamese-LSTM/GoogleNews-vectors-negative300.bin", "rb") as f:
# # #     header = f.readline()
# # #     print(bytes.decode(header))
# # #     vocab_size, layer1_size = map(int, header.split())
# # #     binary_len = np.dtype('float32').itemsize * layer1_size
# # #     print(binary_len)
# # #     count = 0
# # #     print(vocab_size)
# # #     for line in range(vocab_size):
# # #         try:
# # #             count += 1
# # #             word = []
# # #             while True:
# # #                 ch = f.read(1)
# # #                 ch = bytes.decode(ch)
# # #                 if ch == ' ':
# # #                     # print("====", word)
# # #                     word = ''.join(word)
# # #                     break
# # #                 if ch != '\n':
# # #                     word.append(ch)
# # #             print("----", word, count)
# # #             # idx = vocab_processor.vocabulary_.get(word)
# # #             idx = count
# # #             if idx != 0:
# # #                 initW.append(np.fromstring(f.read(binary_len), dtype='float32'))
# # #             else:
# # #                 f.read(binary_len)
# # #         except:
# # #             f.read(binary_len)
# # #
# # #
# # # # def bytes_from_file(filename, chunksize=8192):
# # # #     with open(filename, "rb") as f:
# # # #         while True:
# # # #             chunk = f.read(chunksize)
# # # #             if chunk:
# # # #                 for b in chunk:
# # # #                     yield b
# # # #             else:
# # # #                 break
# # # #
# # # # import time
# # # # # example:
# # # # for b in bytes_from_file("/Users/rudresh/git/Siamese-LSTM/GoogleNews-vectors-negative300.bin"):
# # # #     print(b)
# # # #     time.sleep(1)
# #
# #
# # with open("/Users/rudresh/Downloads/kaggle-quora-nlp/train.csv","rb") as f:
# #     for line in f:
# #         print(line)
#
#



# import numpy as np
# def func_(x):
#     return 10-x
# a = [1, 2, 3, 4, 5, 6, 7, 8]
# a_ = np.asarray(a)
# a_func = np.vectorize(func_)
# print(a_func(a_))
# exit(1)



# import re, csv
# delimiter = ","
# # # string = '142727,"What are the best places to visit at in Rajasthan, India?","What are famous places russians Rajasthan?'
# # string = '1273220,"If 1 = Single, 2 = Double, 3 = Triple, 4 = Quadruple, what are 5, 6, 7, 8, etc. called?","How do I Simplify the following matrices: '
# # # string = '311005,"""I lent money to someone and he gave me an undated cheque in return .But now he is not giving back the money. Its been a year What to do?","My dad is a professional gambler and he gave my sister and me each $25k. The first good one of us who doubles the money anyhow wins another $250k. Should I just bet all at once in Roulette?"'
# # #
# # # string = '122170,"What?","What ""what is""?"'
# #
# # for line in csv.reader([string], skipinitialspace=True):
# #     print(line)
# #
# # exit(1)
# #
# #
# # # s = re.split(delimiter+"(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)", string)
# # # s = re.split(delimiter+"(?=(?:\"[^\"]*?(?: [^\"]*)))", string)
# # # s = re.split(delimiter+"(?=(?:[^\"]|\"[^\"]*\")*)", string)
# # # s = re.split("(?=(?:\"[^\"]*?(?: [^\"]*)*))|, (?=[^\",]+(?:,|$))", string)
# #
# # # regexExp = "(?=(?:\"[^\"]*?(?: [^\"]*)*))|, (?=[^\",]+(?:,|$))"
# # # regexExp = delimiter+"(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)"
# # regexExp = delimiter+"(?=(?:\"[^\"]*?(?: [^\"]*)))"
# # # regexExp = delimiter+"(?=(?:[^\"]|\"[^\"]*\")*)"
# #
# # # string = re.sub(r'(.)\1{1,}', r'\1', string)
# # # print(string)
# # s = re.split(regexExp, string)
# # print(len(s))
# # [print("---",ss,"====") for ss in s]
# #
# # exit(1)
#
#
# import codecs
# count = 0
# for line in codecs.open("/Users/rudresh/git/deep-siamese-text-similarity/test/data/test.csv", "r",encoding='utf-8', errors='ignore'):
#     count+=1
#     try:
#         # line = bytes.decode(line)
#         l = [line for line in csv.reader([line], skipinitialspace=True)][0]
#         if len(l) != 3:
#             print(l)
#             print(len(l), "Condition not satisfied>>>> ", line)
#             print(count)
#             continue
#     except Exception as e:
#         print(e)
# print(count)
# exit(1)
#
#
#
# import codecs
# count = 0
# for line in open("/Users/rudresh/Downloads/kaggle-quora-nlp/train_2.csv", "rb"):
#     try:
#         line = bytes.decode(line)
#         l = [line for line in csv.reader([line], skipinitialspace=True)][0]
#         if len(l) != 3:
#             print(len(l), "Condition not satisfied>>>> ", line)
#             continue
#     except Exception as e:
#         count+=1
#         print(e)
# print(count)
# exit(1)
#
# for string in open("/Users/rudresh/Downloads/kaggle-quora-nlp/train_2.csv", "r"):
# # for line in open("/Users/rudresh/git/deep-siamese-text-similarity/test/data/test.csv", "rb"):
#     string = bytes.decode(string)
#     print(string)
#     l = [line for line in csv.reader([string], skipinitialspace=True)][0]
#     if len(l) != 3:
#         print(len(l),"Condition not satisfied>>>> ", string)
#         continue
#
# exit(1)

import pandas as pd
df1 = pd.read_csv("/Users/rudresh/git/deep-siamese-text-similarity/result-46k.csv")
# df2 = pd.read_csv("/Users/rudresh/Downloads/kaggle-quora-nlp/sample_submission.csv")
#
# s1 = set(list(df1["test_id"]))
# s2 = set(list(df2["test_id"]))
# print(s2-s1)
# exit(1)

import sys
print(sys.argv[1])

def convert_(d):
    # return 1 if d>=float(sys.argv[1]) else 0
    return 1 if d>=float(0.5) else 0
# def convert_(data):
#     return 1-int(data)
df1["is_duplicate"] = df1["is_duplicate"].apply(convert_)
# df1.to_csv("/Users/rudresh/git/deep-siamese-text-similarity/result-new.csv",index=None)
# m = list(df1["test_id"])
# print(m.count(1))
m = list(df1["test_id"]+df1["is_duplicate"])
print(len(m))
print(m.count(2))
print(m.count(1))
print(m.count(0))
print(((m.count(2)+m.count(0))/len(m))*100)

# import numpy as npasd
# x1_text, x2_text= ["hi hi","hello hello"], ["this is a sentence","another sentence"]
# final_text = np.concatenate((x2_text, x1_text),axis=0)
# print(final_text)
# new_map = {}
# for x in final_text:
#     new_map[len(x.split(" "))] = x
# max_document_length = max([len(x.split(" ")) for x in final_text])
# print(max_document_length)
# print(new_map[max_document_length])
# print(sorted(new_map.keys(),reverse=True))


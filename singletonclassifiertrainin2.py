import pandas as pd
import numpy as np
import sys
import os
from get_markabledataframe import get_markable_dataframe, get_embedding_variables
from singleton_classifier import SingletonClassifierModelBuilder
from functools import reduce
from tensorflow import keras
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.models import Model, load_model
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['AUTOGRAPH_VERBOSITY'] = '10'
config =tf.compat.v1.ConfigProto() 
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)
import logging
# from numba import cuda
# cuda.select_device(0)
# cuda.close()






# import tensorflow as tf

# config = tf.compat.v1.ConfigProto()

# config.gpu_options.allow_growth = True

# sess = tf.compat.v1.Session(config=config)

print("singleton classifier Training")


embedding_indexes_file_path = 'helper_files/embedding/embedding_indexes.txt'
indexed_embedding_file_path = 'helper_files/embedding/indexed_embedding.txt'
embedding_file_path = 'helper_files/embedding/generated_embedding.txt'
word_vector, embedding_matrix, idx_by_word, word_by_idx = get_embedding_variables(embedding_indexes_file_path, indexed_embedding_file_path)
data = get_markable_dataframe("data/training/markables_23_2word2.csv", word_vector, idx_by_word)
# print("heloo")
# print(data.columns.tolist())
max_text_length = 10
max_prev_words_length = 10
max_next_words_length = 10

data_text = pad_sequences(data.mention, maxlen=max_text_length, padding='post')
data_previous_words = pad_sequences(data.previous_words.map(lambda seq: seq[(-1*max_prev_words_length):]), maxlen=max_prev_words_length, padding='pre')
data_next_words = pad_sequences(data.next_words.map(lambda seq: seq[:max_next_words_length]), maxlen=max_next_words_length, padding='post')
data_syntactic = data[['is_proper_name','is_pronoun', 'is_first_person']]

data_syntactic = np.array(list(map(lambda p: reduce(lambda x,y: x + y, [i if type(i) is list else [i] for i in p]), data_syntactic.values)))
label = np.vstack(data.is_singleton)
print("build model...")
print(" word model")
words_model_builder = SingletonClassifierModelBuilder(use_words_feature=True, use_context_feature=False, use_syntactic_feature=False, embedding_matrix=embedding_matrix)
words_model = words_model_builder.create_model()
words_model.fit([data_text], label, epochs=20)
words_model.save('models/singleton_classifiers/words.model')
logging.debug(f"Node Conversion Map: {node_conversion_map}")
new_node_index = node_conversion_map[node_key]
print("Node Key:", node_key)
print("Node Conversion Map:", node_conversion_map)
new_node_index = node_conversion_map[node_key]
words_model.save('models/singleton_classifiers/words.model')

words_model.summary()
print("context")

context_model_builder = SingletonClassifierModelBuilder(use_words_feature=False, use_context_feature=True, use_syntactic_feature=False,embedding_matrix=embedding_matrix)
context_model = context_model_builder.create_model()
context_model.fit([data_previous_words, data_next_words], label, epochs=20)
context_model.save('models/singleton_classifiers/context.model')
print("Syntactic")
print(data_syntactic.shape[1])
syntactic_model_builder = SingletonClassifierModelBuilder(use_words_feature=False, use_context_feature=False, use_syntactic_feature=True, syntactic_features_num=data_syntactic.shape[1])
syntactic_model = syntactic_model_builder.create_model()
syntactic_model.fit([data_syntactic], label, epochs=20)
syntactic_model.save('models/singleton_classifiers/syntactic.model')
print("word+context")
words_context_model_builder = SingletonClassifierModelBuilder(use_words_feature=True, use_context_feature=True, use_syntactic_feature=False,embedding_matrix=embedding_matrix)
words_context_model = words_context_model_builder.create_model()
words_context_model.fit([data_text, data_previous_words, data_next_words], label, epochs=20)
words_context_model.save('models/singleton_classifiers/words_context.model')
print("word+syntactic")
words_syntactic_model_builder = SingletonClassifierModelBuilder(use_words_feature=True, use_context_feature=False, use_syntactic_feature=True,embedding_matrix=embedding_matrix,syntactic_features_num=data_syntactic.shape[1])
words_syntactic_model = words_syntactic_model_builder.create_model()
words_syntactic_model.fit([data_text, data_syntactic], label, epochs=20)
words_syntactic_model.save('models/singleton_classifiers/words_syntactic.model')
print("contex+syntactic")
context_syntactic_model_builder = SingletonClassifierModelBuilder(use_words_feature=False,use_context_feature=True, use_syntactic_feature=True,embedding_matrix=embedding_matrix, syntactic_features_num=data_syntactic.shape[1])
context_syntactic_model = context_syntactic_model_builder.create_model()
context_syntactic_model.fit([data_previous_words, data_next_words, data_syntactic], label, epochs=20)
context_syntactic_model.save('models/singleton_classifiers/context_syntactic.model')

print("word+context+syntactic")
words_context_syntactic_model_builder = SingletonClassifierModelBuilder(use_words_feature=True, use_context_feature=True, use_syntactic_feature=True, embedding_matrix=embedding_matrix, syntactic_features_num=data_syntactic.shape[1])
words_context_syntactic_model = words_context_syntactic_model_builder.create_model()
words_context_syntactic_model.fit([data_text, data_previous_words, data_next_words, data_syntactic], label, epochs=20)
words_context_syntactic_model.save('models/singleton_classifiers/words_context_syntactic.model')


























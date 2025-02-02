import pandas as pd
import numpy as np
import sys
from get_markabledataframe import get_markable_dataframe, get_embedding_variables
from singleton_classifier import SingletonClassifierModelBuilder
from functools import reduce
from tensorflow import keras

from tensorflow.keras.preprocessing.sequence import pad_sequences
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

print("singleton classifier Training")


embedding_indexes_file_path = 'helper_files/embedding/embedding_indexes.txt'
indexed_embedding_file_path = 'helper_files/embedding/indexed_embedding.txt'
embedding_file_path = 'helper_files/embedding/generated_embedding.txt'
word_vector, embedding_matrix, idx_by_word, word_by_idx = get_embedding_variables(embedding_indexes_file_path, indexed_embedding_file_path)
data = get_markable_dataframe("data/training/markables_22.csv", word_vector, idx_by_word)
max_text_length = 10
max_prev_words_length = 3
max_next_words_length = 3

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
words_context_syntactic_model.fit([data_text, data_previous_words, data_next_words, data_syntactic], label, epochs=30)
words_context_syntactic_model.save('models/singleton_classifiers/words_context_syntactic.model')


























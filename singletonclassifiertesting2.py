import pandas as pd
import numpy as np
import os
from csv import DictReader, DictWriter
from get_markabledataframe import get_markable_dataframe, get_embedding_variables
from functools import reduce
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
import tensorflow as tf
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

print("singleton classifier testing")
np.random.seed(26061997)
embedding_indexes_file_path = 'helper_files/embedding/embedding_indexes.txt'
indexed_embedding_file_path = 'helper_files/embedding/indexed_embedding.txt'
word_vector, embedding_matrix, idx_by_word, word_by_idx = get_embedding_variables(embedding_indexes_file_path, indexed_embedding_file_path)
data_testing_file_path = "data/testing/markable_23_2word2.csv"

data = get_markable_dataframe(data_testing_file_path, word_vector, idx_by_word)
# print(data)
max_text_length = 10
max_prev_words_length = 10
max_next_words_length = 10

data_text = pad_sequences(data.mention, maxlen=max_text_length, padding='post')
data_previous_words = pad_sequences(data.previous_words.map(lambda seq: seq[(-1*max_prev_words_length):]), maxlen=max_prev_words_length, padding='pre')
data_next_words = pad_sequences(data.next_words.map(lambda seq: seq[:max_next_words_length]), maxlen=max_next_words_length, padding='post')
data_syntactic = data[['is_proper_name','is_pronoun', 'is_first_person']]

data_syntactic = np.array(list(map(lambda p: reduce(lambda x,y: x + y, [i if type(i) is list else [i] for i in p]), data_syntactic.values)))
label = np.vstack(data.is_singleton)

print("Load model")

words_model = tf.keras.models.load_model('models/singleton_classifiers/words.model')
# words_model = tf.keras.models.load_weights('models/singleton_classifiers/words.model')
context_model = tf.keras.models.load_model('models/singleton_classifiers/context.model')
syntactic_model = tf.keras.models.load_model('models/singleton_classifiers/syntactic.model')
words_context_model = tf.keras.models.load_model('models/singleton_classifiers/words_context.model')
words_syntactic_model = tf.keras.models.load_model('models/singleton_classifiers/words_syntactic.model')
context_syntactic_model = tf.keras.models.load_model('models/singleton_classifiers/context_syntactic.model')
words_context_syntactic_model = tf.keras.models.load_model('models/singleton_classifiers/words_context_syntactic.model')

print("Test model")

def get_classes(output, threshold=0.5):
    return list(map(lambda x: 1 if x[1] > threshold else 0, output))

def evaluate(label, pred, threshold=0.5):
    label = get_classes(label)
    pred = get_classes(pred, threshold)
    
    print('threshold %f:' % threshold)
    print(classification_report(label, pred))
    print('accuracy: %f' % accuracy_score(label, pred))

def evaluate_all(label, pred):
    for i in range(1, 10):
        evaluate(label, pred, i*0.1)

print("Words model")

words_pred = words_model.predict([data_text])
evaluate_all(label, words_pred)

print("Context model")

context_pred = context_model.predict([data_previous_words, data_next_words])
evaluate_all(label, context_pred)
print("Syntactic model")
syntactic_pred = syntactic_model.predict([data_syntactic])
evaluate_all(label, syntactic_pred)

print("Words+Context")

words_context_pred = words_context_model.predict([data_text, data_previous_words, data_next_words])
evaluate_all(label, words_context_pred)
print("Words and syntactic model")

words_syntactic_pred = words_syntactic_model.predict([data_text, data_syntactic])

evaluate_all(label, words_syntactic_pred)

print("Context and Syntactic model")


context_syntactic_pred = context_syntactic_model.predict([data_previous_words, data_next_words, data_syntactic])
evaluate_all(label, context_syntactic_pred)
print("Words,Context and Syntactice model")
words_context_syntactic_pred = words_context_syntactic_model.predict([data_text, data_previous_words, data_next_words, data_syntactic])
evaluate_all(label, words_context_syntactic_pred)
# print("output")
# print(" wordcontextsyntacticouput")
print(words_context_syntactic_pred)
output = get_classes(words_context_syntactic_pred, 0.4)
print(output)

print(" predicted singleton mentions")
rich_data_testing_file_path = "data/testing/markables_with_predicted_singleton.csv"
with open(data_testing_file_path, "r") as orifile:
    oricsv = DictReader(orifile, delimiter=' ')    
    with open(rich_data_testing_file_path, "w") as newfile:
        newcsv = DictWriter(newfile, fieldnames=oricsv.fieldnames, delimiter=' ', extrasaction='ignore')
        
        newcsv.writeheader()
        
        for row, is_singleton in zip(oricsv, output):
            newcsv.writerow({**row, 'is_singleton': is_singleton})


            















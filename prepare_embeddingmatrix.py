import numpy as np
import pickle as pkl
import sys
import pandas as pd
import csv
import get_vector_words as gv 
#from keras.layers import Embedding
from tensorflow.keras.layers import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
#import gensim
from collections import Counter

'''
MAX_NB_WORDS = len(word2vec_model.vocab)
print("Number of word vectors: {}".format(len(MAX_NB_WORDS)))
# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size, 300))
for word, i in tokenizer.word_index.items():
 embedding_vector = getVector(word)
 if embedding_vector is not None:
 embedding_matrix[i] = embedding_vector
 from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
word2vec_model.wv.vocab[word].index 
word = "whatever"  # for any word in model
i = model.vocab[word].index
model.index2word[i] == word  # will be true
 '''

# create a weight matrix for words in training docs
def create_embedmatrix(word2vec_model):
    #word2vec_model = gv.load_word_embeddings('/home/kusum/Desktop/MentionDetection-master/embeddings/hi.bin')
    MAX_NB_WORDS = len(word2vec_model.wv.vocab)
    #return type(word2vec_model.wv.vocab)
    word_index = {t[0]: i+1 for i,t in enumerate(Counter(word2vec_model.wv.vocab).most_common(MAX_NB_WORDS))}
    #MAX_NB_WORDS = len(word_index)+1
    #print(MAX_NB_WORDS)
    embedding_matrix = np.zeros((MAX_NB_WORDS+2, 300))
    for word, i in word_index.items():
        if i>=MAX_NB_WORDS:
            continue
        #embedding_vector = getVector(word)
        #if embedding_vector:
         #   embedding_matriword_index = {t[0]: i+1 for i,t in enumerate(Counter(word2vec_model.wv.vocab).most_common(MAX_NB_WORDS))}x[i] = embedding_vector
        try:
            embedding_vector = getVector(word)
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        except:
            pass 
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    return embedding_matrix     
    
    



  
'''
valid_word = Input((1,), dtype='int32')
other_word = Input((1,), dtype='int32')
# setup the embedding layer
embeddings = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1],weights=[embedding_matrix])
embedded_a = embeddings(valid_word)
embedded_b = embeddings(other_word)
'''

#pretrainedEmbeddingLayer = create_embedmatrix()
#print(pretrainedEmbeddingLayer)
import numpy as np
import sys
import pandas as pd
import gensim
#from gensim.models.keyedvectors import KeyedVectors

VECTOR_DIM = 300
embeddings_file_path = '/home/kusum/Desktop/singletonmention/embeddings/hi.bin'
def load_word_embeddings(embeddings_file_path):
    #word2vec_model = KeyedVectors.load_word2vec_format(embeddings_file_path, binary=True, encoding='utf-8', unicode_errors='ignore')
	word2vec_model = gensim.models.Word2Vec.load(embeddings_file_path)
	return word2vec_model

def get_sentence_vectors(sentence_list):
	"""
	Returns word vectors for complete sentence as a python list"""
	# s = sentence.strip().split()
	vec = [ get_word_vector(word) for word in sentence_list]
	return vec

def get_word_vector(word):
	"""
	Returns word vectors for a single word as a python list"""

	word2vec_model = load_word_embeddings(embeddings_file_path)
	#get a list of vocab
	#index_to_word = list(word2vec_model .wv.vocab.keys())
	#word_vectors = list()
	# Populate matrix of word vectors
    #for word in index_to_word:
     #   word_vectors.append(word2vec_model[word])
	if word in word2vec_model.wv.vocab:
		return word2vec_model[word]
	else:
		return np.zeros(VECTOR_DIM)
def word2idx(word):

	return word2vec_model.wv.vocab[word].index 

def idx2word(idx):

    return word2vec_model.wv.index2word[idx]

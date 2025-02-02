
import numpy as np
import pandas as pd
import numpy as np
import gensim
import sys
import pandas as pd
# from transformers import BertTokenizer
import transformers 
import keras
from transformers import BertTokenizer
from transformers import TFBertModel
# from transformers import BertModel
# from transformers import TFAutoModel, AutoTokenizer
import tensorflow as tf
# from transformers import BertTokenizer, BertModel
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
import csv
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
#from gensim.models.keyedvectors import KeyedVectors
from transformers import logging
import nltk
from nltk.tokenize import RegexpTokenizer
import spacy

logging.set_verbosity_warning()
filename = 'document_id.csv'

np.random.seed(26061997)

# data = ET.parse('data/full.xml')
embedding_file_path = 'helper_files/embedding/generated_embedding.txt'
indexed_embedding_file_path = 'helper_files/embedding/indexed_embedding.txt'
embedding_indexes_file_path = 'helper_files/embedding/embedding_indexes.txt'
doc_tagged = pd.read_csv(filename, sep =" ", encoding='utf-8', usecols=['document_id','sentence_id','sentence']) 
# doc_tagged = doc_tagged.fillna(method='ffill')
sent_list= doc_tagged['sentence'].tolist()

from indicnlp.tokenize import indic_tokenize
def generate_text_embedding_indexes(sentences):
    word_index = 1
    output_lines = []
    with open(embedding_indexes_file_path, "w", encoding="utf-8") as embedding_indexes_file:

        for sentence in sentences:
            # words = indic_tokenize.trivial_tokenize(sentence)
            # custom_tokenizer = RegexpTokenizer(custom_tokenization_pattern)
            #doc = nlp(sentence)
            #words = custom_tokenizer.tokenize(sentence)
            #words = [token.text for token in doc]
            words=sentence.split() 
            for word in words:
                print(words)
                # output_lines.append(f"{word} {word_index}")
                embedding_indexes_file.write(word + ' ' + str(word_index) + '\n')
                word_index += 1

    # return output_lines
print("Embeeding files")
# embedding_indexes = generate_text_embedding_indexes(sent_list)
print(" write one file embedding_index")
generate_text_embedding_indexes(sent_list)



from transformers import AutoTokenizer, AutoModel
# from transformers import AutoModelForMaskedLM
import torch

# Initialize the mBERT tokenizer and model with the fast tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased', use_fast=True)
model = AutoModel.from_pretrained('bert-base-multilingual-cased')

# model = AutoModelForMaskedLM.from_pretrained('bert-base-multilingual-cased')

# Function to get mBERT embeddings for a single word
def get_word_embedding(word):
    inputs = tokenizer(word, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[0, 0].tolist()  # Get the embedding of the first token

# Read words from the previously created file
with open(embedding_indexes_file_path, 'r', encoding='utf-8') as file:
    words = [line.split()[0] for line in file.readlines()]

# Generate embeddings and write to a new file
print(" write second file generated embedding")
with open(embedding_file_path, 'w', encoding='utf-8') as f:
    for word in words:
        embedding = get_word_embedding(word)
        f.write(f"{word} {' '.join(map(str, embedding))}\n")
        
print(" write third file indexed_ embedding")
# Generate embeddings and write to a new file
with open(embedding_file_path, 'r', encoding='utf-8') as file:
    words = [line.split()[0] for line in file.readlines()]
    print (words)
with open(indexed_embedding_file_path, 'w', encoding='utf-8') as f:
    for word in words:
        embedding = get_word_embedding(word)
        # indexed_embedding_file.write(' '.join(map(str, embedding[word])) + '\n')
        f.write(' '.join(map(str, embedding)) + "\n")


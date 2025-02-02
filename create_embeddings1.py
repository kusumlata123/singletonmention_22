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
from indicnlp.tokenize import indic_tokenize

#from gensim.models.keyedvectors import KeyedVectors
filename = 'document_id.csv'

np.random.seed(26061997)

# data = ET.parse('data/full.xml')
embedding_file_path = 'helper_files/embedding/generated_embedding.txt'
indexed_embedding_file_path = 'helper_files/embedding/indexed_embedding.txt'
embedding_indexes_file_path = 'helper_files/embedding/embedding_indexes.txt'
doc_tagged = pd.read_csv(filename, sep =" ", encoding='utf-8', usecols=['document_id','sentence_id','sentence']) 
doc_tagged = doc_tagged.fillna(method='ffill')
sent_list= doc_tagged['sentence'].tolist()
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = TFBertModel.from_pretrained('bert-base-multilingual-cased')
# model = TFBertModel.from_pretrained("bert-base-uncased")

def get_word_embeddings(sentence):
    # Tokenize the input sentence
    # tokens = tokenizer.tokenize(sentence)
    tokens = tokenizer.tokenize(tokenizer.encode(sentence))
    print(tokens)
    tokenizer.encode(sentence))
   

# Get word embeddings
    # Tokenize the text into words
    # tokens = indic_tokenize.trivial_tokenize(source[0])
    # print(tokens)


    # Create a list to store word embeddings and corresponding words
    word_embeddings = []
    words = []

  
    # Convert the token to token ID
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # Specify the sequence length (e.g., the number of tokens in the input)
    # Prepare inputs for the model
    inputs =   tf.convert_to_tensor([input_ids])
    # inputs = tf.constant([input_ids])
        # Print the shape of the inputs tensor for debugging
    print("Shape of inputs:", inputs.shape)
    # inputs = inputs.squeeze(0)
    
    # Get the model's output
    with tf.device('/GPU:0'):
        outputs = model(inputs)            
        # outputs = model(inputs=inputs,attention_mask=attention_mask)

    # Extract the hidden states (word embeddings) from the output
    # hidden_states = outputs.last_hidden_state.numpy()[0]
    hidden_states = outputs[0].numpy()[0]
    for i, token in enumerate(tokens):
        if token in ('[CLS]', '[SEP]', '[PAD]', '[UNK]'):
            continue
        words.append(token)
        word_embeddings.append(hidden_states[i])


    # Append the word and its embedding to the lists
    # word_embeddings.append(hidden_states)
    # words.append(tokens)
    input_shape = (1, len(tokens))


    return words, word_embeddings
    

word_to_index = {}
index_to_word = {}
embedding_file = open(embedding_file_path, 'w', encoding='utf-8')
indexed_embedding_file = open(indexed_embedding_file_path, 'w', encoding='utf-8')
embedding_indexes_file = open(embedding_indexes_file_path, 'w', encoding='utf-8')
# Create a list to store word embeddings
# all_word_embeddings = []
idx = 1

# for sentence in sent_list:
#     words, embeddings = get_word_embeddings(sentence)

#     for word, embedding in zip(words, embeddings):
#         # Store word embeddings
#         all_word_embeddings.append(embedding)

#         # Store word-to-index and index-to-word mappings
#         if word not in word_to_index:
#             index = len(word_to_index)
#             word_to_index[word] = index
#             index_to_word[index] = word
for sentence in sent_list:
        words, embeddings = get_word_embeddings(sentence)
        # print(type(words))
        # print(type(embeddings))

        # Write word and its corresponding embeddings to the file
        for word, embedding in zip(words, embeddings):
            # Convert the embedding to a comma-separated string
            # embedding_str = " ".join(map(str, embedding))
            word =' '.join(word)
            embedding_file.write(word + ' '+ ' ' .join(map(str, np.array(embedding).flatten())) + "\n")
            # embedding_file.write(word + ' ' + ' '.join(map(str, embedding[word])) + '\n')
            indexed_embedding_file.write(' '.join(map(str, np.array(embedding).flatten())) + "\n")
            # indexed_embedding_file.write(' '.join(map(str, embedding[word])) + '\n')
            embedding_indexes_file.write(word + ' ' + str(idx) + '\n')
            idx += 1
           

            
print(f"Word embeddings with words saved to {embedding_file}")
print(f"Word emdedding saved to {indexed_embedding_file}")
print(f"Word along with  with index saved to {indexed_embedding_file}")
embedding_file.close()
indexed_embedding_file.close()
embedding_indexes_file.close()
# Define a function to get word embeddings for a sentence
# def get_word_embeddings(sentence):
#     # Tokenize the input sentence and convert it to token IDs
#     tokens = tokenizer.tokenize(sentence)
#     input_ids = tokenizer.convert_tokens_to_ids(tokens)

#     # Add special tokens [CLS] and [SEP]
#     input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]

#     # Pad or truncate to a fixed length
#     max_length = 128
#     if len(input_ids) < max_length:
#         input_ids = input_ids + [tokenizer.pad_token_id] * (max_length - len(input_ids))
#     else:
#         input_ids = input_ids[:max_length]

#     # Convert the input_ids to a TensorFlow tensor
#     input_ids = tf.constant([input_ids])

#     # Get the model's output
#     with tf.device('/GPU:0'):  # You can change this to GPU if available
#         outputs = model(input_ids)

#     # Get the hidden states (word embeddings) from the model
#     hidden_states = outputs[0]

#     # Convert the hidden states to a NumPy array
#     embeddings = hidden_states.numpy()[0]

#     return embeddings

# Example sentences
# sentences = ["Hello, how are you?", "I like pizza.", "This is a test sentence."]

# Define the output file path
# output_file = "word_embeddings.txt"
# indexed_embedding_file = open(indexed_embedding_file_path, 'w', encoding='utf-8')
# embedding_indexes_file = open(embedding_indexes_file_path, 'w', encoding='utf-8')
# embedding_file = open(embedding_file_path, 'w', encoding='utf-8')
# indexed_embedding_file = open(indexed_embedding_file_path, 'w', encoding='utf-8')
# embedding_indexes_file = open(embedding_indexes_file_path, 'w', encoding='utf-8')

# # Open the output file for writing
# with open(embedding_file, "w", encoding="utf-8") as f:
#     # Iterate through sentences and get word embeddings
#     for sentence in sent_list:
#         embeddings = get_word_embeddings(sentence)

#         # Write word embeddings to the file
#         for embedding in embeddings:
#             embedding_str = " ".join(map(str, embedding))
#             embedding_file.write(embedding_str + "\n")

# print(f"Word embeddings saved to {embedding_file}")
#  embedding_file.write(word + ' ' + ' '.join(map(str, embedding[word])) + '\n')
#             indexed_embedding_file.write(' '.join(map(str, embedding[word])) + '\n')
#             # indexed_embedding_file.write(' '.join(map(str, embedding[word])) + '\n')
#             embedding_indexes_file.write(word + ' ' + str(idx) + '\n')

# # Tokenize Hindi sentences
# # tokenized_sentences = [tokenizer.tokenize(sentence, add_special_tokens=False) for sentence in hindi_sentences]  
# # for sent in sent_list: 
# #     print(sent) 
# #     words=sent.split()
# #     print(words) 
# #     for word in words:   
# #         print(word)    
# #         if word not in embedding:
# #             if word in word_vector.wv.vocab:
# #                 embedding[word] = word_vector[word]
# #             else:
# #                 embedding[word] = np.random.rand(vector_size) * (max_val - min_val) + min_val

# #             embedding_file.write(word + ' ' + ' '.join(map(str, embedding[word])) + '\n')
# #             indexed_embedding_file.write(' '.join(map(str, embedding[word])) + '\n')
# #             # indexed_embedding_file.write(' '.join(map(str, embedding[word])) + '\n')
# #             embedding_indexes_file.write(word + ' ' + str(idx) + '\n')
# #             # embedding_indexes_file.write(word + ' ' + str(word_vector.wv.vocab[word].index ) + '\n')

# #             idx += 1
# # embedding_file.close()
# indexed_embedding_file.close()
# embedding_indexes_file.close()

# 	word2vec_model = load_word_embeddings(embeddings_file_path)
# 	#get a list of vocab
# 	#index_to_word = list(word2vec_model .wv.vocab.keys())
# 	#word_vectors = list()
# 	# Populate matrix of word vectors
#     #for word in index_to_word:
#      #   word_vectors.append(word2vec_model[word])
# 	if word in word2vec_model.wv.vocab:
# 		return word2vec_model[word]
# 	else:
# 		return np.zeros(VECTOR_DIM)
# def word2idx(word):

# 	return word2vec_model.wv.vocab[word].index 

# def idx2word(idx):

#     return word2vec_model.wv.index2word[idx]



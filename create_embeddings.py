import numpy as np
import gensim
import sys
import pandas as pd
#from gensim.models.keyedvectors import KeyedVectors
filename = 'document_id.csv'

np.random.seed(26061997)

embeddings_file_path1 ='C:\Users\saivi\Desktop\singletonmention_22\helper_files\word2vec/hi.bin'

# word_vector = Word2Vec.load('/home/kusum/Desktop/singletonmention/helper_files/word2vec/hi.bin')
# word2vec_model = gensim.models.Word2Vec.load(embeddings_file_path)
word_vector = gensim.models.Word2Vec.load(embeddings_file_path1)

# data = ET.parse('data/full.xml')
embedding_file_path = 'helper_files/embedding/generated_embedding.txt'
indexed_embedding_file_path = 'helper_files/embedding/indexed_embedding.txt'
embedding_indexes_file_path = 'helper_files/embedding/embedding_indexes.txt'
doc_tagged = pd.read_csv(filename, sep =" ", encoding='utf-8', usecols=['document_id','sentence_id','sentence']) 
doc_tagged = doc_tagged.fillna(method='ffill')
sent_list= doc_tagged['sentence'].tolist()
# sent_list = list(set(doc_tagged['sentence'].values))

min_val = min(map(min, word_vector.wv.vectors))
max_val = max(map(max, word_vector.wv.vectors))

vector_size = word_vector.wv.vector_size

embedding_file = open(embedding_file_path, 'w', encoding='utf-8')
indexed_embedding_file = open(indexed_embedding_file_path, 'w', encoding='utf-8')
embedding_indexes_file = open(embedding_indexes_file_path, 'w', encoding='utf-8')
embedding = {}
idx = 1
  
for sent in sent_list: 
    print(sent) 
    words=sent.split()
    print(words) 
    for word in words:   
        print(word)    
        if word not in embedding:
            if word in word_vector.wv.vocab:
                embedding[word] = word_vector[word]
            else:
                embedding[word] = np.random.rand(vector_size) * (max_val - min_val) + min_val

            embedding_file.write(word + ' ' + ' '.join(map(str, embedding[word])) + '\n')
            indexed_embedding_file.write(' '.join(map(str, embedding[word])) + '\n')
            # indexed_embedding_file.write(' '.join(map(str, embedding[word])) + '\n')
            embedding_indexes_file.write(word + ' ' + str(idx) + '\n')
            
            # embedding_indexes_file.write(word + ' ' + str(word_vector.wv.vocab[word].index ) + '\n')

            idx += 1
embedding_file.close()
indexed_embedding_file.close()
embedding_indexes_file.close()





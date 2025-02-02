import pandas as pd
import numpy as np
import sys
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import get_vector_words as gv 
import prepare_embeddingmatrix as embedmatrix
import get_vector_words as gv 
from collections import Counter
from sklearn.model_selection import train_test_split
embedfile_path = '/home/kusum/Desktop/singletonmention/embeddings/hi.bin'
def embeddingword(word2vec_model):
    #word2vec_model = gv.load_word_embeddings(embedfile_path)
    MAX_NB_WORDS = len(word2vec_model.wv.vocab)
    print(MAX_NB_WORDS)
    embedding_matrix = embedmatrix.create_embedmatrix(word2vec_model) 
    print("embedding matrix shape")
    print(embedding_matrix.shape[0])
    embeddingLayer = Embedding(MAX_NB_WORDS+1, VECTOR_DIM, weights=[embedding_matrix], input_length= max_len, trainable=False, mask_zero=True)
    print("embedding type")
    print(type(embedding_matrix))
    #print("embedding dimension")
    #print(embedding_matrix.shape[0],embedding_matrix.shape[1])
    return embeddingLayer, MAX_NB_WORDS
# class SentenceGetter(object):
    
#     def __init__(self, data):
#         self.n_sent = 1
#         self.data = data
#         self.empty = False
#         agg_func = lambda s: [(w,p,t) for w,p,t in zip(s['Word'].values.tolist(),s['POS'].values.tolist(),s['MENT_Label'].values.tolist())]
#         self.grouped = self.data.groupby("Sentence_num").apply(agg_func)
#         self.sentences = [s for s in self.grouped]
    
#     def get_next(self):
#         try:
#             s = self.grouped["sentence:{}".format(self.n_sent)]
#             self.n_sent += 1
#             return n_sent
#         except:
#             return None

def preparedata(filename1,filename2):
    document = pd.read_csv(filename1, sep =" ", encoding='utf-8', usecols=['document_id','sentence_id','sentence']) 
    document = markable.fillna(method='ffill')
    markable = pd.read_csv(filename2, sep =" ", encoding='utf-8', usecols=['document_id','sentence_id','mention_id','mention','POS_mention','is_pronoun','Next_Words','Prev_Words']) 
    markable = markable.fillna(method='ffill')
    # document_id	sentence_id	mention_id	mention	POS_mention	is_pronoun	Next_Words	Prev_Words


#     words = list(set(ment_tagged['Word'].values))
#     words.append('ENDPAD')
#     n_words = len(words)
#    # print(n_words)
#     pos_tags = list(set(ment_tagged['POS'].values)) 
#     n_pos_tags = len(pos_tags)
#     #print("Number of pos tag:", n_pos_tags)
#     tags = list(set(ment_tagged['MENT_Label'].values))
#     n_tags = len(tags)
#     print("Number of label:", n_tags)
#     getter = SentenceGetter(ment_tagged)  
#     sentences = getter.sentences
#     # print(getter.get_next())
#     # print(sentences)
    markables.is_proper_name = markables.is_proper_name.map(int)
    markables.is_first_person = markables.is_first_person.map(int)
    markables.is_pronoun = markables.is_pronoun.map(int)
    markables.is_singleton = markables.is_singleton.map(
        lambda x: to_categorical(x, num_classes=2))
    markables.is_antecedentless = markables.is_antecedentless.map(
        lambda x: to_categorical(x, num_classes=2))
    max_len = 100
    max_len_char = 20
    tag2idx = {t: i+1 for i, t in enumerate(tags)}
    tag2idx["PAD"] = 0
    #print(tag2idx)
    X = [[w[0] for w in s] for s in sentences]
    #print("before padding X:")
    #print(X)
    new_X = []
    for seq in X:
        new_seq = []
        for i in range(max_len):
            try:
                new_seq.append(seq[i])
            except:
                new_seq.append("PAD")
        new_X.append(new_seq)
    X = new_X
    #print("after padding X:")
    #print(X)
    print("get word vector index from pretrained wordvector")
    
    VECTOR_DIM = 300
    word2vec_model = gv.load_word_embeddings(embedfile_path)
    MAX_NB_WORDS = len(word2vec_model.wv.vocab)
    #PretrainedwordembedLayer, MAX_NB_WORDS = embeddingword(word2vec_model)
    #MAX_NB_WORDS = len(word2vec_model.wv.vocab)
    #print(PretrainedwordembedLayer)
    #print(word_index)
    print(MAX_NB_WORDS)
    #word,pos_tag,label,n_pos_tags,n_tags = pdata.preparedata(input_file)
    #print(len(pos_tag[0]))
    #print("number of pos tag")
    #print(n_pos_tags)
    #print(word)
    #print(pos_tag)
    # print(label)
    '''for w in word:
        for w1 in w:  
            print(w1)
    '''
    #X_word = [[word_index[w] for w in s] for s in word]
    #sequences = [[word_index.get(t, 0) for t in comment] for comment in comments[:len(list_sentences_train)]]
    word_index = {t[0]: i+2 for i,t in enumerate(Counter(word2vec_model.wv.vocab).most_common(MAX_NB_WORDS))}
    word_index["UNK"] = 1
    word_index["PAD"] = 0
    X_word = [[word_index.get(w,0) for w in s] for s in X]
    
    #print("Before padding:\n")
    #print(X_word)
    X_word = pad_sequences(maxlen=max_len, sequences=X_word, value=word_index["PAD"], padding='post', truncating='post')
    idx2word = {i: w for w, i in word_index.items()}

    # print("X_word")
    # print(X_word.shape)
    #print("after padding:\n")
    #print(X_word)
    # splitting data into train and test.
    #X_word_train, X_word_test, pos_tag_train, pos_tag_test, y_train, y_test = split_data_train_test(X_word, pos_tag, label)
    
    chars = set([w_i for w in words for w_i in w])
    n_chars = len(chars)
    print(n_chars)
    char2idx = {c: i + 2 for i, c in enumerate(chars)}
    char2idx["UNK"] = 1
    char2idx["PAD"] = 0
    X_char = []
    for sentence in sentences:
        sent_seq = []
        for i in range(max_len):
            word_seq = []
            for j in range(max_len_char):
                try:
                    word_seq.append(char2idx.get(sentence[i][0][j]))
                except:
                    word_seq.append(char2idx.get("PAD"))
            sent_seq.append(word_seq)
        X_char.append(np.array(sent_seq))
    #print(X_char)
    

    pos_tag2idx = {p: j+1 for j, p in enumerate(pos_tags)}
    pos_tag2idx["PAD"]=0
    print(len(pos_tag2idx))
    print(pos_tag2idx)
    idx2tag = {i: w for w, i in tag2idx.items()}
    posT = [[w[1] for w in s] for s in sentences] 
    # print("Before padding:")
    # print(posT)
    new_posT = []
    for seq1 in posT:
        new_seq1 = []
        for i in range(max_len):
            try:
                new_seq1.append(seq1[i])
            except:
                new_seq1.append("PAD")
        new_posT.append(new_seq1)
    pos_tags = new_posT
    # print("After padding pos:")
    # print(pos_tags)
    #print(pos_tag2idx)
   
    #print(pos_tag2idx)
    pos_tag = [[pos_tag2idx[w[1]] for w in s] for s in sentences]
    #print(pos_tag)
    pos_tags = pad_sequences(maxlen=max_len, sequences=pos_tag, padding="post", value=pos_tag2idx["SYM"])
    #print(len(pos_tags))
    #POST = [to_categorical(i, num_classes=n_pos_tags+1) for i in pos_tags]
    #print(len(POST[0]))
    #print( POST.size())
    y = [[tag2idx[w[2]] for w in s] for s in sentences]
    
    #print(y)
    Y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["PAD"])
    #print("afterpadding")
    #print(y)
    # Making labels in one hot encoded form for DL model
    #y = [to_categorical(i, num_classes=n_tags) for i in Y]
    return X_word,X_char,pos_tags,Y,n_pos_tags,n_tags,n_chars,idx2word,idx2tag
    #print(y)

   
    
 
    #maxlen = max([len(s) for s in sentences])
    #print ('Maximum sequence length:', maxlen)
   
# if __name__ == '__main__':
    
#     input_file = sys.argv[1]
#     X_word,X_char,pos_tags,Y,n_pos_tags,n_tags,n_chars,idx2word, idx2tag= preparedata(input_file)
    #print(word)
    #print(pos_tag)
    #print(label)
    #print(n_pos_tags)
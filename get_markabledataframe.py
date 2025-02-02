from csv import DictReader
from string import punctuation
from typing import Callable, Dict, List, Tuple
import numpy as np
import pandas as pd
import tensorflow
from tensorflow.keras.utils import to_categorical

# from keras.utils import to_categorical
import os
os.environ['KERAS_BACKEND']='tensorflow'
import snowballstemmer

stemmer = snowballstemmer.stemmer('hindi')

# print(stemmer.stemWords("We are the world".split()));


def is_number(word: str) -> bool:
    # return word.replace(',', '.').replace('.', '@',1).replace('-', '@', 1).isdigit()
    return word.replace(',', '.').replace('.', '').replace('-', '', 1).isdigit()


def get_word(word: str) -> str:
    # if '\\' in word:
    #     word = word.split('\\')[0]
    # print("word")

    # print(word)
    # while word[-1] in punctuation and len(word) > 1:
    #     word = word[:-1]
    #     print(word)
    # while word[0] in punctuation and len(word) > 1:
    #     word = word[1:]
    #     print(word)

    word = word.lower()
    # print(word)

    return word


# def get_words_only(text: str) -> str:
#     words = text.split()
#     words = map(get_word, words)
#     return ' '.join(words)


# def get_abbreviation(text: str) -> str:
#     words = get_words_only(text).split()
#     abb = ''

#     for word in words:
#         abb += word[0]

#     return abb

def clean_word(word: str, word_vector: Dict[str, np.array]) -> str:
    word = get_word(word)
    # print(word)

    # if word not in word_vector:
    #     tmp = word.split('-')
        
    #     if len(tmp) == 2 and tmp[0] == tmp[1]:
    #         word = tmp[0]
    #         print('......')
    #         print(word)

    # if word not in word_vector:
    #     print("stemmer")
    #     word = stemmer.stemWord(word)
        # word = stemmer.stem(word)

    # if word not in word_vector:
    #     tmp = word.split('-')
    #     if len(tmp) == 2 and tmp[0] == tmp[1]:
    #         word = tmp[0]
    #         print('******')

    if word not in word_vector:
        word = stemmer.stemWord(word)
        # print('//////')
        # print(word)

    # if is_number(word):
    #     word = '<अंक>'

    return word


def clean_sentence(sentence: str, word_vector: Dict[str, np.array]) -> str:
    return ' '.join([clean_word(word, word_vector) for word in sentence.split() if clean_word(word, word_vector) != ''])
    


# def clean_arr(arr: List[str], word_vector: Dict[str, np.array]) -> List[str]:
    # return [clean_word(word, word_vector) for word in arr if clean_word(word, word_vector) != '']

def to_sequence(textm: str, idx_by_word: Dict[str, int]) -> List[int]:
    textm = textm.split()
    return list(map(lambda word: idx_by_word[word], textm))

def get_markable_dataframe(markable_file: str, word_vector: Dict[str, np.array],
                           idx_by_word: Dict[str, int]) -> pd.DataFrame:
    markables = pd.read_csv(markable_file, sep=" ", encoding='utf-8')
    # print("heloo")
    # print(markables.columns.tolist())
    # print(get_entity_types(markables.entity_type))
    markables.mention = markables.mention.fillna("").map(lambda x: to_sequence(clean_sentence(str(x), word_vector), idx_by_word))
    markables.is_proper_name = markables.is_proper_name.map(int)                                                                        
    markables.is_pronoun = markables.is_pronoun.map(int)
    # markables.entity_type = markables.entity_type.map(entity_to_bow(get_entity_types(markables.entity_type)))
    # markables.entity_type = markables.entity_type.map(entity_to_bow(['EVENT', 'FACILITY', 'LOCATION', 'NUM', 'ORGANIZATION', 'OTHER', 'PERSON', 'THINGS', 'TIME', 'TITLE']))
    markables.is_first_person = markables.is_first_person.map(int)
    print(markables.previous_words)
    markables.previous_words = markables.previous_words.fillna("").map(lambda x: to_sequence(clean_sentence(str(x), word_vector), idx_by_word))
    markables.next_words = markables.next_words.fillna("").map(lambda x: to_sequence(clean_sentence(str(x), word_vector), idx_by_word))
    # markables.all_previous_words = markables.all_previous_words.fillna("").map(lambda x: to_sequence(clean_sentence(str(x), word_vector), idx_by_word))
    markables.is_singleton = markables.is_singleton.map(lambda x: to_categorical(x, num_classes=2))
    # markables.is_antecedentless = markables.is_antecedentless.map(lambda x: to_categorical(x, num_classes=2))

    return markables
    

def get_embedding_variables(embedding_indexes_file_path: str,
                            indexed_embedding_file_path: str) \
        -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, int], Dict[int, str]]:
    word_vector = {}
    embedding_matrix = []
    idx_by_word = {}
    word_by_idx = {}

    for element in open(embedding_indexes_file_path, 'r', errors="ignore", encoding="utf-8").readlines():
        element = element.split()
        # print(element[0])
        # print(element[1])
        word, index = element[0], int(element[1])
        idx_by_word[word] = index
        word_by_idx[index] = word

    for element in open(indexed_embedding_file_path, 'r').readlines():
        element = element.split()

        embedding = np.asarray(element, dtype='float64')

        if len(embedding_matrix) == 0:
            embedding_matrix.append(np.zeros(embedding.shape))

        index = len(embedding_matrix)
        word_vector[word_by_idx[index]] = embedding

        embedding_matrix.append(embedding)

    embedding_matrix = np.array(embedding_matrix)

    return word_vector, embedding_matrix, idx_by_word, word_by_idx
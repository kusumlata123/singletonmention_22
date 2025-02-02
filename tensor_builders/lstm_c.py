from typing import List

from tensorflow.keras.layers import Embedding, Dense, Reshape
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.framework.ops import Tensor
from .base import BaseTensorBuilder
# class LSTMTensorBuilder(BaseTensorBuilder):
#     variables = ['vocab_size', 'vector_size', 'embedding_matrix', 'trainable_embedding', 'output_size']

#     def __init__(self, vocab_size: int = None, vector_size: int = None, embedding_matrix: List[List[int]] = None,
#                  trainable_embedding: bool = False, output_size: int = 16, *args, **kwargs):
#         super().__init__(input_shape=(None,), *args, **kwargs)

#         self.output_size = output_size
#         self.trainable_embedding = trainable_embedding
#         self.embedding_matrix = embedding_matrix
#         self.vector_size = vector_size
#         self.vocab_size = vocab_size

#     def create_processing_tensor(self, input_tensor: Tensor) -> Tensor:
#         tensor = Embedding(self.vocab_size, self.vector_size, weights=[self.embedding_matrix],
#                            trainable=self.trainable_embedding)(input_tensor)
#         tensor = LSTM(self.output_size)(tensor)
#         tensor = Dense(self.output_size, activation='relu')(tensor)

#         return tensor

from typing import List

from tensorflow.keras.layers import Embedding, Dense
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.keras.layers import LSTM

from .base import BaseTensorBuilder
# import tensorflow as tf

class LSTMTensorBuilder(BaseTensorBuilder):
    variables = ['vocab_size', 'vector_size', 'embedding_matrix', 'trainable_embedding', 'output_size']

    def __init__(self, vocab_size: int = None, vector_size: int = None, embedding_matrix: List[List[int]] = None,
                 trainable_embedding: bool = False, output_size: int = 16, *args, **kwargs):
        super().__init__(input_shape=(None,), *args, **kwargs)

        self.output_size = output_size
        self.trainable_embedding = trainable_embedding
        self.embedding_matrix = embedding_matrix
        self.vector_size = vector_size
        self.vocab_size = vocab_size

    def create_processing_tensor(self, input_tensor: Tensor) -> Tensor:
        tensor = Embedding(self.vocab_size, self.vector_size, weights=[self.embedding_matrix],
                           trainable=self.trainable_embedding)(input_tensor)
        tensor = LSTM(self.output_size)(tensor)
        tensor = Dense(self.output_size, activation='relu')(tensor)

        return tensor






# class LSTMTensorBuilder(BaseTensorBuilder):
#     variables = ['output_size','dropout']

#     def __init__(self, output_size: int = 16,dropout: float = 0, *args, **kwargs):
#         super().__init__(input_shape=(None,), *args, **kwargs)

#         self.output_size = output_size
#         self.dropout = dropout

        

#     def create_processing_tensor(self, input_tensor: Tensor) -> Tensor:
#         tensor = input_tensor
#         print(tensor.shape)
#         # tensor = tensor.reshape(1, 10, 2)
        
#         tensor = Reshape(tensor.shape,1)(tensor)
    
#         tensor = LSTM(self.output_size)(tensor)
#         # tensor = Dense(self.output_size, activation='relu')(tensor)
#         tensor = Dropout(self.dropout)(tensor)
#         tensor = Dense(self.output_size, activation='relu')(tensor)
#         # for i in range(len(self.ouput_size)):
#         #     tensor = Dense(self.output_size[i], activation='relu')(tensor)

#         #     if i < len(self.output_size) - 1 and self.dropout > 0:
#         #         tensor = Dropout(self.dropout)(tensor)

#         return tensor

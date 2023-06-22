


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense

class BidirectionalRNN(Model):
    def __init__(self, word_index, embedding_matrix, max_len):
        super(BidirectionalRNN, self).__init__()

        self.word_index = word_index
        self.embedding_matrix = embedding_matrix
        self.max_len = max_len

        self.build_model()

    def build_model(self):
        self.embedding = Embedding(len(self.word_index) + 1, 300,
                                   weights=[self.embedding_matrix],
                                   input_length=self.max_len,
                                   trainable=False)
        self.lstm = Bidirectional(LSTM(300, dropout=0.3, recurrent_dropout=0.3))
        self.dense = Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.lstm(x)
        outputs = self.dense(x)
        return outputs

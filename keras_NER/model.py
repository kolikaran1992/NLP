from keras.layers import Dense, LSTM, Bidirectional, Embedding, Input, Dropout, TimeDistributed
from keras.layers.merge import Concatenate
from keras.models import Model
from keras_contrib.layers import CRF
from .__common__ import DTYPE, LOGGER_NAME
import logging
logger = logging.getLogger(LOGGER_NAME)

class BiLSTMCRF(object):
    """
    --> A Keras implementation of BiLSTM-CRF for sequence labeling.
    """

    def __init__(self,
                 num_labels,
                 word_vocab_size,
                 char_vocab_size=None,
                 word_embedding_dim=100,
                 char_embedding_dim=25,
                 word_lstm_size=100,
                 char_lstm_size=25,
                 fc_dim=100,
                 fc_act='tanh',
                 dropout=0.5,
                 embeddings=None,
                 use_char=True,
                 use_crf=True):
        """
        --> Build a Bi-LSTM CRF model.

        :param word_vocab_size (int): word vocabulary size.
        :param char_vocab_size (int): character vocabulary size.
        :param num_labels (int): number of entity labels.
        :param word_embedding_dim (int): word embedding dimensions.
        :param char_embedding_dim (int): character embedding dimensions.
        :param word_lstm_size (int): character LSTM feature extractor output dimensions.
        :param char_lstm_size (int): word tagger LSTM output dimensions.
        :param fc_dim (int): output fully-connected layer size.
        :param fc_act (str): output fully-connected activation
        :param dropout (float): dropout rate.
        :param embeddings (numpy array): word embedding matrix.
        :param use_char (boolean): add char feature.
        :param use_crf (boolean): use crf as last layer.
        """
        super(BiLSTMCRF).__init__()
        self._char_embedding_dim = char_embedding_dim
        self._word_embedding_dim = word_embedding_dim
        self._char_lstm_size = char_lstm_size
        self._word_lstm_size = word_lstm_size
        self._char_vocab_size = char_vocab_size
        self._word_vocab_size = word_vocab_size
        self._fc_dim = fc_dim
        self._fc_act = fc_act
        self._dropout = dropout
        self._use_char = use_char
        self._use_crf = use_crf
        self._embeddings = embeddings
        self._num_labels = num_labels

    def build(self):
        # build word embedding
        logger.info('{} :: building the bilstm model'.format(self.__class__.__name__))
        word_ids = Input(batch_shape=(None, None), dtype=DTYPE['int'], name='word_input')
        inputs = [word_ids]
        if self._embeddings is None:
            word_embeddings = Embedding(input_dim=self._word_vocab_size,
                                        output_dim=self._word_embedding_dim,
                                        mask_zero=True,
                                        name='word_embedding')(word_ids)
            logger.info('{} :: initializing word embedding with random weights'.format(self.__class__.__name__))
        else:
            word_embeddings = Embedding(input_dim=self._embeddings.shape[0],
                                        output_dim=self._embeddings.shape[1],
                                        mask_zero=True,
                                        weights=[self._embeddings],
                                        name='word_embedding')(word_ids)
            logger.info('{} :: initializing word embedding with the weights provided'.format(self.__class__.__name__))

        # build character based word embedding
        if self._use_char:
            char_ids = Input(batch_shape=(None, None, None), dtype=DTYPE['int'], name='char_input')
            inputs.append(char_ids)
            char_embeddings = Embedding(input_dim=self._char_vocab_size,
                                        output_dim=self._char_embedding_dim,
                                        mask_zero=True,
                                        name='char_embedding')(char_ids)
            char_embeddings = TimeDistributed(Bidirectional(LSTM(self._char_lstm_size)))(char_embeddings)
            word_embeddings = Concatenate()([word_embeddings, char_embeddings])
            logger.info('{} :: using char embedding'.format(self.__class__.__name__))

        word_embeddings = Dropout(self._dropout)(word_embeddings)
        z = Bidirectional(LSTM(units=self._word_lstm_size, return_sequences=True))(word_embeddings)
        z = Dense(self._fc_dim, activation=self._fc_act)(z)

        if self._use_crf:
            crf = CRF(self._num_labels)
            loss = crf.loss_function
            pred = crf(z)
            logger.info('{} :: using CRF as the final layer'.format(self.__class__.__name__))
        else:
            loss = 'categorical_crossentropy'
            pred = Dense(self._num_labels, activation='softmax')(z)
            logger.info('{} :: using fully connected layer with "softmax activation" and "categorical crossentropy" '
                        'loss as final layer'.format(self.__class__.__name__))

        model = Model(inputs=inputs, outputs=pred)

        return model, loss
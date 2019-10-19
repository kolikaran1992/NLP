from .vocabulary import Vocabulary
from .__common__ import LOGGER_NAME
import logging
from .__utils__ import read_label_file
logger = logging.getLogger(LOGGER_NAME)
from gensim.models import KeyedVectors
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from collections import namedtuple
encoded_seq = namedtuple('EncodedSequence', 'tok_ids char_ids lab_ids')
decoded_seq = namedtuple('DecodedSequence', 'tokens labels')

class SequenceEncoder(object):
    """
    --> encode input text sequence to a sequence of numbers
    """

    def __init__(self,
                 path_to_w2v='',
                 path_to_label_vocab='',
                 char=False,
                 max_seq_len=50,
                 max_word_len=10):
        """
        :param path_to_w2v: path to vectors (word2vec format)
        :param path_to_label_vocab: path to labels file
        :param char: whether to use characters
        :param max_seq_len: maximum sequence length
        :param max_word_len: maximum word length
        """
        self._max_seq_len = max_seq_len
        self._max_word_len = max_word_len

        _temp = KeyedVectors.load_word2vec_format(path_to_w2v)

        self._word_vocab = Vocabulary(vocab=list(_temp.wv.vocab.keys()))
        self._word_vectors = _temp.wv
        logger.info('{} :: word vocabulary size = {}'.format(self.__class__.__name__, len(self._word_vocab)))

        labels = read_label_file(path_to_label_vocab, logger)
        self._label_to_1hot = OneHotEncoder()
        self._label_to_1hot.fit(np.array([labels + ['<pad>']]).reshape(-1,1))
        logger.info('{} :: label vocabulary size = {}'.format(self.__class__.__name__, len(labels)+1))

        self._use_char = char
        if char:
            logger.info('{} :: using character encoding'.format(self.__class__.__name__))
            all_chars = list(set([ch for tok in _temp.wv.vocab.keys() for ch in tok]))
            self._char_vocab = Vocabulary(vocab=all_chars)
            logger.info('{} :: char vocabulary size = {}'.format(self.__class__.__name__, len(self._char_vocab)))

        del _temp

    def encode(self, tokens, labels = None):
        """
        --> encoding tokens to ids
        --> pad to maximum len
        :param tokens: list
        :param labels: list
        :return: named tuple
        """
        tokens = self._word_vocab.pad_sequence(tokens, self._max_word_len)
        tok2ids = self._word_vocab.doc2id(tokens)

        char2ids = None
        lab2ids = None

        if labels:
            lab2ids = self._label_to_1hot.transform(np.array([labels]).reshape(-1, 1)).todense().astype(int).tolist()
            lab2ids.extend([self._label_to_1hot.transform([['<pad>']]).todense().astype(int).tolist()[0]] * (self._max_seq_len - len(lab2ids)))
        if self._use_char:
            char2ids = [self._char_vocab.doc2id(self._char_vocab.pad_sequence(list(tok), self._max_word_len)) for tok in tokens] + [
                self._char_vocab.doc2id(self._char_vocab.pad_sequence(['<pad>'], self._max_word_len))] * (self._max_seq_len - len(tokens))

        return encoded_seq(tok_ids=tok2ids, char_ids=char2ids, lab_ids=lab2ids)



    def decode(self, enc_seq):
        """
        --> named tuple containing token ids, char ids, label ids
        --> decode token ids and label ids
        --> remove pad characters
        :param enc_seq: named tuple
        :return: named tuple
        """
        tokens = self._word_vocab.id2doc(enc_seq.tok_ids)
        labels = self._label_to_1hot.inverse_transform(enc_seq.lab_ids).reshape(1,-1).tolist()[0]

        return decoded_seq(tokens=list(filter(lambda x: x != '<pad>', tokens)),
                           labels=list(filter(lambda x: x != '<pad>', labels)))


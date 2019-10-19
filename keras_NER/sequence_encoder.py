from .vocabulary import Vocabulary
from .__common__ import LOGGER_NAME
import logging
from .__utils__ import read_label_file
logger = logging.getLogger(LOGGER_NAME)
from gensim.models import KeyedVectors

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
                 max_tok_len=50):
        """
        :param path_to_w2v: path to vectors (word2vec format)
        :param path_to_label_vocab: path to labels file
        :param char: whether to use characters
        :param max_tok_len: maximum sequence length
        """
        self._max_tok_len = max_tok_len

        _temp = KeyedVectors.load_word2vec_format(path_to_w2v)

        self._word_vocab = Vocabulary(vocab=list(_temp.wv.vocab.keys()))
        self._word_vectors = _temp.wv
        logger.info('{} :: word vocabulary size = {}'.format(self.__class__.__name__, len(self._word_vocab)))

        labels = read_label_file(path_to_label_vocab, logger)
        self._label_vocab = Vocabulary(vocab=labels)
        logger.info('{} :: label vocabulary size = {}'.format(self.__class__.__name__, len(self._label_vocab)))

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

        tok2ids = self._word_vocab.doc2id(tokens)
        char2ids = None
        lab2ids = None
        tok2ids.extend([self._word_vocab.token_to_id('<pad>')] * (self._max_tok_len - len(tok2ids)))
        if labels:
            lab2ids = self._label_vocab.doc2id(labels)
            lab2ids.extend([self._label_vocab.token_to_id('<pad>')] * (self._max_tok_len - len(lab2ids)))

        if self._use_char:
            char2ids = [self._char_vocab.doc2id(list(tok)) for tok in tokens] + [
                [self._char_vocab.token_to_id('<pad>')]] * (self._max_tok_len - len(tokens))

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
        labels = self._label_vocab.id2doc(enc_seq.lab_ids)

        return decoded_seq(tokens=list(filter(lambda x: x != '<pad>', tokens)),
                           labels=list(filter(lambda x: x != '<pad>', labels)))


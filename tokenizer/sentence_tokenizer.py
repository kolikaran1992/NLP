from tokenizer.__common_parameters__ import LOGGER_NAME

import logging
logger = logging.getLogger(LOGGER_NAME)

import re
def regexp_span_tokenize(s, regexp):
    """
    --> nltk's regex_span_tokenize
    --> the only difference, this function operates on spans of the 1st or second group in the match
    :param s:
    :param regexp:
    :return:
    """
    left = 0
    for m in re.finditer(regexp, s):
        right, next = m.span(1)
        if right == next == -1:
            right, next = m.span(2)
        if right != left:
            yield left, right
        left = next
    yield left, len(s)

class SentenceTokenizer(object):
    def __init__(self,
                 gap_re = r'[^\sA-Z\d]+\s*([\.] *)[A-Z]',
                 flags=re.UNICODE | re.MULTILINE | re.DOTALL):

        self._gap_pattern = gap_re
        self._gap_regex = None
        self._flags = flags

        logger.info('{} : Tokenizer Initialized'.format(self.__class__.__name__))

    def log_message(self, message, level):
        if level == 'info':
            logger.info('{} : {}'.format(self.__class__.__name__, message))
        if level == 'error':
            logger.error('{} : {}'.format(self.__class__.__name__, message))
        if level == 'info':
            logger.exception('{} : {}'.format(self.__class__.__name__, message))

    def __repr__(self):
        return '{}(pattern={}, flags={})'.format(
            self.__class__.__name__,
            self._gap_pattern,
            self._flags
        )

    def _check_regex(self):
            if self._gap_regex is None:
                self._gap_regex = re.compile(self._gap_pattern, self._flags)

    def _gap_split(self, text, return_spans = False):
        if len(text) == 0:
            return []
        self._check_regex()
        spans = [(left, right) for left, right in regexp_span_tokenize(text, self._gap_regex) if not (left == right)]
        tokens = [text[span[0]:span[1]] for span in spans]

        if return_spans:
            return spans
        else:
            return tokens

    def tokenize(self, text, return_spans = False):
        if len(text) == 0:
            return []
        self._check_regex()

        spans = self._gap_split(text, return_spans=True)
        tokens = [text[span[0]:span[1]] for span in spans]

        if return_spans:
            return spans
        else:
            return tokens

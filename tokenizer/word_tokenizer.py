from tokenizer.__common_parameters__ import LOGGER_NAME

import logging
logger = logging.getLogger(LOGGER_NAME)

import re

from nltk.tokenize.util import regexp_span_tokenize

class Tokenizer(object):
    def __init__(self,
                 prefix_re = '',
                 suffix_re = '',
                 infix_re = None,
                 gap_re = r' ',
                 flags=re.UNICODE | re.MULTILINE | re.DOTALL):

        self._pattern = {
            'prefix' : prefix_re,
            'suffix' : suffix_re,
            'infix' : infix_re,
            'gap' : gap_re
        }

        self._regex = {
            'prefix' : None,
            'suffix' : None,
            'infix' : None,
            'gap' : None
        }

        self._flags = flags

        logger.info('Tokenizer Initialized')

    def __repr__(self):
        return '{}(pattern={}, flags={})'.format(
            self.__class__.__name__,
            self._pattern,
            self._flags
        )

    def _check_regex(self):
        for key in ['prefix', 'suffix', 'infix', 'gap']:
            if self._regex[key] is None:
                if self._pattern[key] is not None:
                    self._regex[key] = re.compile(self._pattern[key], self._flags)

    def _infix_tokenizer(self, token):
        if self._regex['infix'] is None:
            return [(0, len(token))]
        idx = 0
        spans = []
        for item in re.finditer(self._regex['infix'], token):
            left, right = item.span()
            spans += [(idx, left), (left, right)]
            idx = right

        spans.append((idx, len(token)))

        return spans

    def _process_token(self, token, idx):
        ## prefix split

        if token == '':
            return []

        prefix_split = [item.span() for item in re.finditer(self._regex['suffix'], token) if not (item.span()[0] == item.span()[1])]

        final_tok = token[prefix_split[-1][0]: prefix_split[-1][1]]

        suffix_spans = [(len(final_tok) + prefix_split[-1][0] - item.span()[1], len(final_tok) + prefix_split[-1][0] - item.span()[0])
                            for item in re.finditer(self._regex['prefix'], final_tok[::-1]) if not (item.span()[0] == item.span()[1])]

        infix_tok = token[suffix_spans[-1][0]: suffix_spans[-1][1]]

        infix_spans = [(left + suffix_spans[-1][0] + idx, right + suffix_spans[-1][0] + idx) for left, right in
                       self._infix_tokenizer(infix_tok) if not (left == right)]

        return [(idx+item[0], idx+item[1]) for item in prefix_split[:-1]] + \
                infix_spans + \
               [(item[0] + idx, item[1] + idx) for item in reversed(suffix_spans[:-1])]


    def gap_split(self, text, return_spans = False):
        if len(text) == 0:
            return []
        self._check_regex()

        spans = [(left, right) for left, right in regexp_span_tokenize(text, self._regex['gap']) if not (left == right)]
        tokens = [text[span[0]:span[1]] for span in spans]

        if return_spans:
            return spans
        else:
            return tokens

    def tokenize(self, text, return_spans = False):
        if len(text) == 0:
            return []
        self._check_regex()

        spans = [span for left, right in self.gap_split(text, return_spans=True) for span in self._process_token(text[left: right], left)]

        tokens = [text[span[0]:span[1]] for span in spans]

        if return_spans:
            return spans
        else:
            return tokens

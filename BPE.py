from collections import defaultdict
from copy import deepcopy
import re
import joblib
import logging

logger = logging.getLogger('BPE')
formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(name)s :: %(message)s')

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

logger.setLevel(logging.INFO)

from nltk.tokenize import word_tokenize,RegexpTokenizer

def process_text(text):
    return text

def tokenize_word_old(string, sorted_tokens, unknown_token='</u>'):
    if string == '':
        return []
    if sorted_tokens == []:
        return [unknown_token]

    string_tokens = []
    for i in range(len(sorted_tokens)):
        token = sorted_tokens[i]
        token_reg = re.escape(token)  # .replace('.', '[.]'))

        matched_positions = [(m.start(0), m.end(0)) for m in re.finditer(token_reg, string)]
        if len(matched_positions) == 0:
            continue
        substring_end_positions = [matched_position[0] for matched_position in matched_positions]

        substring_start_position = 0
        for substring_end_position in substring_end_positions:
            substring = string[substring_start_position:substring_end_position]
            string_tokens += tokenize_word(string=substring, sorted_tokens=sorted_tokens[i + 1:],
                                           unknown_token=unknown_token)
            string_tokens += [token]
            substring_start_position = substring_end_position + len(token)
        remaining_substring = string[substring_start_position:]
        string_tokens += tokenize_word(string=remaining_substring, sorted_tokens=sorted_tokens[i + 1:],
                                       unknown_token=unknown_token)
        break
    return string_tokens


def tokenize_word(string, sorted_tokens, unknown_token = '<unk>'):
    if string == '':
        return []
    if sorted_tokens == []:
        return [unknown_token]

    ## find first token in the sorted token list
    ## which is a substring of the original string
    matched_positions = []

    for i in range(len(sorted_tokens)):
        token = sorted_tokens[i]
        token_reg = re.escape(token)  # .replace('.', '[.]'))

            ## length of matched_positions can be greater than 1
            ## e.g. have is appears 2 times in ihavehavethepower
        matched_positions = [(m.start(0), m.end(0)) for m in re.finditer(token_reg, string)]

        if len(matched_positions) == 0:
            continue
        else:
            break

    if len(matched_positions) == 0:
        return [unknown_token]

        ## the original string is broken in 3 parts
        ## part that matched a token in the sorted token list
        ## part left of the matched part
        ## part right of the matched part

    substring_sets = [(string[:i], string[i:j], string[j:]) for i, j in matched_positions]

        ## list of list
        ## each sublist is a list of tokens for each item in substring set
    all_possible_tokens = []

    for left, match, right in substring_sets[:1]:
        left_tokens = tokenize_word(left, sorted_tokens, unknown_token=unknown_token)
        right_tokens = tokenize_word(right, sorted_tokens, unknown_token=unknown_token)
        all_possible_tokens.append(left_tokens + [match] + right_tokens)

    return all_possible_tokens


class BPE(object):
    """
    --> Transform the base vocabulary according to BPE
    --> Convert a collection of raw documents to a ids of the transformed vocab
    --> Inverse transform the transformed ids to text
    Attributes:
        _use_char: boolean. Whether to use char feature.
        _num_norm: boolean. Whether to normalize text.
        _word_vocab: dict. A mapping of words to feature indices.
        _char_vocab: dict. A mapping of chars to feature indices.
        _label_vocab: dict. A mapping of labels to feature indices.
    """

    def __init__(self,
                 tokenizer = RegexpTokenizer(r'\w+|[^\w\s]').tokenize,
                 max_merges = 1000):
        """Create a preprocessor object.
        Args:
            lower: boolean. Whether to convert the texts to lowercase.
            use_char: boolean. Whether to use char feature.
            num_norm: boolean. Whether to normalize text.
            initial_vocab: Iterable. Initial vocabulary for expanding word_vocab.
        """
        self._base_vocab = None
        self._vocab = None
        self._tokenizer = tokenizer
        self._max_merges = max_merges
        self._end_word_tok = '<e>'
        self._start_word_tok = '<s>'
        self._vocab_tokenization = {}
        self._unk_tok = '<unk>'
        self._sorted_tokens = None

    def token_len(self, token):
        if token[-len(self._end_word_tok):] == '<e>':
            return len(token[:-len(self._end_word_tok)]) + 1
        else:
            return len(token)

    def modify_token(self, token):
        return self._start_word_tok + token + self._end_word_tok

    def _make_base_vocab(self, texts):
        """
        --> Learn base vocabulary from texts
        """
        self._base_vocab = defaultdict(int)
        for text in texts:
            tokens = self._tokenizer(text)
            for tok in tokens:
                self._base_vocab[self._start_word_tok + ' '.join(list(tok) + [self._end_word_tok])] += 1
        self._vocab = deepcopy(self._base_vocab)
        logger.info('base vocab size = {}'.format(len(self._base_vocab)))

    def _get_byte_pairs(self):
        if self._vocab is None:
            return defaultdict(int)

        pairs = defaultdict(int)
        for word, freq in self._vocab.items():
            symbols = word.split(' ')
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs

    def _merge(self):
        updated_vocab = {}
        pairs = self._get_byte_pairs()

        if len(pairs) == 0:
            logger.info('no pairs exists')
            return

        bigram = max(pairs.items(), key = lambda x: x[1])
        logger.debug('merging "{}" with freq = {}'.format(bigram[0], bigram[1]))
        regex = re.compile(re.escape(' '.join(bigram[0])))
        for word in self._vocab.keys():
            updated_word = regex.sub(''.join(bigram[0]), word)
            updated_vocab[updated_word] = self._vocab[word]

        self._vocab = deepcopy(updated_vocab)

    def fit(self):
        for i in range(self._max_merges):
            self._merge()
            logger.info('merge number {} finished'.format(i + 1))
        for tok in self._vocab.keys():
            split_ = tok.split(' ')
            self._vocab_tokenization[''.join(split_)] = split_

        self._sorted_tokens = [item for tok, _ in sorted(self.get_tokens().items(), key = lambda x: (self.token_len(x[0]), x[1]), reverse=True) for item in tok.split(' ')]

    def get_tokens(self):
        tokens = defaultdict(int)
        for word, freq in self._vocab.items():
            word_tokens = word.split(' ')
            for token in word_tokens:
                tokens[token] += freq
        logger.info('total tokens = {}'.format(len(tokens)))
        return tokens

    def encode(self, text):
        tokens = self._tokenizer(text)
        base_tokens = list(map(lambda x: self._start_word_tok + x + self._end_word_tok, tokens))
        new_tokens = []
        for token in base_tokens:
            if token in self._vocab_tokenization.keys():
                new_tokens += self._vocab_tokenization[token]
            else:
                logger.info('token {} not in vocab'.format(token))
                _temp_tokens = tokenize_word(token, self._sorted_tokens, unknown_token=self._unk_tok)
                logger.info('adding tokenization rule for {}'.format(token))
                self._vocab_tokenization[token] = _temp_tokens
                new_tokens += _temp_tokens
        return new_tokens

    @property
    def word_vocab_size(self):
        return len(self._vocab)

    def save(self, file_path):
        joblib.dump(self, file_path)

    @classmethod
    def load(cls, file_path):
        p = joblib.load(file_path)
        return p

from nltk import FreqDist
from nltk.corpus import stopwords

stops = stopwords.words('english')


class TokenDist(object):
    def __init__(self, toks,
                 rev=True,
                 tokenizer=None,
                 stops=None):
        self._rev = rev
        self._toks = toks
        self._freq_dict = None
        self._tokenizer = tokenizer
        self._stops = stops

    def get_freq_dict(self):
        if self._freq_dict is None:
            fd = FreqDist([word for word in self._toks])
            fd = dict(sorted([(k, v) for k, v in fd.items()], reverse=self._rev, key=lambda x: x[1]))
            self._freq_dict = fd
            return fd
        else:
            return self._freq_dict

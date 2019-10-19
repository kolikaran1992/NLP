from nltk.tokenize import RegexpTokenizer
from unidecode import unidecode

def get_word_tokenizer():
    return RegexpTokenizer(r'\w+|[^\w\s]').tokenize

def process_text(text):
    return unidecode(text)

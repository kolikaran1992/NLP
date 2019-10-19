from nltk.tokenize import RegexpTokenizer
from unidecode import unidecode
from pathlib import Path
import numpy as np

def get_word_tokenizer():
    return RegexpTokenizer(r'\w+|[^\w\s]').tokenize

def process_text(text):
    return unidecode(text)

def validate_path(path, logger):
    if not isinstance(path, Path):
        path = Path(path)

    if not path.is_file():
        logger.error('{} is not a valid file'.format(path.as_posix()))
        exit(101)

def read_label_file(path, logger):
    validate_path(path, logger)
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().strip().split(' ')


def read_word2vec(path_to_file):
    """
    --> read text file in word2vec format
    --> returns vocab list and vector matrix
    :param path_to_file: path
    :return: list, np.vector
    """
    vocab = []
    with open(path_to_file, 'r') as f:
        lines = f.readlines()
        vocab_size, vector_dim = tuple(map(int, lines[0].split(' ')))
        vectors = np.zeros((vocab_size, vector_dim))
        for idx, line in enumerate(lines[1:]):
            line = line.split(' ')
            print(len(line))
            vocab.append(line[0])
            try:
                vectors[idx,:] = tuple(map(np.float, map(str.strip, line[1:])))
            except:
                print(vocab[-2])
                exit(1000)

    return vocab, np.array(vectors)
## data type to be used
import tensorflow as tf
import numpy as np
from TF_NER.__paths__ import PATH_TO_DATA
from collections import defaultdict

from spacy.gold import biluo_tags_from_offsets
import spacy
nlp = spacy.blank('en')
text = 'I have the power'
doc = text.split(' ')
print(biluo_tags_from_offsets(doc, [(2,6, 'aaa')]))

DTYPE = {
    'float' : {
        'tf' : tf.float32,
        'np' : np.float32
    },
    'int' : {
        'tf' : tf.int32,
        'np' : np.int32
    }
}

LOGGER_NAME = 'TF_NER_LOGGER'

## vocab items

import logging

logger = logging.getLogger(LOGGER_NAME)

def check_path_existence(name):
    if not PATH_TO_DATA.joinpath(name).is_file():
        logger.error('file handling - {} file does not exist'.format(name))
        exit()
    return True


def read_file(name):
    if check_path_existence(name):
        with open(PATH_TO_DATA.joinpath(name).as_posix(), 'r') as file_obj:
            return file_obj.readlines()


vocab_text_to_id = {token.strip(): idx for idx, token in enumerate(read_file('vocab_text'))}
vocab_id_to_text = {idx: token.strip() for idx, token in enumerate(read_file('vocab_text'))}
logger.info('file handling - total words = {}'.format(len(vocab_text_to_id)))

vocab_tag_to_id = {token.strip(): idx for idx, token in enumerate(read_file('vocab_tag'))}
vocab_id_to_tag = {idx: token.strip() for idx, token in enumerate(read_file('vocab_tag'))}
logger.info('file handling - total tags = {}'.format(len(vocab_tag_to_id)))

char_to_id = {token.strip(): idx for idx, token in enumerate(read_file('characters'))}
id_to_char = {idx: token.strip() for idx, token in enumerate(read_file('characters'))}
logger.info('file handling - total characters = {}'.format(len(char_to_id)))

## get ents

tag_to_id_for_ent = {
    'B' : [(tag.split('-')[-1], id_) for tag, id_ in vocab_tag_to_id.items() if tag.split('-')[0] == 'B'],
    'I' : vocab_tag_to_id['I'],
    'E' : vocab_tag_to_id['E'],
    'S' : [(tag.split('-')[-1], id_) for tag, id_ in vocab_tag_to_id.items() if tag.split('-')[0] == 'S']
}

def extract_entities_single(toks):
    ent_word = ''
    ents = defaultdict(list)
    temp_ent = []
    for idx, tok in enumerate(toks):
        if tok in [idx for (tag, idx) in tag_to_id_for_ent['B']]:
            ent = vocab_id_to_tag[tok].split('-')[-1]
            ent_word = ent
            temp_ent.append(idx)
        elif tok  == tag_to_id_for_ent['I']:
            temp_ent.append(idx)
        elif tok == tag_to_id_for_ent['E']:
            temp_ent.append(idx)
            ents[ent_word].append(temp_ent)
            temp_ent = []
        elif tok in [idx for (tag, idx) in tag_to_id_for_ent['S']]:
            ent = vocab_id_to_tag[tok].split('-')[-1]
            ents[ent].append([idx])
    return ents

def extract_entities_batch(batch):
    return [extract_entities_single(item) for item in batch]
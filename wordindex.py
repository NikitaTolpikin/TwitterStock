import numpy as np
from collections import Counter
import data_loader as dl


def get_word_2_index(vocab):
    word2index = {}
    for i, word in enumerate(vocab):
        word2index[word] = i
    word2index['НС']=len(vocab)

    return word2index


def get_indexer():

    vocab = Counter()
    all_texts = dl.load_all_texts()

    text = ''
    for item in all_texts:
        text += ' ' + item

    for word in text.split(' '):
        vocab[word] += 1

    norm_vocab = vocab.copy()

    for word in vocab:
        if vocab[word] < 4:
            del norm_vocab[word]

    return get_word_2_index(norm_vocab)


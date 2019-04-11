import csv
import numpy as np
import random

def load_all():
    data = []
    with open('all_prep.csv', mode='r', encoding='utf-8-sig') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            data.append({'id': row['id'], 'text': row['text']})
    return data


def load_marked():
    data = []
    with open('marked_prep.csv', mode='r', encoding='utf-8-sig') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            data.append({'id': row['id'], 'text': row['text'], 'mark': row['mark']})
    return data


def load_all_texts():
    data = []
    with open('all_prep.csv', mode='r', encoding='utf-8-sig') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            data.append(row['text'])
    return data


def load_marked_texts():
    data = []
    max_sen_len = 0
    with open('marked_prep.csv', mode='r', encoding='utf-8-sig') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            data.append(row['text'])
            text_len=len(row['text'].split())
            if text_len>max_sen_len: max_sen_len = text_len
    return data, max_sen_len


def load_marked_labels():
    data = []
    with open('marked_prep.csv', mode='r', encoding='utf-8-sig') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            data.append(row['mark'])
    return data


def get_batches(data, batch_size, num_epochs, shuffle=True):
    data = list(data)
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def get_sets(data, labels, ratio):
    c = list(zip(data, labels))
    random.shuffle(c)
    test_sample_index = -1 * int(ratio * float(len(data)))
    data, labels = zip(*c)
    x_train, x_test = data[:test_sample_index], data[test_sample_index:]
    y_train, y_test = labels[:test_sample_index], labels[test_sample_index:]
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

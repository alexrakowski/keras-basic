import random

import numpy as np

DATASET_PATH = '../../datasets/cambridge-hand-gesture'
NO_CLASSES = 2
DATA_DIM = (1, 1)


def gen_data(dataset_path=DATASET_PATH, shuffle=True, loop=True):
    while True:
        data = [i for i in range(1000)]
        labels = [0 if i < 500 else 1 for i in range(1000)]
        data_and_labels = [(x, y) for x, y in zip(data, labels)]

        if shuffle:
            random.shuffle(data_and_labels)

        for X, label in data_and_labels:
            if NO_CLASSES > 2:
                Y = np.zeros(NO_CLASSES)
                Y[label] = 1
                Y = Y.reshape(1, NO_CLASSES)
            else:
                Y = np.array([label])
                Y = Y.reshape((1, 1, 1))

            X = np.array([X])
            X = X.reshape((1, 1, X.shape[0]))
            yield X, Y
        if not loop:
            break

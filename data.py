import os
import random

import numpy as np
import cv2

DATASET_PATH = '../../datasets/cambridge-hand-gesture'
NO_CLASSES = 2


def gen_data(dataset_path=DATASET_PATH, shuffle=True, loop=True):
    while True:
        data = []
        labels = []
        data_and_labels = []

        if shuffle:
            random.shuffle(data_and_labels)

        for X, label in data_and_labels:
            Y = np.zeros(NO_CLASSES)
            Y[label] = 1
            Y = Y.reshape(1, NO_CLASSES)
            yield X, Y
        if not loop:
            break

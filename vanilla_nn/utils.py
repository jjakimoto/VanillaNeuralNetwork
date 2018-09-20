import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.abs(x) * (x > 0)


def one_hot_encoding(x, n_classes):
    return np.eye(n_classes)[x]


def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

import numpy as np

from .core import BaseLoss
from .utils import one_hot_encoding, softmax


class MSE(BaseLoss):
    def compute(self, y, y_pred):
        loss = np.mean((y - y_pred) ** 2)
        return np.sqrt(loss)

    def get_delta(self, y, y_pred):
        return 2 * (y_pred - y)


class CrossEntropy(BaseLoss):
    def compute(self, y, y_pred):
        n_classes = y_pred.shape[1]
        y_ = one_hot_encoding(y, n_classes)
        y_pred = np.log(softmax(y_pred))
        return -np.mean(np.sum(y_ * y_pred, axis=1))

    def get_delta(self, y, y_pred, eps=1e-6):
        y_pred = softmax(y_pred)
        delta = y_pred.copy()
        num_data = len(y)
        delta[np.arange(num_data), y] -= 1.
        return delta
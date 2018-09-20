import numpy as np

from .core import BaseLayer, Node


class Linear(BaseLayer):
    def __init__(self, in_dim, out_dim, W=None, b=None):
        if W is None:
            W = np.random.randn(in_dim, out_dim) / np.sqrt(in_dim)
        assert W.shape == (
        in_dim, out_dim), f"W.shape has to be ({in_dim}, {out_dim})"
        if b is None:
            b = np.zeros((1, out_dim))
        assert b.shape == (1, out_dim), f"b.shape has to be (1, {out_dim})"
        self.W = Node(W)
        self.b = Node(b)

    def forward(self, x, *args, **kwargs):
        return x.dot(self.W.data) + self.b.data

    def backward(self, delta, *args):
        b_grads = np.mean(delta, axis=0)
        self.b.gradient = np.expand_dims(b_grads, 0)
        x_ = np.expand_dims(self.x, 2)
        delta_ = np.expand_dims(delta, 1)
        W_grads = np.matmul(x_, delta_)
        self.W.gradient = np.mean(W_grads, axis=0)
        return np.dot(delta, self.W.data.T)

    def update(self, scale=1.):
        self.W.update(scale)
        self.b.update(scale)


class Dropout(BaseLayer):

    def __init__(self, drop_rate):
        self.drop_rate = drop_rate

    def forward(self, x, training=True, *args, **kwargs):
        size = x.shape[1:]
        size = (1,) + size
        if training:
            self.mask = np.random.binomial(1, 1 - self.drop_rate, size)
            return (x * self.mask) / (1. - self.drop_rate)
        else:
            return x

    def backward(self, delta, training=True):
        delta *= self.mask / (1 - self.drop_rate)
        return delta

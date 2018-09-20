from abc import ABC, abstractmethod


class BaseLayer(ABC):
    @abstractmethod
    def forward(self, x, *args, **kwargs):
        """Implement me"""
        raise NotImplementedError()

    @abstractmethod
    def backward(self, delta, *args):
        """Implement me"""
        raise NotImplementedError()

    def update(self, scale):
        pass

    def __call__(self, x, *args, **kwargs):
        self.x = x
        return self.forward(x, *args)


class BaseLoss(ABC):
    @abstractmethod
    def compute(self, y, y_pred):
        """Implement me"""
        raise NotImplementedError()

    @abstractmethod
    def get_delta(self, y, y_pred):
        """Implement me"""
        raise NotImplementedError()

    def update(self, scale):
        pass

    def __call__(self, y, y_pred):
        return self.compute(y, y_pred)


class Node(object):
    def __init__(self, data, gradient=None):
        self._data = data
        self._gradient = gradient

    def update(self, scale=1.):
        self._data -= scale * self.gradient

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def gradient(self):
        return self._gradient

    @gradient.setter
    def gradient(self, value):
        self._gradient = value
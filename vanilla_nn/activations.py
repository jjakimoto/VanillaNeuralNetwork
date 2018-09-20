from .core import BaseLayer
from .utils import sigmoid, relu


class Activation(BaseLayer):
    @property
    def gradient(self, x, *args):
        raise NotImplementedError()

    def backward(self, delta, *args):
        return delta * self.gradient(self.x)


class Sigmoid(Activation):
    def forward(self, x, *args, **kwargs):
        return sigmoid(x)

    def gradient(self, x, *args):
        return sigmoid(x) * (1. - sigmoid(x))


class ReLU(Activation):
    def forward(self, x, *args, **kwargs):
        return relu(x)

    def gradient(self, x, *args):
        x[x < 0] = 0
        x[x > 0] = 1
        return x
from .core import BaseLayer

import numpy as np


class Model(BaseLayer):
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x, training=True, *args, **kwargs):
        for layer in self.layers:
            x = layer(x, training)
        return x

    def backward(self, delta):
        for layer in self.layers[::-1]:
            delta = layer.backward(delta)

    def update(self, scale):
        for layer in self.layers:
            layer.update(scale)


class MultiClassifier(Model):
    def predict(self, x, *args, **kwargs):
        x = self.forward(x, *args, **kwargs)
        x = np.argmax(x, axis=1)
        return x

    def predict_proba(self, x, *args, **kwargs):
        return self.forward(x, *args, **kwargs)

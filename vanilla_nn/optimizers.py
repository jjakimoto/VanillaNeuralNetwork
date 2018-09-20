class GDOptimizer(object):
    def __init__(self, model, loss, lr=1.):
        self.model = model
        self.loss = loss
        self.lr = lr

    def update(self, X, y):
        y_pred = self.model(X)
        delta = self.loss.get_delta(y, y_pred)
        self.model.backward(delta)
        self.model.update(self.lr)

    def evaluate(self, X, y):
        y_pred = self.model(X)
        return self.loss.compute(y, y_pred)

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

    def evaluate(self, X, y, training=True, score_func=None):
        if score_func is None:
            y_pred = self.model(X, training)
            score_func = self.loss.compute
        else:
            y_pred = self.model.predict(X, training)
        return score_func(y, y_pred)

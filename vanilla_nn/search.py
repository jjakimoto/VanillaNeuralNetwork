from sklearn.model_selection import StratifiedKFold
import numpy as np

from .models import MultiClassifier
from .layers import Linear, Dropout
from .activations import Sigmoid, ReLU
from .optimizers import GDOptimizer


def build_model(configs):
    layers = []
    input_dim = configs['input_dim']
    output_dim = configs['output_dim']
    layer_configs = configs['layer']
    for config in layer_configs:
        layers.append(Dropout(config['drop_rate']))
        hidden_dim = config['n_units']
        layers.append(Linear(input_dim, hidden_dim))
        activation = config.get('activation')
        if activation == 'sigmoid':
            layers.append(Sigmoid())
        elif activation == 'relu':
            layers.append(ReLU())
        input_dim = hidden_dim
    layers.append(Linear(input_dim, output_dim))
    model = MultiClassifier(layers)
    return model


def generate_configs(search_config):
    n_layers = np.random.randint(search_config['n_layers'][0],
                                 search_config['n_layers'][1])
    layer_configs = []
    for _ in range(n_layers):
        config = dict()
        config['n_units'] = np.random.randint(search_config['n_units'][0],
                                              search_config['n_units'][1])
        config['activation'] = np.random.choice(search_config['activations'])
        config['drop_rate'] = np.random.uniform(search_config['drop_rate'][0],
                                                search_config['drop_rate'][1])
        layer_configs.append(config)
    configs = dict()
    configs['layer'] = layer_configs
    configs['input_dim'] = search_config['input_dim']
    configs['output_dim'] = search_config['output_dim']
    power = np.random.uniform(search_config['lr'][0],
                              search_config['lr'][1])
    configs['lr'] = 10 ** power
    return configs


def random_search(X, y, loss, score_func, search_config,
                  n_epochs=1000, n_iter=10):
    """Random Search for hyperparameters

    Parameters
    ----------
    X: array-like, shape=(n_samples, n_fetures)
    y: array-like, shape=(n_samples,)
    loss: Instance of loss function from losses.py
    score_func: func
    search_config: dict
        Define the range of search
    n_epochs: int, (default 1000)
        The number of iteration of gradient descent
    n_iter: int
        The number of trials to find parameters

    Returns
    -------
    model: MultiCalssifier instance
    best_config: dict
    best_score: float
    """
    best_score = -float('inf')
    for _ in range(n_iter):
        configs = generate_configs(search_config)

        model = build_model(configs)

        kfold = StratifiedKFold(n_splits=4, shuffle=True)
        scores = []
        for train_idx, valid_idx in kfold.split(X, y):
            train_X = X[train_idx]
            train_y = y[train_idx]
            valid_X = X[valid_idx]
            valid_y = y[valid_idx]

            optimizer = GDOptimizer(model,
                                    loss,
                                    lr=configs['lr'])
            for i in range(n_epochs):
                optimizer.update(train_X, train_y)
            pred = np.argmax(model.forward(valid_X, training=False), axis=1)
            _score = score_func(valid_y, pred)
            scores.append(_score)
        score = np.mean(scores)
    if score > best_score:
        best_score = score
        best_config = configs
    model = build_model(best_config)
    return model, best_config, best_score
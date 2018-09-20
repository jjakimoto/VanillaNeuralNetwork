import sys

from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

from vanilla_nn import random_search
from vanilla_nn.losses import CrossEntropy
from vanilla_nn.optimizers import GDOptimizer


if  __name__ == '__main__':
    dataset = load_iris()
    X = dataset.data
    y = dataset.target

    # To keep the class distribution, we use stratified sampling
    split = StratifiedShuffleSplit(n_splits=1, test_size=.2)
    train_idx, test_idx = next(split.split(X, y))
    train_X = X[train_idx]
    train_y = y[train_idx]
    test_X = X[test_idx]
    test_y = y[test_idx]


    # Each factor determine the distributino of hyperparameters of neural network
    search_config = {'input_dim': 4,
                     'output_dim': 3,
                     'drop_rate': [0, .5],
                     'n_layers': [0, 4],
                     'n_units': [4, 64],
                     'activations': [None, 'sigmoid', 'relu'],
                     'lr': [0, -3]}
    loss = CrossEntropy()
    score_func = accuracy_score
    if len(sys.argv) >= 2:
        n_epochs = sys.argv[1]
    else:
        n_epochs = 1000

    if len(sys.argv) >= 3:
        n_iter = sys.argv[2]
    else:
        n_iter = 10
    best_model, best_config, best_score = random_search(train_X, train_y, loss,
                                                        score_func, search_config,
                                                        n_epochs=n_epochs,
                                                        n_iter=n_iter)
    print('Valiidation Score: ', best_score)
    print('Best Configuration: ', best_config)

    # Check the performance on test dataset
    optimizer = GDOptimizer(best_model,
                            CrossEntropy(),
                            lr=best_config['lr'])
    for i in range(n_epochs):
        optimizer.update(train_X, train_y)

    pred = best_model.predict(test_X, training=False)
    test_score = accuracy_score(test_y, pred)
    print('Test Score:', test_score)

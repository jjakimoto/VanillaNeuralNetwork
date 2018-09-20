# Vanilla Neural Network
* An implementation of feed forward neural network through numpy
* Random search for hyper parameters

# Installation
```buildoutcfg
python setup.py install
```

# Example
* Vanilla Example

Please check `examples/Vanilla Example.ipynb`

* Random Search for hyperparameters
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

from vanilla_nn import random_search
from vanilla_nn.losses import CrossEntropy

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
best_model, best_config, best_score = random_search(train_X, train_y, loss,
                                                    score_func, search_config)
```

This example code can be tested through the following command :
```buildoutcfg
python examples/random_search.py n_epochs n_iter
```
`n_epochs` and `n_iter` are optional parmeters, which defines the number of training epochs and trials
for hyperparameter search, respectively.
#!/usr/bin/python
'''adaline_sgd.py illustrates Stochastic Gradient Descent.

First, the weights are updated after each training sample instead of 
calculating the error for the entire batch. This causes the weights to
converge faster than the batch method.

Second, the samples can be shuffled to avoid bias based on the order of samples in
the training set.

The decision regions and the speed of convergence is plotted

Created on Jun 22, 2016

from Python Machine Learning by Sebastian Raschka

@author: richard lyman
'''
import ocr_utils
import numpy as np
from numpy.random import  seed
import matplotlib.pyplot as plt


#############################################################################
# read images and scatter plot

# retrieve 100 sets of target numbers and column sums
#    y: the ascii characters 48 and 49 ('0', '1')
#    X: the sum of the vertical pixels in the rows in horizontal columns 9 and 17

ascii_characters_to_train=(48,51)
columnsXY = (9,17)       
nchars=500
y, X, y_test,  X_test, labels  = ocr_utils.load_E13B(chars_to_train = ascii_characters_to_train , columns=columnsXY,nChars=120) 
y = np.where(y==ascii_characters_to_train[1],-1,1)

#############################################################################
# AdalineSGD from Python Machine Learning
class AdalineSGD(object):
    """ADAptive LInear NEuron classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications in every epoch.
    shuffle : bool (default: True)
        Shuffles training data every epoch if True to prevent cycles.
    random_state : int (default: None)
        Set random state for shuffling and initializing the weights.
        
    """
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        if random_state:
            seed(random_state)
        
    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = np.random.permutation(len(y))
        return X[r], y[r]
    
    def _initialize_weights(self, m):
        """Initialize weights to zeros"""
        self.w_ = np.zeros(1 + m)
        self.w_initialized = True
        
    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.net_input(xi)
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost
    
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return self.net_input(X)

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(X) >= 0.0, 1, -1)
#############################################################################
# standardize features,fit, and plot
X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
ada = AdalineSGD(n_iter=15, eta=0.01)
ada.fit(X_std, y)

ocr_utils.plot_decision_regions(X=X_std, 
                           y=y,
                           classifier=ada, 
                           title='Adaline - Stochastic Gradient Descent',
                           labels=labels)

title='Adaline - Stochastic Gradient Descent'
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.title(title)
plt.tight_layout()
ocr_utils.show_figures(plt, title)

print ('\n########################### No Errors ####################################')
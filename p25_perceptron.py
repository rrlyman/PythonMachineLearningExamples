#!/usr/bin/python
'''perceptron is a Python implementation of the Rosenblatt perceptron which
uses a non-differentiatable unit step function as the activation function.

The target classes such as the characters '0' and '1' are changed to -1 and 1
The difference between target value and the predicted target value multiplied
by a small 'eta' value is an update value used for adjusting the weights.

The weights are adjusted by the update value times the image.

This eventually converges the weights to value that can provide a good
prediction of new images.

The misclassification versus Epochs and the resulting decision regions
are plotted.

Created on Jun 20, 2016

from Python Machine Learning by Sebastian Raschka under the following license

The MIT License (MIT)

Copyright (c) 2015, 2016 SEBASTIAN RASCHKA (mail@sebastianraschka.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

@author: richard lyman
'''
import ocr_utils
import numpy as np
import matplotlib.pyplot as plt

#############################################################################
# read features and scatter plot

# retrieve 500 sets of target numbers and column sums
#    y: the ascii characters 48 and 49 ('0', '1')
#    X: the sum of the vertical pixels in the rows in horizontal columns 9 and 17
ascii_characters_to_train=(48,49)
columnsXY = (9,17)       
nchars=500
y, X, y_test,  X_test, labels  = ocr_utils.load_E13B(chars_to_train = ascii_characters_to_train , columns=columnsXY,nChars=nchars) 
 

#############################################################################
# Perceptron implementation from Python Machine Learning
class Perceptron(object):
    """Perceptron classifier.

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

    """
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """Fit training data.

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
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)

#############################################################################
# convert targets 9'0','1') to -1,+1
# fit train the Perceptron
# plot the misclassifications versus Epochs
# plot the decision regions
 
y = np.where(y == ascii_characters_to_train[0], -1, 1)
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)

title = 'Simple Perception'
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.title(title)
plt.tight_layout()
ocr_utils.show_figures(plt, title)

ocr_utils.plot_decision_regions(X=X, 
                           y=y, 
                           classifier=ppn,
                           labels = ['column {} sum'.format(columnsXY[i]) for i in range(len(columnsXY))], 
                           title="Perceptron Decision Regions")



print ('\n########################### No Errors ####################################')
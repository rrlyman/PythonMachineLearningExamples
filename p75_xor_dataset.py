#!/usr/bin/python
''' xor_dataset.py shows how non linearly separable datasets can use
    a non linear combination of the original features to project the
    features onto a higher dimensional space where the features are
    linearly separable
    
    A non linearly dataset consisting of XOR values is created.
    This is fitted to a Support Vector Machine using the radial basis 
    function kernel parameter
    
Created on Jun 23, 2016

from Python Machine Learning by Sebastian Raschka

@author: richard lyman
'''
import numpy as np

import ocr_utils

from sklearn.svm import SVC

np.random.seed(0)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)

ocr_utils.scatter_plot(X=X_xor, 
                  y=y_xor,                   
                  title='xor',
                  xlim=(-3,3),
                  ylim=(-3,3))


svm = SVC(kernel='rbf', random_state=0, gamma=0.10, C=10.0)
svm.fit(X_xor, y_xor)
ocr_utils.plot_decision_regions(X=X_xor, y=y_xor, 
                      classifier=svm,title='support vector machine rbf xor')
print ('\n########################### No Errors ####################################')

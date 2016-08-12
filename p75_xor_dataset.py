#!/usr/bin/python
''' xor_dataset.py shows how non linearly separable datasets can use
    a non linear combination of the original features to project the
    features onto a higher dimensional space where the features are
    linearly separable
    
    A non linearly dataset consisting of XOR values is created.
    This is fitted to a Support Vector Machine using the radial basis 
    function kernel parameter
    
Created on Jun 23, 2016

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

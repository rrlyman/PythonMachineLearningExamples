#!/usr/bin/python
''' random_forest.py 
    The k nearest neighbor classifier memorizes the training set.  When the 
    class label of a new sample is to be predicted, the distance, typically
    the Euclidean distance some number, like 5, of the nearest memorized
    points is found. The class label of the new point is that of the 
    majority of the nearest neighbors. 
        
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
from sklearn.preprocessing import StandardScaler

y_train, X_train, y_test,  X_test, labels  = ocr_utils.load_E13B(chars_to_train = (48,49,50) , columns=(9,17), test_size=0.3, nChars=300, random_state=0) 


sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train_std, y_train)

ocr_utils.plot_decision_regions(X=X_combined_std, 
                      y=y_combined, 
                      classifier=knn, 
                      labels=labels,                      
                      test_idx=range(len(X_test_std),len(X_combined_std)),
                      title='k_nearest_neighbors')
print ('\n########################### No Errors ####################################')
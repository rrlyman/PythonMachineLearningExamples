#!/usr/bin/python
''' scaling_features.py

    Investigate normalization versus standardization

    The features in the ocr_utils are already normalized.  That is, each image
    has been stretched to go from pure black to pure white.  The values in the
    .csv file are 0 to 255.  ocr_utils.py changes these value to be in the range
    0.0 to 1.0
    
    1) prints out a sampling of the normalized values. 
    2) standardize the values and print them out.
     
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
from sklearn.neighbors import KNeighborsClassifier

y_train, X_train, y_test,  X_test, labels  = ocr_utils.load_E13B(chars_to_train = (48,49,50) , columns=(9,17), test_size=0.3, nChars=300, random_state=0) 

# put the ASCII equivalent of the unique characters in y into the legend of the plot
legend=[]
for ys in np.unique(y_train):
    legend.append('{} \'{}\''.format(ys, chr(ys)))

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train, y_train)


ocr_utils.plot_decision_regions(X=X_combined, 
                      y=y_combined, 
                      classifier=knn, 
                      labels=labels,                      
                      test_idx=range(len(X_test),len(X_combined)),
                      title='k_nearest_neighbors no scaling')

from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)
X_combined_norm = np.vstack((X_train_norm, X_test_norm))

knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train_norm, y_train)


ocr_utils.plot_decision_regions(X=X_combined_norm, 
                      y=y_combined, 
                      classifier=knn, 
                      labels=labels,                      
                      test_idx=range(len(X_test_norm),len(X_combined_norm)),
                      title='k_nearest_neighbors MinMaxScaller')

from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
X_combined_std = np.vstack((X_train_std, X_test_std))

knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train_std, y_train)

ocr_utils.plot_decision_regions(X=X_combined_std, 
                      y=y_combined, 
                      classifier=knn, 
                      labels=labels,                      
                      test_idx=range(len(X_test_std),len(X_combined_std)),
                      title='k_nearest_neighbors Standard Normalized')

print ('\n########################### No Errors ####################################')
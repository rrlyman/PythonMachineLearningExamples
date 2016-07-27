#!/usr/bin/python
''' random_forest.py 
    The k nearest neighbor classifier memorizes the training set.  When the 
    class label of a new sample is to be predicted, the distance, typically
    the Euclidean distance some number, like 5, of the nearest memorized
    points is found. The class label of the new point is that of the 
    majority of the nearest neighbors. 
        
Created on Jun 23, 2016

from Python Machine Learning by Sebastian Raschka

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
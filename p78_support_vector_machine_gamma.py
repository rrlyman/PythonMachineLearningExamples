#!/usr/bin/python
''' support_vector_machine_gamma.py illustrates changing the gamma parameter
    for a SVM.  This is a cut-off parameters for the Gaussian sphere.  A 
    higher value tightens the decision boundary around the samples
    
    Run the SVM with two values of gamma and plot the decision regions
    
Created on Jun 23, 2016

from Python Machine Learning by Sebastian Raschka

@author: richard lyman
'''
import numpy as np
import ocr_utils
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

y_train, X_train, y_test,  X_test, labels  = ocr_utils.load_E13B(chars_to_train = (48,49,50) , columns=(9,17), test_size=0.3, nChars=300, random_state=0) 

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))


svm = SVC(kernel='rbf', random_state=0, gamma=0.2, C=1.0)
svm.fit(X_train_std, y_train)

ocr_utils.plot_decision_regions(X=X_combined_std, 
                      y=y_combined, 
                      classifier=svm, 
                      labels = labels,
                      test_idx=range(len(X_test_std),len(X_combined_std)),
                      title='SVM with gamma 0.2')

svm = SVC(kernel='rbf', random_state=0, gamma=100.0, C=1.0)
svm.fit(X_train_std, y_train)

ocr_utils.plot_decision_regions(X=X_combined_std, 
                      y=y_combined, 
                      classifier=svm, 
                      labels = labels,
                      test_idx=range(len(X_test_std),len(X_combined_std)),
                      title='SVM with gamma 100')


print ('\n########################### No Errors ####################################')

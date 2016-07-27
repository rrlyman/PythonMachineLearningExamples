'''support_vector_machine_linear.py illustrates a support vector machine
    The SVM attempts to maximize the margin of error between linearly 
    separable feature sets.
    
    Column sums are read in from the E13B character set and fitted to
    a SVM.  The decision regions are plotted.
    
Created on Jun 30, 2016

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

from sklearn.svm import SVC

svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(X_train_std, y_train)

ocr_utils.plot_decision_regions(X=X_combined_std, 
                                         y=y_combined,                       
                                         classifier=svm, 
                                         test_idx=range(len(X_test_std),len(X_combined_std)),
                                         labels = labels, 
                                         title='support_vector_machine_linear')
print ('\n########################### No Errors ####################################')

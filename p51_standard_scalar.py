#!/usr/bin/python
'''standard_scalar illustrates the use of the scaling from
    the sklearn tools.
    1) column sums from the E13B dataset are read in as features.
    2) Features are scaled with the sklearn StandardScaler
    3) The features are then fitted to a Perceptron and the decision regions
        are plotted.

Created on Jun 23, 2016

from Python Machine Learning by Sebastian Raschka

@author: richard lyman
'''
import numpy as np
import ocr_utils
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron


#############################################################################
# read images and scatter plot

# retrieve 500 sets of target numbers and column sums
#    y: the ascii characters 48 and 49 ('0', '1')
#    X: the sum of the vertical pixels in the rows in horizontal columns 9 and 17

y, X, y_test,  X_test, labels  = ocr_utils.load_E13B(chars_to_train = (48,51) , columns=(9,17),nChars=500, random_state=0) 

print('Class labels:', np.unique(y))

from sklearn.cross_validation import train_test_split
#############################################################################
# standardize the features

X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.3, random_state=0)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())

from sklearn.metrics import accuracy_score

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
       
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

ocr_utils.plot_decision_regions(X_combined_std, y_combined, ppn, 
                           test_idx=range(len(X_test_std),len(X_combined_std)),
                           labels=labels, 
                           title='perceptron_scikit')



print ('\n########################### No Errors ####################################')
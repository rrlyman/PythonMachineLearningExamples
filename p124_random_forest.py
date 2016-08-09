#!/usr/bin/python
''' random_forest.py 
    The random forest uses weak learners to build a strong learner
    
    A random subset of samples is drawn and then at each node a decision
    tree is grown from a smaller subset of those bootstrap samples
    
    This is repeated a number of times and then the decision trees are 
    combined via majority vote.
    
    1) run the random forest on the e13b data
    2) plot the decision regions
    
    
Created on Jun 23, 2016

from Python Machine Learning by Sebastian Raschka

@author: richard lyman
'''
import numpy as np
import ocr_utils
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    
    y_train, X_train, y_test,  X_test, labels  = ocr_utils.load_E13B(chars_to_train = (48,49,50) , columns=(9,17), test_size=0.3, nChars=300, random_state=0) 


    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))

    from sklearn.ensemble import RandomForestClassifier

    forest = RandomForestClassifier(criterion='entropy',
                                    n_estimators=10, 
                                    random_state=1,
                                    n_jobs=2)
    forest.fit(X_train, y_train)

    ocr_utils.plot_decision_regions(X=X_combined, 
                                             y=y_combined, 
                                             classifier=forest, 
                                             labels=labels,                                         
                                             test_idx=range(len(X_test_std),len(X_combined_std)),
                                             title='random_forest')

    print ('\n########################### No Errors ####################################')

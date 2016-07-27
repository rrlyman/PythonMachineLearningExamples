#!/usr/bin/python
''' random_forest_feature_importance.py

    Using a random forest (construct strong learners from weak learners,
    the importance of each features is evaluated by measuring the impurity
    decrease for each of 10000 trees
    
    
Created on Jun 23, 2016

from Python Machine Learning by Sebastian Raschka

@author: richard lyman
'''
import numpy as np
import ocr_utils
import matplotlib.pyplot as plt


y_train, X_train, y_test,  X_test, labels  = ocr_utils.load_E13B(chars_to_train = (48,49,50) , columns=range(0,20), nChars=1000, test_size=0.3,random_state=0) 


from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=10000,
                                random_state=0,
                                n_jobs=-1)

forest.fit(X_train, y_train)
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            indices[f], 
                            importances[indices[f]]))

title = 'Feature Importances from Random Forest'
plt.title(title)
plt.bar(range(X_train.shape[1]), 
        importances[indices],
        color='lightblue', 
        align='center')

plt.xticks(range(X_train.shape[1]), 
           indices, rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.xlabel('column sums')
plt.tight_layout()
ocr_utils.show_figures(plt,title)

print ('\n########################### No Errors ####################################')
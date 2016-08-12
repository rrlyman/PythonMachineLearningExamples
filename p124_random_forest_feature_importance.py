#!/usr/bin/python
''' random_forest_feature_importance.py

    Using a random forest (construct strong learners from weak learners,
    the importance of each features is evaluated by measuring the impurity
    decrease for each of 10000 trees
    
    
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
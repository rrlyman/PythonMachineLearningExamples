''' confusion_matrix.py

A confusion matrix shows a map of true positives, false positives, false
negatives, and true negatives for a decision.

Some decisions require a biased output where it we may want to reduce
the number of false positives, for instance.  This is especially true
in medical diagnosis.


Created on Jul 8, 2016

from Python Machine Learning by Sebastian Raschka

@author: richard lyman
'''

import matplotlib.pyplot as plt
import ocr_utils  
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix

y_train, X_train, y_test,  X_test, labels  = ocr_utils.load_E13B(chars_to_train = (48,51) , test_size=0.3, columns=(9,17), random_state=0) 

pipe_svc = Pipeline([('scl', StandardScaler()),
            ('clf', SVC(random_state=1))])

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid = [{'clf__C': param_range, 
               'clf__kernel': ['linear']},
                 {'clf__C': param_range, 
                  'clf__gamma': param_range, 
                  'clf__kernel': ['rbf']}]

pipe_svc.fit(X_train, y_train)
y_pred = pipe_svc.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)

fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')
title='c5_confusion_matrix'
plt.title(title)
plt.tight_layout()
ocr_utils.show_figures(plt,title)

print ('\n########################### No Errors ####################################')
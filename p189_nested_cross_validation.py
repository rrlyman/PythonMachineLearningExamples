''' nested_cross_validation.py
Nested Cross Validation is a method for tuning model parameters minimizing bias.

There is an outer k-fold cross validation loop and an inner k-fold cross 
validation loop.

The outer fold selects a number, such as 10 different training and
test sets without replacement so each sample ends up being used as a
test sample exactly once.

The inner fold uses the training portion of the outer fold, and does a 
Grid Search to select a classification model, such as 'linear' SVM version 'rbf'
or Decision Tree versus SVM.

If the model is stable, then the inner loops should all chose the same 
classifier type.

After selecting the classifier then the outer folds are used for tuning, via
k-fold classification.

This program uses the sklearn GridSearch Cross Validation that internally uses
a 5 outer fold, 2 inner folder algorithm to tune parameters.


Created on Jul 8, 2016

from Python Machine Learning by Sebastian Raschka

@author: richard lyman
'''

import numpy as np
import ocr_utils
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
if __name__ == '__main__':
        y_train, X_train, y_test,  X_test, labels  = ocr_utils.load_E13B(chars_to_train = (48,51) , test_size=0.3, columns=(9,17), random_state=0) 


        pipe_svc = Pipeline([('scl', StandardScaler()),
                    ('clf', SVC(random_state=1))])

        c_gamma_range = [0.01, 0.1, 1.0, 10.0]
         
        param_grid = [{'clf__C': c_gamma_range, 
                       'clf__kernel': ['linear']},
                         {'clf__C': c_gamma_range, 
                          'clf__gamma': c_gamma_range, 
                          'clf__kernel': ['rbf'],}]

        gs = GridSearchCV(estimator=pipe_svc, 
                                    param_grid=param_grid, 
                                    scoring='accuracy', 
                                    cv=5,
                                    n_jobs=-1)


        scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
        print('\nSupport Vector Cross Validation accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

        gs = gs.fit(X_train, y_train)
        print('Support Vector Machine Grid Search best score: {}'.format(gs.best_score_))
        print('Support Vector Machine Grid Search best params: {}\n'.format(gs.best_params_))

        from sklearn.tree import DecisionTreeClassifier
        gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0), 
                                    param_grid=[{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}], 
                                    scoring='accuracy', 
                                    cv=5)


        scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
        print('Decision Tree Cross Validation accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

        gs = gs.fit(X_train, y_train)
        print('Decision Tree Grid Search best score: {}'.format(gs.best_score_))
        print('Decision Tree Grid Search best params: {}'.format(gs.best_params_))

        print ('\n########################### No Errors ####################################')

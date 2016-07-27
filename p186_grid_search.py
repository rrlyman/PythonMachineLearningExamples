'''
Created on Jul 8, 2016 grid_search.py
    Grid search does a brute force train and test of sample data, trying
    a grid of parameters.
    
    The SVM attempts to maximize the margin of error between linearly 
    separable feature sets.

from Python Machine Learning by Sebastian Raschka

@author: richard lyman
'''
import ocr_utils
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC

y_train, X_train, y_test,  X_test, labels  = ocr_utils.load_E13B(chars_to_train = (48,51) , test_size=0.3, columns=(9,17), random_state=0) 

pipe_svc = Pipeline([('scl', StandardScaler()),
            ('clf', SVC(random_state=1))])

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid = [{'clf__C': param_range, 
               'clf__kernel': ['linear']},
                 {'clf__C': param_range, 
                  'clf__gamma': param_range, 
                  'clf__kernel': ['rbf']}]

gs = GridSearchCV(estimator=pipe_svc, 
                  param_grid=param_grid, 
                  scoring='accuracy', 
                  cv=10,
                  n_jobs=-1)
gs = gs.fit(X_train, y_train)

print('Support Vector Machine Grid Search best score: {}'.format(gs.best_score_))
print('Support Vector Machine Grid Search best params: {}'.format(gs.best_params_))

clf = gs.best_estimator_
clf.fit(X_train, y_train)
print('Support Vector Machine Test accuracy: %.3f' % clf.score(X_test, y_test))

print ('\n########################### No Errors ####################################')




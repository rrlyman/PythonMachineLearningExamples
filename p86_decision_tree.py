#!/usr/bin/python
''' decision_tree.py shows three impurity measures Gini, entropy, and
    misclassification error when used with a decision tree classifier.
    
    These measures are used to estimate the information gain at each split.

    1) plot the 3 kinds of impurity measure 
    2) run the decision tree on the e13b data and plot the decision regions 
       
    To create a drawing of the tree run:
        dot -Tpng tree.dot -o tree.png
    
Created on Jun 23, 2016

from Python Machine Learning by Sebastian Raschka

@author: richard lyman
'''
import numpy as np
import ocr_utils
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

y_train, X_train, y_test,  X_test, labels  = ocr_utils.load_E13B(chars_to_train = (48,49,50) , columns=(9,17), test_size=0.3, nChars=300, random_state=0) 

def gini(p):
    return (p)*(1 - (p)) + (1-p)*(1 - (1-p))

def entropy(p):
    return - p*np.log2(p) - (1 - p)*np.log2((1 - p))

def error(p):
    return 1 - np.max([p, 1 - p])

x = np.arange(0.0, 1.0, 0.01)

ent = [entropy(p) if p != 0 else None for p in x]
sc_ent = [e*0.5 if e else None for e in ent]
err = [error(i) for i in x]


fig = plt.figure()

ax = plt.subplot(111)
for i, lab, ls, c, in zip([ent, sc_ent, gini(x), err], 
                  ['Entropy', 'Entropy (scaled)', 
                   'Gini Impurity', 'Misclassification Error'],
                  ['-', '-', '--', '-.'],
                  ['black', 'lightgray', 'red', 'green', 'cyan']):
    line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)

ax.legend(loc='upper center', ncol=2, fancybox=True, shadow=False)

ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
plt.ylim([0, 1.2])
plt.xlabel('p(i=1)')
plt.ylabel('Impurity Index')
plt.tight_layout()

ocr_utils.show_figures(plt, title='impurity')

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
tree.fit(X_train, y_train)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
ocr_utils.plot_decision_regions(X=X_combined, 
                                         y=y_combined, 
                                         classifier=tree, 
                                         test_idx=range(len(X_test),len(X_combined)),
                                         labels=labels,
                                         title='decision tree entropy')


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
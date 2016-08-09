'''bagging_bootstrap_samples.py

Bagging draws samples with replacement in order to train classifiers that are 
then combined my majority voting.


Created on Jul 10, 2016

from Python Machine Learning by Sebastian Raschka

@author: richard lyman
'''

from sklearn.preprocessing import LabelEncoder
import ocr_utils
from sklearn.cross_validation import train_test_split
import numpy as np
import matplotlib.pyplot as plt
if __name__ == '__main__':
    
    charsToTrain=(48,51)
    nChars = 1000
    y, X, y_test, X_test, labels  = ocr_utils.load_E13B(chars_to_train = charsToTrain , columns=(9,17), nChars=nChars)

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.40,random_state=1)

    from sklearn.ensemble import BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier

    tree = DecisionTreeClassifier(criterion='entropy', 
                                  max_depth=None)

    bag = BaggingClassifier(base_estimator=tree,
                            n_estimators=500, 
                            max_samples=1.0, 
                            max_features=1.0, 
                            bootstrap=True, 
                            bootstrap_features=False, 
                            n_jobs=-1, 
                            random_state=1)

    from sklearn.metrics import accuracy_score

    tree = tree.fit(X_train, y_train)
    y_train_pred = tree.predict(X_train)
    y_test_pred = tree.predict(X_test)

    tree_train = accuracy_score(y_train, y_train_pred)
    tree_test = accuracy_score(y_test, y_test_pred)
    print('Decision tree train/test accuracies %.3f/%.3f'
          % (tree_train, tree_test))

    bag = bag.fit(X_train, y_train)
    y_train_pred = bag.predict(X_train)
    y_test_pred = bag.predict(X_test)

    bag_train = accuracy_score(y_train, y_train_pred) 
    bag_test = accuracy_score(y_test, y_test_pred) 
    print('Bagging train/test accuracies %.3f/%.3f'
          % (bag_train, bag_test))

    x_min = X_train[:, 0].min() - 1
    x_max = X_train[:, 0].max() + 1
    y_min = X_train[:, 1].min() - 1
    y_max = X_train[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, (x_max-x_min)/100),
                         np.arange(y_min, y_max, (y_max-y_min)/100))

    f, axarr = plt.subplots(nrows=1, ncols=2, 
                            sharex='col', 
                            sharey='row', 
                            figsize=(8, 3))


    for idx, clf, tt in zip([0, 1],
                            [tree, bag],
                            ['Decision Tree', 'Bagging']):
        clf.fit(X_train, y_train)
        
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        axarr[idx].contourf(xx, yy, Z, alpha=0.3)
        axarr[idx].scatter(X_train[y_train==0, 0], 
                           X_train[y_train==0, 1], 
                           c='blue', marker='^')
        
        axarr[idx].scatter(X_train[y_train==1, 0], 
                           X_train[y_train==1, 1], 
                           c='red', marker='o')
        
        axarr[idx].set_title(tt)
        axarr[idx].set_ylabel(labels[0], fontsize=12)
        axarr[idx].set_xlabel(labels[1], fontsize=12)
    # plt.text(10.2, -1.2, 
    #          s='Hue', 
    #          ha='center', va='center', fontsize=12)
        
    plt.tight_layout()
    title='Bagging'
    #plt.savefig('./figures/bagging_region.png', 
    #            dpi=300, 
    #            bbox_inches='tight')
    ocr_utils.show_figures(plt, title)

    print ('\n########################### No Errors ####################################')

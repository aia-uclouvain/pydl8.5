from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from dl85 import DL85Booster, DL85Classifier, MODEL_LP_RATSCH, MODEL_LP_DEMIRIZ, MODEL_QP_MDBOOST
import time
import numpy as np
from sklearn.metrics import confusion_matrix
import os

depth = 3
n_trees = 100
percent_data = .4
percent_features = .5
reuse_data = True
reuse_features = True
use_unseen_as_test = True
keep_trees = False
seed = 0
n_folds = 5
verbose_level = 0

print("######################################################################\n"
      "#                   Bagging classifiers comparison                   #\n"
      "######################################################################\n")
file_out = open("out_depth_" + str(depth) + ".csv", "w")
directory = '../datasets'
for filename in sorted(os.listdir(directory)):
    if filename.endswith(".txt") and not filename.startswith("paper"):
        print("filename:", filename)
        dataset = np.genfromtxt("../datasets/" + filename, delimiter=' ')
        X, y = dataset[:, 1:], dataset[:, 0]
        to_write = [filename.split(".")[0], X.shape[1], X.shape[0], y.tolist().count(0), y.tolist().count(1)]

        print("=====> CV on DL8.5 <=====")
        clf_results = cross_validate(estimator=DL85Classifier(max_depth=depth), X=X, y=y, scoring='accuracy', cv=n_folds,
                                     n_jobs=-1, verbose=verbose_level, return_train_score=True, return_estimator=True, error_score=np.nan)
        print("Model built. Avg duration of building =", round(float(np.mean(clf_results['fit_time'])), 4))
        print("Accuracy on training set =", clf_results['train_score'], round(float(np.mean(clf_results['train_score'])), 4))
        print("Avg accuracy on test set =", clf_results['test_score'], round(float(np.mean(clf_results['test_score'])), 4))
        print("list of time :", clf_results['fit_time'])
        tmp_to_write = [[1, round(clf_results['fit_time'][k], 4), round(clf_results['train_score'][k], 4), round(clf_results['test_score'][k], 4)] for k in range(n_folds)]
        to_write += [val for sublist in tmp_to_write for val in sublist]
        print(tmp_to_write, "\n\n")

        for base_clf, name in zip([DL85Classifier(max_depth=depth), DecisionTreeClassifier(max_depth=depth)],
                                  ["DL8.5", "CART"]):
            print("=====> CV on Bagging + {} <=====".format(name))
            clf = BaggingClassifier(base_estimator=base_clf,
                                    n_estimators=n_trees,
                                    max_samples=percent_data,
                                    max_features=percent_features,
                                    bootstrap=reuse_data,
                                    bootstrap_features=reuse_features,
                                    oob_score=use_unseen_as_test,
                                    warm_start=keep_trees,
                                    n_jobs=-1,
                                    random_state=seed)
            clf_results = cross_validate(estimator=clf, X=X, y=y, scoring='accuracy', cv=n_folds, n_jobs=-1, verbose=verbose_level,
                                         return_train_score=True, return_estimator=True, error_score=np.nan)
            print("Model built. Avg duration of building =", round(float(np.mean(clf_results['fit_time'])), 4))
            print("Accuracy on training set =", clf_results['train_score'], round(float(np.mean(clf_results['train_score'])), 4))
            print("Avg accuracy on test set =", clf_results['test_score'], round(float(np.mean(clf_results['test_score'])), 4))
            print("list of time :", clf_results['fit_time'])
            # print("n_trees:", list(map(lambda x: len(x), clf_results['estimator'])))
            tmp_to_write = [[len(clf_results['estimator'][k].estimators_), round(clf_results['fit_time'][k], 4), round(clf_results['train_score'][k], 4), round(clf_results['test_score'][k], 4)] for k in range(n_folds)]
            to_write += [val for sublist in tmp_to_write for val in sublist]
            print(tmp_to_write, "\n\n")

        print("=====> CV on Bagging + RF <=====")
        clf = RandomForestClassifier(max_depth=depth,
                                     n_estimators=n_trees,
                                     random_state=seed,
                                     max_features=percent_features,
                                     bootstrap=reuse_data,
                                     oob_score=use_unseen_as_test,
                                     warm_start=keep_trees,
                                     max_samples=percent_data
                                     )
        clf_results = cross_validate(estimator=clf, X=X, y=y, scoring='accuracy', cv=n_folds, n_jobs=-1, verbose=verbose_level,
                                     return_train_score=True, return_estimator=True, error_score=np.nan)
        print("Model built. Avg duration of building =", round(float(np.mean(clf_results['fit_time'])), 4))
        print("Accuracy on training set =", clf_results['train_score'], round(float(np.mean(clf_results['train_score'])), 4))
        print("Avg accuracy on test set =", clf_results['test_score'], round(float(np.mean(clf_results['test_score'])), 4))
        print("list of time :", clf_results['fit_time'])
        tmp_to_write = [[len(clf_results['estimator'][k].estimators_), round(clf_results['fit_time'][k], 4), round(clf_results['train_score'][k], 4), round(clf_results['test_score'][k], 4)] for k in range(n_folds)]
        to_write += [val for sublist in tmp_to_write for val in sublist]
        print(tmp_to_write, "\n\n")

        print(to_write, "\n\n")
        file_out.write(";".join(map(lambda x: str(x), to_write)) + "\n")
        file_out.flush()

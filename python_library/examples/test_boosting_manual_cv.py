"""
======================
Default DL85Classifier
======================

"""
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import GridSearchCV
import time
import warnings
from dl85 import DL85Booster, DL85Classifier, BOOST_SVM1, BOOST_SVM2
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.exceptions import FitFailedWarning
import os
import sys

# filename = "letter"
# dataset = np.genfromtxt("../datasets/" + filename + ".txt", delimiter=' ')

N_FOLDS, MAX_DEPTH, MIN_SUP = 5, int(sys.argv[1]) if len(sys.argv) > 1 else 1, 1
MAX_ITERATIONS, MAX_TREES, TIME_LIMIT = 0, 0, 600
VERBOSE_LEVEL = 10

file_out = open("out_depth_" + str(MAX_DEPTH) + ".csv", "a+")
directory = '../datasets'
for filename in sorted(os.listdir(directory)):
    if filename.endswith(".txt") and not filename.startswith("paper"):
        dataset = np.genfromtxt("../datasets/" + filename, delimiter=' ')

        X = dataset[:, 1:]
        y = dataset[:, 0]
        X = X.astype('int32')
        y = y.astype('int32')

        print("Dataset :", filename)
        print("size of 0 :", y.tolist().count(0), "size of 1 :", y.tolist().count(1))
        print("n_feat = ", X.shape[1], "n_trans = ", X.shape[0])

        to_write = [filename.split(".")[0], X.shape[1], X.shape[0], y.tolist().count(0), y.tolist().count(1)]

        X_trains, y_trains = [], []
        X_tests, y_tests = [], []
        # kf = KFold(n_splits=N_FOLDS, random_state=42, shuffle=True)
        kf = StratifiedKFold(n_splits=N_FOLDS)
        for train_index, test_index in kf.split(X, y):
            X_trains.append(X[train_index])
            y_trains.append(y[train_index])
            X_tests.append(X[test_index])
            y_tests.append(y[test_index])

        parameters = {'regulator': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 10, 100, 1000, 10000, 100000, 1000000]}

        print("######################################################################\n"
              "#                                START                               #\n"
              "######################################################################")

        print("DL8.5")
        print("Dataset :", filename)
        clf_results = cross_validate(estimator=DL85Classifier(max_depth=MAX_DEPTH, min_sup=MIN_SUP, time_limit=TIME_LIMIT), X=X, y=y, scoring='accuracy',
                                     cv=N_FOLDS, n_jobs=-1, verbose=VERBOSE_LEVEL, return_train_score=True, return_estimator=True, error_score=np.nan)
        n_trees = [1 for k in range(N_FOLDS)]
        fps = [len([i for i in [j for j, val in enumerate(clf_results['estimator'][k].predict(X_tests[k])) if val == 1] if y_tests[k][i] != 1]) for k in range(N_FOLDS)]
        fns = [len([i for i in [j for j, val in enumerate(clf_results['estimator'][k].predict(X_tests[k])) if val == 0] if y_tests[k][i] != 0]) for k in range(N_FOLDS)]
        print("Model built. Avg duration of building =", round(float(np.mean(clf_results['fit_time'])), 4))
        print("Avg number of trees =", round(float(np.mean(n_trees)), 4))
        print("Avg accuracy on training set =", round(float(np.mean(clf_results['train_score'])), 4))
        print("Avg accuracy on test set =", round(float(np.mean(clf_results['test_score'])), 4))
        print("list of time :", clf_results['fit_time'])
        print("sum false positives =", sum(fps))
        print("sum false negatives =", sum(fns), "\n\n\n")
        tmp_to_write = [[n_trees[k], n_trees[k], clf_results['fit_time'][k], not clf_results['estimator'][k].timeout_, clf_results['train_score'][k], clf_results['test_score'][k], fps[k], fns[k], -1] for k in range(N_FOLDS)]
        to_write += [val for sublist in tmp_to_write for val in sublist]

        print("CART")
        print("Dataset :", filename)
        clf_results = cross_validate(estimator=DecisionTreeClassifier(max_depth=MAX_DEPTH, min_samples_leaf=MIN_SUP), X=X, y=y, scoring='accuracy',
                                     cv=N_FOLDS, n_jobs=-1, verbose=VERBOSE_LEVEL, return_train_score=True, return_estimator=True, error_score=np.nan)
        n_trees = [1 for k in range(N_FOLDS)]
        fps = [len([i for i in [j for j, val in enumerate(clf_results['estimator'][k].predict(X_tests[k])) if val == 1] if y_tests[k][i] != 1]) for k in range(N_FOLDS)]
        fns = [len([i for i in [j for j, val in enumerate(clf_results['estimator'][k].predict(X_tests[k])) if val == 0] if y_tests[k][i] != 0]) for k in range(N_FOLDS)]
        print("Model built. Avg duration of building =", round(float(np.mean(clf_results['fit_time'])), 4))
        print("Avg number of trees =", round(float(np.mean(n_trees)), 4))
        print("Avg accuracy on training set =", round(float(np.mean(clf_results['train_score'])), 4))
        print("Avg accuracy on test set =", round(float(np.mean(clf_results['test_score'])), 4))
        print("list of time :", clf_results['fit_time'])
        print("sum false positives =", sum(fps))
        print("sum false negatives =", sum(fns), "\n\n\n")
        tmp_to_write = [[n_trees[k], n_trees[k], clf_results['fit_time'][k], True, clf_results['train_score'][k], clf_results['test_score'][k], fps[k], fns[k], -1] for k in range(N_FOLDS)]
        to_write += [val for sublist in tmp_to_write for val in sublist]

        print("LPBoost + DL8.5")
        print("Dataset :", filename)
        print("Search for the best regulator using grid search...", MAX_TREES)
        # each regulator is tested without constraint on trees numbers
        gd_sr = GridSearchCV(estimator=DL85Booster(max_depth=MAX_DEPTH, min_sup=MIN_SUP, time_limit=TIME_LIMIT, max_estimators=MAX_TREES, model=BOOST_SVM2),
                             param_grid=parameters, scoring='accuracy', cv=N_FOLDS, n_jobs=-1, verbose=VERBOSE_LEVEL)
        gd_sr.fit(X, y)
        print()
        print("Running cross validation for LPBoost + DL8.5 with best regulator =", gd_sr.best_params_["regulator"], "on", filename)
        clf_results = cross_validate(estimator=DL85Booster(max_depth=MAX_DEPTH, min_sup=MIN_SUP, time_limit=TIME_LIMIT, model=BOOST_SVM2,
                                     max_estimators=MAX_TREES, regulator=gd_sr.best_params_["regulator"]), X=X, y=y, scoring='accuracy',
                                     cv=N_FOLDS, n_jobs=-1, verbose=VERBOSE_LEVEL, return_train_score=True, return_estimator=True, error_score=np.nan)
        n_trees = list(map(lambda clf: clf.n_estimators_, clf_results['estimator']))
        fps = [len([i for i in [j for j, val in enumerate(clf_results['estimator'][k].predict(X_tests[k])) if val == 1] if y_tests[k][i] != 1]) for k in range(N_FOLDS)]
        fns = [len([i for i in [j for j, val in enumerate(clf_results['estimator'][k].predict(X_tests[k])) if val == 0] if y_tests[k][i] != 0]) for k in range(N_FOLDS)]
        max_estimators = int(sum(n_trees)/len(n_trees))
        print("Model built. Avg duration of building =", round(float(np.mean(clf_results['fit_time'])), 4))
        print("Avg number of trees =", round(float(np.mean(n_trees)), 4))
        print("Avg accuracy on training set =", round(float(np.mean(clf_results['train_score'])), 4))
        print("Avg accuracy on test set =", round(float(np.mean(clf_results['test_score'])), 4))
        print("list of optimality :", list(map(lambda clf: clf.optimal_, clf_results['estimator'])))
        print("list of iterations :", list(map(lambda clf: clf.n_iterations_, clf_results['estimator'])))
        print("list of time :", clf_results['fit_time'])
        print("sum false positives =", sum(fps))
        print("sum false negatives =", sum(fns), "\n\n\n")
        tmp_to_write = [[clf_results['estimator'][k].n_iterations_, clf_results['estimator'][k].n_estimators_, clf_results['fit_time'][k], clf_results['estimator'][k].optimal_, clf_results['train_score'][k], clf_results['test_score'][k], fps[k], fns[k], gd_sr.best_params_["regulator"]] for k in range(N_FOLDS)]
        to_write += [val for sublist in tmp_to_write for val in sublist]

        print("LPBoost + CART")
        print("Dataset :", filename)
        print("Search for the best regulator using grid search...")
        gd_sr = GridSearchCV(estimator=DL85Booster(base_estimator=DecisionTreeClassifier(max_depth=MAX_DEPTH,
                             min_samples_leaf=MIN_SUP), time_limit=TIME_LIMIT, max_estimators=max_estimators, model=BOOST_SVM2), param_grid=parameters,
                             scoring='accuracy', cv=N_FOLDS, n_jobs=-1, verbose=VERBOSE_LEVEL)
        gd_sr.fit(X, y)
        print()
        print("Running cross validation for LPBoost + CART with best regulator =", gd_sr.best_params_["regulator"], "on", filename)
        clf_results = cross_validate(estimator=DL85Booster(base_estimator=DecisionTreeClassifier(max_depth=MAX_DEPTH, min_samples_leaf=MIN_SUP), model=BOOST_SVM2,
                                     time_limit=TIME_LIMIT, max_estimators=max_estimators, regulator=gd_sr.best_params_["regulator"]), X=X, y=y, scoring='accuracy',
                                     cv=N_FOLDS, n_jobs=-1, verbose=VERBOSE_LEVEL, return_train_score=True, return_estimator=True, error_score=np.nan)
        n_trees = list(map(lambda clf: clf.n_estimators_, clf_results['estimator']))
        fps = [len([i for i in [j for j, val in enumerate(clf_results['estimator'][k].predict(X_tests[k])) if val == 1] if y_tests[k][i] != 1]) for k in range(N_FOLDS)]
        fns = [len([i for i in [j for j, val in enumerate(clf_results['estimator'][k].predict(X_tests[k])) if val == 0] if y_tests[k][i] != 0]) for k in range(N_FOLDS)]
        print("Model built. Avg duration of building =", round(float(np.mean(clf_results['fit_time'])), 4))
        print("Avg number of trees =", round(float(np.mean(n_trees)), 4))
        print("Avg accuracy on training set =", round(float(np.mean(clf_results['train_score'])), 4))
        print("Avg accuracy on test set =", round(float(np.mean(clf_results['test_score'])), 4))
        print("sum false positives =", sum(fps))
        print("sum false negatives =", sum(fns), "\n\n\n")
        tmp_to_write = [[clf_results['estimator'][k].n_iterations_, clf_results['estimator'][k].n_estimators_, clf_results['fit_time'][k], clf_results['estimator'][k].optimal_, clf_results['train_score'][k], clf_results['test_score'][k], fps[k], fns[k], gd_sr.best_params_["regulator"]] for k in range(N_FOLDS)]
        to_write += [val for sublist in tmp_to_write for val in sublist]

        print("Adaboost + DL8.5")
        print("Dataset :", filename)
        print("Running cross validation")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FitFailedWarning)
            clf_results = cross_validate(estimator=AdaBoostClassifier(base_estimator=DL85Classifier(max_depth=MAX_DEPTH, min_sup=MIN_SUP, time_limit=TIME_LIMIT), algorithm="SAMME",
                                                                      n_estimators=max_estimators), X=X, y=y, scoring='accuracy', cv=N_FOLDS, n_jobs=-1, verbose=VERBOSE_LEVEL, return_train_score=True,
                                         return_estimator=True, error_score=np.nan)
        n_trees = [np.nan if np.isnan(clf_results['train_score'][k]) else len(clf_results['estimator'][k].estimators_) for k in range(N_FOLDS)]
        fps, fns = [], []
        for k in range(N_FOLDS):
            if np.isnan(clf_results['train_score'][k]):
                fps.append(np.nan)
                fns.append(np.nan)
            else:
                fps.append(len([i for i in [j for j, val in enumerate(clf_results['estimator'][k].predict(X_tests[k])) if val == 1] if y_tests[k][i] != 1]))
                fns.append(len([i for i in [j for j, val in enumerate(clf_results['estimator'][k].predict(X_tests[k])) if val == 0] if y_tests[k][i] != 0]))

        print("Model built. Avg duration of building =", round(float(np.mean(clf_results['fit_time'])), 4))
        print("Avg number of trees =", round(float(np.mean(n_trees)), 4))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            print("Avg accuracy on training set =", np.nan if np.isnan(fps[0]) and len(set(fps)) == 1 else round(float(np.nanmean(clf_results['train_score'])), 4))
            print("Avg accuracy on test set =", np.nan if np.isnan(fps[0]) and len(set(fps)) == 1 else round(float(np.nanmean(clf_results['test_score'])), 4))
        print("sum false positives =", np.nan if np.isnan(fps[0]) and len(set(fps)) == 1 else np.nansum(fps))
        print("sum false negatives =", sum(fns), "\n\n\n")
        tmp_to_write = [[n_trees[k], n_trees[k], clf_results['fit_time'][k], True, clf_results['train_score'][k], clf_results['test_score'][k], fps[k], fns[k], -1] for k in range(N_FOLDS)]
        to_write += [val for sublist in tmp_to_write for val in sublist]

        print("Adaboost + CART")
        print("Dataset :", filename)
        print("Running cross validation")
        clf_results = cross_validate(estimator=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=MAX_DEPTH, min_samples_leaf=MIN_SUP),
                                     n_estimators=max_estimators), X=X, y=y, scoring='accuracy', cv=N_FOLDS, n_jobs=-1, verbose=VERBOSE_LEVEL,
                                     return_train_score=True, return_estimator=True, error_score=np.nan)
        n_trees = list(map(lambda clf: len(clf.estimators_), clf_results['estimator']))
        fps = [len([i for i in [j for j, val in enumerate(clf_results['estimator'][k].predict(X_tests[k])) if val == 1] if y_tests[k][i] != 1]) for k in range(N_FOLDS)]
        fns = [len([i for i in [j for j, val in enumerate(clf_results['estimator'][k].predict(X_tests[k])) if val == 0] if y_tests[k][i] != 0]) for k in range(N_FOLDS)]
        print("Model built. Avg duration of building =", round(float(np.mean(clf_results['fit_time'])), 4))
        print("Avg number of trees =", round(float(np.mean(n_trees)), 4))
        print("Avg accuracy on training set =", round(float(np.mean(clf_results['train_score'])), 4))
        print("Avg accuracy on test set =", round(float(np.mean(clf_results['test_score'])), 4))
        print("sum false positives =", sum(fps))
        print("sum false negatives =", sum(fns), "\n\n\n")
        tmp_to_write = [[n_trees[k], n_trees[k], clf_results['fit_time'][k], True, clf_results['train_score'][k], clf_results['test_score'][k], fps[k], fns[k], -1] for k in range(N_FOLDS)]
        to_write += [val for sublist in tmp_to_write for val in sublist]

        print("Adaboost + CART 50")
        print("Dataset :", filename)
        print("Running cross validation")
        clf_results = cross_validate(estimator=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=MAX_DEPTH, min_samples_leaf=MIN_SUP),
                                                                  n_estimators=50), X=X, y=y, scoring='accuracy', cv=N_FOLDS, n_jobs=-1, verbose=VERBOSE_LEVEL,
                                     return_train_score=True, return_estimator=True, error_score=np.nan)
        n_trees = list(map(lambda clf: len(clf.estimators_), clf_results['estimator']))
        fps = [len([i for i in [j for j, val in enumerate(clf_results['estimator'][k].predict(X_tests[k])) if val == 1] if y_tests[k][i] != 1]) for k in range(N_FOLDS)]
        fns = [len([i for i in [j for j, val in enumerate(clf_results['estimator'][k].predict(X_tests[k])) if val == 0] if y_tests[k][i] != 0]) for k in range(N_FOLDS)]
        print("Model built. Avg duration of building =", round(float(np.mean(clf_results['fit_time'])), 4))
        print("Avg number of trees =", round(float(np.mean(n_trees)), 4))
        print("Avg accuracy on training set =", round(float(np.mean(clf_results['train_score'])), 4))
        print("Avg accuracy on test set =", round(float(np.mean(clf_results['test_score'])), 4))
        print("sum false positives =", sum(fps))
        print("sum false negatives =", sum(fns), "\n\n\n")
        tmp_to_write = [[n_trees[k], n_trees[k], clf_results['fit_time'][k], True, clf_results['train_score'][k], clf_results['test_score'][k], fps[k], fns[k], -1] for k in range(N_FOLDS)]
        to_write += [val for sublist in tmp_to_write for val in sublist]

        print("Adaboost + CART 100")
        print("Dataset :", filename)
        print("Running cross validation")
        clf_results = cross_validate(estimator=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=MAX_DEPTH, min_samples_leaf=MIN_SUP),
                                                                  n_estimators=100), X=X, y=y, scoring='accuracy', cv=N_FOLDS, n_jobs=-1, verbose=VERBOSE_LEVEL,
                                     return_train_score=True, return_estimator=True, error_score=np.nan)
        n_trees = list(map(lambda clf: len(clf.estimators_), clf_results['estimator']))
        fps = [len([i for i in [j for j, val in enumerate(clf_results['estimator'][k].predict(X_tests[k])) if val == 1] if y_tests[k][i] != 1]) for k in range(N_FOLDS)]
        fns = [len([i for i in [j for j, val in enumerate(clf_results['estimator'][k].predict(X_tests[k])) if val == 0] if y_tests[k][i] != 0]) for k in range(N_FOLDS)]
        print("Model built. Avg duration of building =", round(float(np.mean(clf_results['fit_time'])), 4))
        print("Avg number of trees =", round(float(np.mean(n_trees)), 4))
        print("Avg accuracy on training set =", round(float(np.mean(clf_results['train_score'])), 4))
        print("Avg accuracy on test set =", round(float(np.mean(clf_results['test_score'])), 4))
        print("sum false positives =", sum(fps))
        print("sum false negatives =", sum(fns), "\n\n\n")
        tmp_to_write = [[n_trees[k], n_trees[k], clf_results['fit_time'][k], True, clf_results['train_score'][k], clf_results['test_score'][k], fps[k], fns[k], -1] for k in range(N_FOLDS)]
        to_write += [val for sublist in tmp_to_write for val in sublist]

        print("Gradient Boosting")
        print("Dataset :", filename)
        print("Running cross validation")
        clf_results = cross_validate(estimator=GradientBoostingClassifier(max_depth=MAX_DEPTH, min_samples_leaf=MIN_SUP, n_estimators=max_estimators),
                                     X=X, y=y, scoring='accuracy', cv=N_FOLDS, n_jobs=-1, verbose=VERBOSE_LEVEL, return_train_score=True,
                                     return_estimator=True, error_score=np.nan)
        n_trees = list(map(lambda clf: clf.n_estimators_, clf_results['estimator']))
        fps = [len([i for i in [j for j, val in enumerate(clf_results['estimator'][k].predict(X_tests[k])) if val == 1] if y_tests[k][i] != 1]) for k in range(N_FOLDS)]
        fns = [len([i for i in [j for j, val in enumerate(clf_results['estimator'][k].predict(X_tests[k])) if val == 0] if y_tests[k][i] != 0]) for k in range(N_FOLDS)]
        print("Model built. Avg duration of building =", round(float(np.mean(clf_results['fit_time'])), 4))
        print("Avg number of trees =", round(float(np.mean(n_trees)), 4))
        print("Avg accuracy on training set =", round(float(np.mean(clf_results['train_score'])), 4))
        print("Avg accuracy on test set =", round(float(np.mean(clf_results['test_score'])), 4))
        print("sum false positives =", sum(fps))
        print("sum false negatives =", sum(fns), "\n\n\n")
        tmp_to_write = [[n_trees[k], n_trees[k], clf_results['fit_time'][k], True, clf_results['train_score'][k], clf_results['test_score'][k], fps[k], fns[k], -1] for k in range(N_FOLDS)]
        to_write += [val for sublist in tmp_to_write for val in sublist]

        print("Random Forest")
        print("Dataset :", filename)
        print("Running cross validation")
        clf_results = cross_validate(estimator=RandomForestClassifier(max_depth=MAX_DEPTH, min_samples_leaf=MIN_SUP, n_estimators=max_estimators),
                                     X=X, y=y, scoring='accuracy', cv=N_FOLDS, n_jobs=-1, verbose=VERBOSE_LEVEL, return_train_score=True,
                                     return_estimator=True, error_score=np.nan)
        n_trees = list(map(lambda clf: len(clf.estimators_), clf_results['estimator']))
        fps = [len([i for i in [j for j, val in enumerate(clf_results['estimator'][k].predict(X_tests[k])) if val == 1] if y_tests[k][i] != 1]) for k in range(N_FOLDS)]
        fns = [len([i for i in [j for j, val in enumerate(clf_results['estimator'][k].predict(X_tests[k])) if val == 0] if y_tests[k][i] != 0]) for k in range(N_FOLDS)]
        print("Model built. Avg duration of building =", round(float(np.mean(clf_results['fit_time'])), 4))
        print("Avg number of trees =", round(float(np.mean(n_trees)), 4))
        print("Avg accuracy on training set =", round(float(np.mean(clf_results['train_score'])), 4))
        print("Avg accuracy on test set =", round(float(np.mean(clf_results['test_score'])), 4))
        print("sum false positives =", sum(fps))
        print("sum false negatives =", sum(fns), "\n\n\n")
        tmp_to_write = [[n_trees[k], n_trees[k], clf_results['fit_time'][k], True, clf_results['train_score'][k], clf_results['test_score'][k], fps[k], fns[k], -1] for k in range(N_FOLDS)]
        to_write += [val for sublist in tmp_to_write for val in sublist]

        file_out.write(";".join(map(lambda x: str(x), to_write)) + "\n")
        file_out.flush()
        print(to_write)
file_out.close()

"""
======================
Default DL85Classifier
======================

"""
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import GridSearchCV
import time
import warnings
from dl85 import DL85Booster, DL85Classifier, MODEL_RATSCH, MODEL_DEMIRIZ, DL85Boostera
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.exceptions import FitFailedWarning
import os
import sys

# filename = "letter"
# dataset = np.genfromtxt("../datasets/" + filename + ".txt", delimiter=' ')

N_FOLDS, N_FOLDS_TUNING, MAX_DEPTH, MIN_SUP, MODEL = 5, 4, int(sys.argv[1]) if len(sys.argv) > 1 else 1, 1, 'gurobi' if len(sys.argv) > 4 else 'cvxpy'
# N_FOLDS, N_FOLDS_TUNING, MAX_DEPTH, MIN_SUP, MODEL = 5, 4, int(sys.argv[1]) if len(sys.argv) > 1 else 1, 1, MODEL_RATSCH if len(sys.argv) > 4 else MODEL_DEMIRIZ
MAX_ITERATIONS, MAX_TREES, TIME_LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 0, 0, 0
VERBOSE_LEVEL = 10
first_file = sys.argv[3] + '.txt' if len(sys.argv) > 3 else 'zoo-1.txt'

# depth max_iter first_file model

file_out = open("../output/out_md_each_" + ((str(MAX_ITERATIONS) + '_') if MAX_ITERATIONS > 0 else '') + "depth_" + str(MAX_DEPTH) + "_none.csv", "a+")
#directory = '../datasets'
#directory = '../datasets/boosting/mm'
files = ['zoo-1.txt', 'hepatitis.txt', 'lymph.txt', 'audiology.txt', 'heart-cleveland.txt', 'primary-tumor.txt', 'tic-tac-toe.txt', 'vote.txt', 'soybean.txt',
         'anneal.txt', 'yeast.txt', 'australian-credit.txt', 'breast-wisconsin.txt', 'diabetes.txt', 'german-credit.txt', 'kr-vs-kp.txt', 'hypothyroid.txt',
         'mushroom.txt', 'vehicle.txt', 'ionosphere.txt', 'segment.txt', 'splice-1.txt', 'pendigits.txt', 'letter.txt']
# for filename in sorted(os.listdir(directory)):

for filename in files[files.index(first_file):]:
    #for filename in ['matchmaker.txt']:
    #for filename in ['kr-vs-kp.txt', 'splice-1.txt', 'yeast.txt', 'hypothyroid.txt', 'letter.txt', 'pendigits.txt', 'segment.txt']:
    # if filename.endswith(".txt") and not filename.startswith("paper") and not filename.startswith("anneal") and not filename.startswith("audiology"):
    if True:
        dataset = np.genfromtxt("../datasets/" + filename, delimiter=' ')
        #dataset = np.genfromtxt("../datasets/boosting/mm/" + filename, delimiter=' ')

        X = dataset[:, 1:]
        y = dataset[:, 0]
        X = X.astype('int32')
        y = y.astype('int32')
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        kf = StratifiedKFold(n_splits=N_FOLDS)
        X_trains, X_tests, y_trains, y_tests = [], [], [], []
        train_indices, test_indices = [], []
        for train_index, test_index in kf.split(X, y):
            if X.shape[0] <= 1000:  # 80%(tr) - 20%(te)
                train_indices.append(train_index)
                test_indices.append(test_index)
                X_trains.append(X[train_index])
                y_trains.append(y[train_index])
                X_tests.append(X[test_index])
                y_tests.append(y[test_index])
            else:  # 700(tr) - remaining(te)
                kk = StratifiedShuffleSplit(n_splits=2, train_size=800, random_state=0)
                for tr_i, te_i in kk.split(X[train_index], y[train_index]):
                    train_indices.append(train_index[tr_i])
                    test_indices.append(np.concatenate((train_index[te_i], test_index)))
                    X_trains.append(X[train_index[tr_i]])
                    y_trains.append(y[train_index[tr_i]])
                    X_tests.append(X[np.concatenate((train_index[te_i], test_index))])
                    y_tests.append(y[np.concatenate((train_index[te_i], test_index))])
                    break
                # X_trains.append(X[train_index[:len(train_index)//2]])
                #train_indices.append(train_index[:800])
                #test_indices.append(np.concatenate((train_index[-800:], test_index)))
                #X_trains.append(X[train_index[:800]])
                #y_trains.append(y[train_index[:800]])
                #X_tests.append(X[np.concatenate((train_index[-800:], test_index))])
                #y_tests.append(y[np.concatenate((train_index[-800:], test_index))])
        custom_cv_dl85 = zip(train_indices, test_indices)
        custom_cv_cart = zip(train_indices, test_indices)

        print("Dataset :", filename)
        print("size of 0 :", y.tolist().count(0), "size of 1 :", y.tolist().count(1))
        print("n_feat = ", X.shape[1], "n_trans = ", X.shape[0])

        to_write = [filename.split(".")[0], X.shape[1], X.shape[0], y.tolist().count(0), y.tolist().count(1)]

        # parameters = {'regulator': [2, 5, 8, 10, 12, 15, 20, 30, 40, 50, 70, 90, 100, 120]}
        # parameters = {'regulator': [2, 5, 8, 10, 12, 15, 20, 30, 40, 50, 70, 90, 100, 120], 'gamma': [None, 'auto', 'scale', 'nscale', 1, 0.1, 0.01, 0.001, 0.0001]}
        # parameters = {'regulator': list(map(lambda x: pow(2, x), list(range(-5, 16)))), 'gamma': [None] + list(map(lambda x: pow(2, x), list(range(-15, 4))))}
        # parameters = {'regulator': [0.01, 0.1, 1, 2, 5, 8, 10, 20, 40, 70, 100, 150], 'gamma': [None, 'auto', 'scale', 'nscale']}
        #parameters = {'regulator': [0.01, 0.1, 1, 2, 5, 8, 10, 15, 20, 30, 40, 50, 70, 90, 100, 150]}
        #parameters = {'regulator': [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 10, 100, 1000, 10000, 100000]}
        #parameters = {'regulator': [0.01, 0.1, 0.3, 0.6, 0.8, 1, 5, 7, 10, 20, 30, 50, 80, 100, 150, 1000]}
        # parameters = {'regulator': [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
        # regulators = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]  # ratsch
        regulators = [0.01, 0.1, 1, 2, 5, 8, 10, 15, 20, 30, 40, 50, 70, 90, 100, 150]  # mdboost
        # regulators = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 10, 100, 1000, 10000, 100000]  # demiriz

        print("######################################################################\n"
              "#                                START                               #\n"
              "######################################################################")

        print("DL8.5")
        print("Dataset :", filename)
        clf_results = cross_validate(estimator=DL85Classifier(max_depth=MAX_DEPTH), X=X, y=y, scoring='accuracy',
                                     cv=custom_cv_dl85, n_jobs=-1, verbose=VERBOSE_LEVEL, return_train_score=True, return_estimator=True, error_score=np.nan)
        n_trees = [1 for k in range(N_FOLDS)]
        fps = [len([i for i in [j for j, val in enumerate(clf_results['estimator'][k].predict(X_tests[k])) if val == 1] if y_tests[k][i] != 1]) for k in range(N_FOLDS)]
        fns = [len([i for i in [j for j, val in enumerate(clf_results['estimator'][k].predict(X_tests[k])) if val == 0] if y_tests[k][i] != 0]) for k in range(N_FOLDS)]
        print("Model built. Avg duration of building =", round(float(np.mean(clf_results['fit_time'])), 4))
        print("Avg number of trees =", round(float(np.mean(n_trees)), 4))
        print("Accuracy on training set =", clf_results['train_score'], round(float(np.mean(clf_results['train_score'])), 4))
        print("Avg accuracy on test set =", clf_results['test_score'], round(float(np.mean(clf_results['test_score'])), 4))
        print("list of time :", clf_results['fit_time'])
        print("list of n_nodes :", [clf_results['estimator'][k].get_nodes_count() for k in range(N_FOLDS)])
        print("sum false positives =", sum(fps))
        print("sum false negatives =", sum(fns), "\n\n\n")
        tmp_to_write = [[n_trees[k], n_trees[k], clf_results['fit_time'][k], not clf_results['estimator'][k].timeout_, clf_results['train_score'][k], clf_results['test_score'][k], fps[k], fns[k], clf_results['estimator'][k].get_nodes_count(), -1, -1] for k in range(N_FOLDS)]
        to_write += [val for sublist in tmp_to_write for val in sublist]

        print("CART")
        print("Dataset :", filename)
        clf_results = cross_validate(estimator=DecisionTreeClassifier(max_depth=MAX_DEPTH), X=X, y=y, scoring='accuracy',
                                     cv=custom_cv_cart, n_jobs=-1, verbose=VERBOSE_LEVEL, return_train_score=True, return_estimator=True, error_score=np.nan)
        n_trees = [1 for k in range(N_FOLDS)]
        fps = [len([i for i in [j for j, val in enumerate(clf_results['estimator'][k].predict(X_tests[k])) if val == 1] if y_tests[k][i] != 1]) for k in range(N_FOLDS)]
        fns = [len([i for i in [j for j, val in enumerate(clf_results['estimator'][k].predict(X_tests[k])) if val == 0] if y_tests[k][i] != 0]) for k in range(N_FOLDS)]
        print("Model built. Avg duration of building =", round(float(np.mean(clf_results['fit_time'])), 4))
        print("Avg number of trees =", round(float(np.mean(n_trees)), 4))
        print("Accuracy on training set =", clf_results['train_score'], round(float(np.mean(clf_results['train_score'])), 4))
        print("Avg accuracy on test set =", clf_results['test_score'], round(float(np.mean(clf_results['test_score'])), 4))
        print("list of time :", clf_results['fit_time'])
        print("list of n_nodes :", [clf_results['estimator'][k].tree_.node_count for k in range(N_FOLDS)])
        print("sum false positives =", sum(fps))
        print("sum false negatives =", sum(fns), "\n\n\n")
        tmp_to_write = [[n_trees[k], n_trees[k], clf_results['fit_time'][k], True, clf_results['train_score'][k], clf_results['test_score'][k], fps[k], fns[k], clf_results['estimator'][k].tree_.node_count, -1, -1] for k in range(N_FOLDS)]
        to_write += [val for sublist in tmp_to_write for val in sublist]

        for reg in regulators:
            print("LPBoost + DL8.5")
            print("Dataset :", filename)
            print("CV - Running cross validation for LPBoost + DL8.5 with best regulator =", reg, "on", filename)
            clf_results = cross_validate(estimator=DL85Boostera(max_depth=MAX_DEPTH, model=MODEL, max_iterations=MAX_ITERATIONS, regulator=reg), X=X, y=y,
                                   error_score=np.nan, scoring='accuracy', cv=zip(train_indices, test_indices), n_jobs=-1, verbose=VERBOSE_LEVEL, return_train_score=True, return_estimator=True)
            print()
            n_trees = list(map(lambda x: x.n_estimators_, [clf for clf in clf_results['estimator']]))
            n_iter = list(map(lambda x: x.n_iterations_, [clf for clf in clf_results['estimator']]))
            train_scores = clf_results['train_score'][:]
            test_scores = clf_results['test_score'][:]
            fit_times = list(map(lambda x: x.duration_, [clf for clf in clf_results['estimator']]))
            n_nodes = list(map(lambda x: x.get_nodes_count(), [clf for clf in clf_results['estimator']]))
            n_opti = list(map(lambda x: x.optimal_, [clf for clf in clf_results['estimator']]))
            fps = [len([i for i in [j for j, val in enumerate(clf_results['estimator'][k].predict(X_tests[k])) if val == 1] if y_tests[k][i] != 1]) for k in range(N_FOLDS)]
            fns = [len([i for i in [j for j, val in enumerate(clf_results['estimator'][k].predict(X_tests[k])) if val == 0] if y_tests[k][i] != 0]) for k in range(N_FOLDS)]
            print("Model built. Avg duration of building =", round(float(np.mean(clf_results['fit_time'])), 4))
            print("Avg number of trees =", n_trees, round(float(np.mean(n_trees)), 4))
            print("Accuracy on training set =", train_scores, round(float(np.mean(train_scores)), 4))
            print("Avg accuracy on test set =", test_scores, round(float(np.mean(test_scores)), 4))
            print("list of time :", fit_times)
            print("list of n_nodes :", n_nodes)
            print("number of optimality :", n_opti)
            print("list of iterations :", n_iter)
            print("list of time :", fit_times)
            print("sum false positives =", sum(fps))
            print("sum false negatives =", sum(fns), "\n\n\n")
            tmp_to_write = [[n_iter[k], n_trees[k], fit_times[k], n_opti[k], train_scores[k], test_scores[k], fps[k], fns[k], n_nodes[k], -1, -1] for k in range(N_FOLDS)]
            to_write += [val for sublist in tmp_to_write for val in sublist]
            max_estimators = n_trees[:]

            print("Adaboost + DL8.5")
            print("Dataset :", filename)
            print("Running cross validation")
            train_accs, test_accs, n_trees, fps, fns, fit_times, n_nodes = [], [], [], [], [], [], []
            for k in range(N_FOLDS):
                X_train, X_test, y_train, y_test = X_trains[k], X_tests[k], y_trains[k], y_tests[k]
                clf = AdaBoostClassifier(base_estimator=DL85Classifier(max_depth=MAX_DEPTH), n_estimators=max_estimators[k])
                start = time.perf_counter()
                clf.fit(X_train, y_train)
                fit_times.append(time.perf_counter() - start)
                y_pred = clf.predict(X_test)
                n_trees.append(len(clf.estimators_))
                train_accs.append(accuracy_score(y_train, clf.predict(X_train)))
                test_accs.append(accuracy_score(y_test, y_pred))
                n_nodes.append(sum([c.get_nodes_count() for c in clf.estimators_]))
                fps.append(len([i for i in [j for j, val in enumerate(y_pred) if val == 1] if y_test[i] != 1]))
                fns.append(len([i for i in [j for j, val in enumerate(y_pred) if val == 0] if y_test[i] != 0]))
            print("Model built. Avg duration of building =", round(float(np.mean(fit_times)), 4))
            print("Number of trees =", n_trees, np.mean(n_trees))
            print("Accuracy on training set =", train_accs, round(float(np.mean(train_accs)), 4))
            print("Accuracy on test set =", test_accs, round(float(np.mean(test_accs)), 4))
            print("list of n_nodes =", n_nodes)
            print("sum false positives =", sum(fps))
            print("sum false negatives =", sum(fns), "\n\n\n")
            tmp_to_write = [[n_trees[k], n_trees[k], fit_times[k], True, train_accs[k], test_accs[k], fps[k], fns[k], n_nodes[k], -1, -1] for k in range(N_FOLDS)]
            to_write += [val for sublist in tmp_to_write for val in sublist]

            print("Adaboost + CART")
            print("Dataset :", filename)
            print("Running cross validation")
            train_accs, test_accs, n_trees, fps, fns, fit_times, n_nodes = [], [], [], [], [], [], []
            for k in range(N_FOLDS):
                X_train, X_test, y_train, y_test = X_trains[k], X_tests[k], y_trains[k], y_tests[k]
                clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=MAX_DEPTH), n_estimators=max_estimators[k])
                start = time.perf_counter()
                clf.fit(X_train, y_train)
                fit_times.append(time.perf_counter() - start)
                y_pred = clf.predict(X_test)
                n_trees.append(len(clf.estimators_))
                train_accs.append(accuracy_score(y_train, clf.predict(X_train)))
                test_accs.append(accuracy_score(y_test, y_pred))
                n_nodes.append(sum([c.tree_.node_count for c in clf.estimators_]))
                fps.append(len([i for i in [j for j, val in enumerate(y_pred) if val == 1] if y_test[i] != 1]))
                fns.append(len([i for i in [j for j, val in enumerate(y_pred) if val == 0] if y_test[i] != 0]))
            print("Model built. Avg duration of building =", round(float(np.mean(fit_times)), 4))
            print("Number of trees =", n_trees, np.mean(n_trees))
            print("Accuracy on training set =", train_accs, round(float(np.mean(train_accs)), 4))
            print("Accuracy on test set =", test_accs, round(float(np.mean(test_accs)), 4))
            print("list of n_nodes =", n_nodes)
            print("sum false positives =", sum(fps))
            print("sum false negatives =", sum(fns), "\n\n\n")
            tmp_to_write = [[n_trees[k], n_trees[k], fit_times[k], True, train_accs[k], test_accs[k], fps[k], fns[k], n_nodes[k], -1, -1] for k in range(N_FOLDS)]
            to_write += [val for sublist in tmp_to_write for val in sublist]

            print("Gradient Boosting")
            print("Dataset :", filename)
            print("Running cross validation")
            train_accs, test_accs, n_trees, fps, fns, fit_times, n_nodes = [], [], [], [], [], [], []
            for k in range(N_FOLDS):
                X_train, X_test, y_train, y_test = X_trains[k], X_tests[k], y_trains[k], y_tests[k]
                clf = GradientBoostingClassifier(max_depth=MAX_DEPTH, n_estimators=max_estimators[k])
                start = time.perf_counter()
                try:
                    clf.fit(X_train, y_train)
                    t = time.perf_counter() - start
                    y_pred = clf.predict(X_test)
                    fit_times.append(t)
                    n_trees.append(clf.n_estimators_)
                    train_accs.append(accuracy_score(y_train, clf.predict(X_train)))
                    test_accs.append(accuracy_score(y_test, y_pred))
                    n_nodes.append(sum([c[0].tree_.node_count for c in clf.estimators_]))
                    fps.append(len([i for i in [j for j, val in enumerate(y_pred) if val == 1] if y_test[i] != 1]))
                    fns.append(len([i for i in [j for j, val in enumerate(y_pred) if val == 0] if y_test[i] != 0]))
                except:
                    fit_times.append(0)
                    n_trees.append(0)
                    train_accs.append(0)
                    test_accs.append(0)
                    n_nodes.append(0)
                    fps.append(0)
                    fns.append(0)
            print("Model built. Avg duration of building =", round(float(np.mean(fit_times)), 4))
            print("Number of trees =", n_trees, np.mean(n_trees))
            print("Accuracy on training set =", train_accs, round(float(np.mean(train_accs)), 4))
            print("Accuracy on test set =", test_accs, round(float(np.mean(test_accs)), 4))
            print("list of n_nodes =", n_nodes)
            print("sum false positives =", sum(fps))
            print("sum false negatives =", sum(fns), "\n\n\n")
            tmp_to_write = [[n_trees[k], n_trees[k], fit_times[k], True, train_accs[k], test_accs[k], fps[k], fns[k], n_nodes[k], -1, -1] for k in range(N_FOLDS)]
            to_write += [val for sublist in tmp_to_write for val in sublist]

            print("Random Forest")
            print("Dataset :", filename)
            print("Running cross validation")
            train_accs, test_accs, n_trees, fps, fns, fit_times, n_nodes = [], [], [], [], [], [], []
            for k in range(N_FOLDS):
                X_train, X_test, y_train, y_test = X_trains[k], X_tests[k], y_trains[k], y_tests[k]
                clf = RandomForestClassifier(max_depth=MAX_DEPTH, n_estimators=max_estimators[k])
                start = time.perf_counter()
                clf.fit(X_train, y_train)
                fit_times.append(time.perf_counter() - start)
                y_pred = clf.predict(X_test)
                n_trees.append(len(clf.estimators_))
                train_accs.append(accuracy_score(y_train, clf.predict(X_train)))
                test_accs.append(accuracy_score(y_test, y_pred))
                n_nodes.append(sum([c.tree_.node_count for c in clf.estimators_]))
                fps.append(len([i for i in [j for j, val in enumerate(y_pred) if val == 1] if y_test[i] != 1]))
                fns.append(len([i for i in [j for j, val in enumerate(y_pred) if val == 0] if y_test[i] != 0]))
            print("Model built. Avg duration of building =", round(float(np.mean(fit_times)), 4))
            print("Number of trees =", n_trees, np.mean(n_trees))
            print("Accuracy on training set =", train_accs, round(float(np.mean(train_accs)), 4))
            print("Accuracy on test set =", test_accs, round(float(np.mean(test_accs)), 4))
            print("list of n_nodes =", n_nodes)
            print("sum false positives =", sum(fps))
            print("sum false negatives =", sum(fns), "\n\n\n")
            tmp_to_write = [[n_trees[k], n_trees[k], fit_times[k], True, train_accs[k], test_accs[k], fps[k], fns[k], n_nodes[k], -1, -1] for k in range(N_FOLDS)]
            to_write += [val for sublist in tmp_to_write for val in sublist]


        # print("LPBoost + CART")
        # print("Dataset :", filename)
        # n_trees, fps, fns, fit_times, train_scores, test_scores, n_iter, regulators, gammas, n_opti = [], [], [], [], [], [], [], [], [], 0
        # for k in range(N_FOLDS):
        #     X_train, X_test, y_train, y_test = X_trains[k], X_tests[k], y_trains[k], y_tests[k]
        #     print("Fold", k+1, "- Search for the best regulator using grid search...", MAX_TREES)
        #     gd_sr = GridSearchCV(estimator=DL85Boostera(base_estimator=DecisionTreeClassifier(max_depth=MAX_DEPTH), model=MODEL, max_iterations=max_estimators[k]),
        #                          param_grid=parameters, scoring='accuracy', cv=N_FOLDS_TUNING, n_jobs=-1, verbose=VERBOSE_LEVEL)
        #     gd_sr.fit(X_train, y_train)
        #     print()
        #     print("Fold", k+1, "- Running training for LPBoost + CART with best regulator =", gd_sr.best_params_["regulator"], "on", filename)
        #     clf = gd_sr.best_estimator_
        #     y_pred = clf.predict(X_test)
        #     n_trees.append(clf.n_estimators_)
        #     fit_times.append(clf.duration_)
        #     train_scores.append(accuracy_score(y_train, clf.predict(X_train)))
        #     test_scores.append(accuracy_score(y_test, y_pred))
        #     n_iter.append(clf.n_iterations_)
        #     regulators.append(gd_sr.best_params_["regulator"])
        #     gammas.append(gd_sr.best_params_["gamma"])
        #     n_opti = n_opti + 1 if clf.optimal_ else n_opti
        #     fps.append(len([i for i in [j for j, val in enumerate(y_pred) if val == 1] if y_test[i] != 1]))
        #     fns.append(len([i for i in [j for j, val in enumerate(y_pred) if val == 0] if y_test[i] != 0]))
        #     to_write += [n_iter[-1], n_trees[-1], fit_times[-1], clf.optimal_, train_scores[-1], test_scores[-1], fps[-1], fns[-1], regulators[-1], gammas[-1]]
        #     print("fold :", k+1, "n_trees :", n_trees[k], "train_acc :", train_scores[k], "test acc :", test_scores[k], "regu :", regulators[k], "gamma :", gammas[k], "\n")
        # print("Model built. Avg duration of building =", round(float(np.mean(fit_times)), 4))
        # print("Number of trees =", n_trees)
        # print("Accuracy on training set =", train_scores, round(float(np.mean(train_scores)), 4))
        # print("Accuracy on test set =", test_scores, round(float(np.mean(test_scores)), 4))
        # print("number of optimality :", n_opti)
        # print("list of iterations :", n_iter)
        # print("list of time :", fit_times)
        # print("list of regulator :", regulators)
        # print("list of gammas :", gammas)
        # print("sum false positives =", sum(fps))
        # print("sum false negatives =", sum(fns), "\n\n\n")

        # max_estimators = [60, 83, 81, 82, 80]
        # print("Adaboost + DL8.5")
        # print("Dataset :", filename)
        # print("Running cross validation")
        # train_accs, test_accs, n_trees, fps, fns, fit_times, n_nodes = [], [], [], [], [], [], []
        # for k in range(N_FOLDS):
        #     X_train, X_test, y_train, y_test = X_trains[k], X_tests[k], y_trains[k], y_tests[k]
        #     clf = AdaBoostClassifier(base_estimator=DL85Classifier(max_depth=MAX_DEPTH), n_estimators=max_estimators[k])
        #     start = time.perf_counter()
        #     clf.fit(X_train, y_train)
        #     fit_times.append(time.perf_counter() - start)
        #     y_pred = clf.predict(X_test)
        #     n_trees.append(len(clf.estimators_))
        #     train_accs.append(accuracy_score(y_train, clf.predict(X_train)))
        #     test_accs.append(accuracy_score(y_test, y_pred))
        #     n_nodes.append(sum([c.get_nodes_count() for c in clf.estimators_]))
        #     fps.append(len([i for i in [j for j, val in enumerate(y_pred) if val == 1] if y_test[i] != 1]))
        #     fns.append(len([i for i in [j for j, val in enumerate(y_pred) if val == 0] if y_test[i] != 0]))
        # print("Model built. Avg duration of building =", round(float(np.mean(fit_times)), 4))
        # print("Number of trees =", n_trees, np.mean(n_trees))
        # print("Accuracy on training set =", train_accs, round(float(np.mean(train_accs)), 4))
        # print("Accuracy on test set =", test_accs, round(float(np.mean(test_accs)), 4))
        # print("list of n_nodes =", n_nodes)
        # print("sum false positives =", sum(fps))
        # print("sum false negatives =", sum(fns), "\n\n\n")
        # tmp_to_write = [[n_trees[k], n_trees[k], fit_times[k], True, train_accs[k], test_accs[k], fps[k], fns[k], n_nodes[k], -1, -1] for k in range(N_FOLDS)]
        # to_write += [val for sublist in tmp_to_write for val in sublist]

        # print("Adaboost + CART")
        # print("Dataset :", filename)
        # print("Running cross validation")
        # train_accs, test_accs, n_trees, fps, fns, fit_times, n_nodes = [], [], [], [], [], [], []
        # for k in range(N_FOLDS):
        #     X_train, X_test, y_train, y_test = X_trains[k], X_tests[k], y_trains[k], y_tests[k]
        #     clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=MAX_DEPTH), n_estimators=max_estimators[k])
        #     start = time.perf_counter()
        #     clf.fit(X_train, y_train)
        #     fit_times.append(time.perf_counter() - start)
        #     y_pred = clf.predict(X_test)
        #     n_trees.append(len(clf.estimators_))
        #     train_accs.append(accuracy_score(y_train, clf.predict(X_train)))
        #     test_accs.append(accuracy_score(y_test, y_pred))
        #     n_nodes.append(sum([c.tree_.node_count for c in clf.estimators_]))
        #     fps.append(len([i for i in [j for j, val in enumerate(y_pred) if val == 1] if y_test[i] != 1]))
        #     fns.append(len([i for i in [j for j, val in enumerate(y_pred) if val == 0] if y_test[i] != 0]))
        # print("Model built. Avg duration of building =", round(float(np.mean(fit_times)), 4))
        # print("Number of trees =", n_trees, np.mean(n_trees))
        # print("Accuracy on training set =", train_accs, round(float(np.mean(train_accs)), 4))
        # print("Accuracy on test set =", test_accs, round(float(np.mean(test_accs)), 4))
        # print("list of n_nodes =", n_nodes)
        # print("sum false positives =", sum(fps))
        # print("sum false negatives =", sum(fns), "\n\n\n")
        # tmp_to_write = [[n_trees[k], n_trees[k], fit_times[k], True, train_accs[k], test_accs[k], fps[k], fns[k], n_nodes[k], -1, -1] for k in range(N_FOLDS)]
        # to_write += [val for sublist in tmp_to_write for val in sublist]

        # for mm in [10, 20, 30, 40, 50]:
        #     print("Adaboost + CART " + str(mm))
        # print("Dataset :", filename)
        # print("Running cross validation")
        # train_accs, test_accs, n_trees, fps, fns, fit_times, n_nodes = [], [], [], [], [], [], []
        # for k in range(N_FOLDS):
        #     X_train, X_test, y_train, y_test = X_trains[k], X_tests[k], y_trains[k], y_tests[k]
        #     clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=MAX_DEPTH), n_estimators=mm)
        #     start = time.perf_counter()
        #     clf.fit(X_train, y_train)
        #     fit_times.append(time.perf_counter() - start)
        #     y_pred = clf.predict(X_test)
        #     n_trees.append(len(clf.estimators_))
        #     train_accs.append(accuracy_score(y_train, clf.predict(X_train)))
        #     test_accs.append(accuracy_score(y_test, y_pred))
        #     n_nodes.append(sum([c.tree_.node_count for c in clf.estimators_]))
        #     fps.append(len([i for i in [j for j, val in enumerate(y_pred) if val == 1] if y_test[i] != 1]))
        #     fns.append(len([i for i in [j for j, val in enumerate(y_pred) if val == 0] if y_test[i] != 0]))
        # print("Model built. Avg duration of building =", round(float(np.mean(fit_times)), 4))
        # print("Number of trees =", n_trees, np.mean(n_trees))
        # print("Accuracy on training set =", train_accs, round(float(np.mean(train_accs)), 4))
        # print("Accuracy on test set =", test_accs, round(float(np.mean(test_accs)), 4))
        # print("list of n_nodes =", n_nodes)
        # print("sum false positives =", sum(fps))
        # print("sum false negatives =", sum(fns), "\n\n\n")
        # tmp_to_write = [[n_trees[k], n_trees[k], fit_times[k], True, train_accs[k], test_accs[k], fps[k], fns[k], n_nodes[k], -1, -1] for k in range(N_FOLDS)]
        # to_write += [val for sublist in tmp_to_write for val in sublist]

        # print("Adaboost + CART 100")
        # print("Dataset :", filename)
        # print("Running cross validation")
        # train_accs, test_accs, n_trees, fps, fns, fit_times, n_nodes = [], [], [], [], [], [], []
        # for k in range(N_FOLDS):
        #     X_train, X_test, y_train, y_test = X_trains[k], X_tests[k], y_trains[k], y_tests[k]
        #     clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=MAX_DEPTH), n_estimators=100)
        #     start = time.perf_counter()
        #     clf.fit(X_train, y_train)
        #     fit_times.append(time.perf_counter() - start)
        #     y_pred = clf.predict(X_test)
        #     n_trees.append(len(clf.estimators_))
        #     train_accs.append(accuracy_score(y_train, clf.predict(X_train)))
        #     test_accs.append(accuracy_score(y_test, y_pred))
        #     n_nodes.append(sum([c.tree_.node_count for c in clf.estimators_]))
        #     fps.append(len([i for i in [j for j, val in enumerate(y_pred) if val == 1] if y_test[i] != 1]))
        #     fns.append(len([i for i in [j for j, val in enumerate(y_pred) if val == 0] if y_test[i] != 0]))
        # print("Model built. Avg duration of building =", round(float(np.mean(fit_times)), 4))
        # print("Number of trees =", n_trees, np.mean(n_trees))
        # print("Accuracy on training set =", train_accs, round(float(np.mean(train_accs)), 4))
        # print("Accuracy on test set =", test_accs, round(float(np.mean(test_accs)), 4))
        # print("list of n_nodes =", n_nodes)
        # print("sum false positives =", sum(fps))
        # print("sum false negatives =", sum(fns), "\n\n\n")
        # tmp_to_write = [[n_trees[k], n_trees[k], fit_times[k], True, train_accs[k], test_accs[k], fps[k], fns[k], n_nodes[k], -1, -1] for k in range(N_FOLDS)]
        # to_write += [val for sublist in tmp_to_write for val in sublist]

        # print("Gradient Boosting")
        # print("Dataset :", filename)
        # print("Running cross validation")
        # train_accs, test_accs, n_trees, fps, fns, fit_times, n_nodes = [], [], [], [], [], [], []
        # for k in range(N_FOLDS):
        #     X_train, X_test, y_train, y_test = X_trains[k], X_tests[k], y_trains[k], y_tests[k]
        #     clf = GradientBoostingClassifier(max_depth=MAX_DEPTH, n_estimators=max_estimators[k])
        #     start = time.perf_counter()
        #     try:
        #         clf.fit(X_train, y_train)
        #         t = time.perf_counter() - start
        #         y_pred = clf.predict(X_test)
        #         fit_times.append(t)
        #         n_trees.append(clf.n_estimators_)
        #         train_accs.append(accuracy_score(y_train, clf.predict(X_train)))
        #         test_accs.append(accuracy_score(y_test, y_pred))
        #         n_nodes.append(sum([c[0].tree_.node_count for c in clf.estimators_]))
        #         fps.append(len([i for i in [j for j, val in enumerate(y_pred) if val == 1] if y_test[i] != 1]))
        #         fns.append(len([i for i in [j for j, val in enumerate(y_pred) if val == 0] if y_test[i] != 0]))
        #     except:
        #         fit_times.append(0)
        #         n_trees.append(0)
        #         train_accs.append(0)
        #         test_accs.append(0)
        #         n_nodes.append(0)
        #         fps.append(0)
        #         fns.append(0)
        # print("Model built. Avg duration of building =", round(float(np.mean(fit_times)), 4))
        # print("Number of trees =", n_trees, np.mean(n_trees))
        # print("Accuracy on training set =", train_accs, round(float(np.mean(train_accs)), 4))
        # print("Accuracy on test set =", test_accs, round(float(np.mean(test_accs)), 4))
        # print("list of n_nodes =", n_nodes)
        # print("sum false positives =", sum(fps))
        # print("sum false negatives =", sum(fns), "\n\n\n")
        # tmp_to_write = [[n_trees[k], n_trees[k], fit_times[k], True, train_accs[k], test_accs[k], fps[k], fns[k], n_nodes[k], -1, -1] for k in range(N_FOLDS)]
        # to_write += [val for sublist in tmp_to_write for val in sublist]

        # for mm in [10, 20, 30, 40, 50, 100]:
        #     print("Gradient Boosting " + str(mm))
        #     print("Dataset :", filename)
        #     print("Running cross validation")
        #     train_accs, test_accs, n_trees, fps, fns, fit_times, n_nodes = [], [], [], [], [], [], []
        #     for k in range(N_FOLDS):
        #         X_train, X_test, y_train, y_test = X_trains[k], X_tests[k], y_trains[k], y_tests[k]
        #         clf = GradientBoostingClassifier(max_depth=MAX_DEPTH, n_estimators=mm)
        #         start = time.perf_counter()
        #         try:
        #             clf.fit(X_train, y_train)
        #             t = time.perf_counter() - start
        #             y_pred = clf.predict(X_test)
        #             fit_times.append(t)
        #             n_trees.append(clf.n_estimators_)
        #             train_accs.append(accuracy_score(y_train, clf.predict(X_train)))
        #             test_accs.append(accuracy_score(y_test, y_pred))
        #             n_nodes.append(sum([c[0].tree_.node_count for c in clf.estimators_]))
        #             fps.append(len([i for i in [j for j, val in enumerate(y_pred) if val == 1] if y_test[i] != 1]))
        #             fns.append(len([i for i in [j for j, val in enumerate(y_pred) if val == 0] if y_test[i] != 0]))
        #         except:
        #             fit_times.append(0)
        #             n_trees.append(0)
        #             train_accs.append(0)
        #             test_accs.append(0)
        #             n_nodes.append(0)
        #             fps.append(0)
        #             fns.append(0)
        #     print("Model built. Avg duration of building =", round(float(np.mean(fit_times)), 4))
        #     print("Number of trees =", n_trees, np.mean(n_trees))
        #     print("Accuracy on training set =", train_accs, round(float(np.mean(train_accs)), 4))
        #     print("Accuracy on test set =", test_accs, round(float(np.mean(test_accs)), 4))
        #     print("list of n_nodes =", n_nodes)
        #     print("sum false positives =", sum(fps))
        #     print("sum false negatives =", sum(fns), "\n\n\n")
        #     tmp_to_write = [[n_trees[k], n_trees[k], fit_times[k], True, train_accs[k], test_accs[k], fps[k], fns[k], n_nodes[k], -1, -1] for k in range(N_FOLDS)]
        #     to_write += [val for sublist in tmp_to_write for val in sublist]

        # print("Random Forest")
        # print("Dataset :", filename)
        # print("Running cross validation")
        # train_accs, test_accs, n_trees, fps, fns, fit_times, n_nodes = [], [], [], [], [], [], []
        # for k in range(N_FOLDS):
        #     X_train, X_test, y_train, y_test = X_trains[k], X_tests[k], y_trains[k], y_tests[k]
        #     clf = RandomForestClassifier(max_depth=MAX_DEPTH, n_estimators=max_estimators[k])
        #     start = time.perf_counter()
        #     clf.fit(X_train, y_train)
        #     fit_times.append(time.perf_counter() - start)
        #     y_pred = clf.predict(X_test)
        #     n_trees.append(len(clf.estimators_))
        #     train_accs.append(accuracy_score(y_train, clf.predict(X_train)))
        #     test_accs.append(accuracy_score(y_test, y_pred))
        #     n_nodes.append(sum([c.tree_.node_count for c in clf.estimators_]))
        #     fps.append(len([i for i in [j for j, val in enumerate(y_pred) if val == 1] if y_test[i] != 1]))
        #     fns.append(len([i for i in [j for j, val in enumerate(y_pred) if val == 0] if y_test[i] != 0]))
        # print("Model built. Avg duration of building =", round(float(np.mean(fit_times)), 4))
        # print("Number of trees =", n_trees, np.mean(n_trees))
        # print("Accuracy on training set =", train_accs, round(float(np.mean(train_accs)), 4))
        # print("Accuracy on test set =", test_accs, round(float(np.mean(test_accs)), 4))
        # print("list of n_nodes =", n_nodes)
        # print("sum false positives =", sum(fps))
        # print("sum false negatives =", sum(fns), "\n\n\n")
        # tmp_to_write = [[n_trees[k], n_trees[k], fit_times[k], True, train_accs[k], test_accs[k], fps[k], fns[k], n_nodes[k], -1, -1] for k in range(N_FOLDS)]
        # to_write += [val for sublist in tmp_to_write for val in sublist]

        file_out.write(";".join(map(lambda x: str(x), to_write)) + "\n")
        file_out.flush()
        print(to_write)
file_out.close()
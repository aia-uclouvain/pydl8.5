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
from dl85 import DL85Booster, DL85Classifier, MODEL_RATSCH, MODEL_DEMIRIZ, DL85Boostera, MODEL_AGLIN
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.exceptions import FitFailedWarning
import os
import sys

# filename = "letter"
# dataset = np.genfromtxt("../datasets/" + filename + ".txt", delimiter=' ')

N_FOLDS, N_FOLDS_TUNING, MAX_DEPTH, MIN_SUP, MODEL = 5, 4, int(sys.argv[1]) if len(sys.argv) > 1 else 2, 1, 'gurobi' if len(sys.argv) > 4 else 'cvxpy'
#N_FOLDS, N_FOLDS_TUNING, MAX_DEPTH, MIN_SUP, MODEL = 5, 4, int(sys.argv[1]) if len(sys.argv) > 1 else 1, 1, MODEL_AGLIN if len(sys.argv) > 4 else MODEL_AGLIN
MAX_ITERATIONS, MAX_TREES, TIME_LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 0, 0, 0
VERBOSE_LEVEL = 10
first_file = sys.argv[3] + '.txt' if len(sys.argv) > 3 else 'zoo-1.txt'
print("model", MODEL)
# depth max_iter first_file model

file_out = open("../output/out_md_ada_grid_" + ((str(MAX_ITERATIONS) + '_') if MAX_ITERATIONS > 0 else '') + "depth_" + str(MAX_DEPTH) + "_none.csv", "a+")
file_out.write(",".join(["dataset", "n_trees", "train", "test", "n_nodes", "f_ntree", "f_learn_r"]) + "\n")
#directory = '../datasets'
#directory = '../datasets/boosting/mm'
files = ['zoo-1.txt', 'hepatitis.txt', 'lymph.txt', 'audiology.txt', 'heart-cleveland.txt', 'primary-tumor.txt', 'tic-tac-toe.txt', 'vote.txt', 'soybean.txt',
         'anneal.txt', 'yeast.txt', 'australian-credit.txt', 'breast-wisconsin.txt', 'diabetes.txt', 'german-credit.txt', 'kr-vs-kp.txt', 'hypothyroid.txt',
         'mushroom.txt', 'vehicle.txt', 'ionosphere.txt', 'segment.txt', 'splice-1.txt', 'pendigits.txt', 'letter.txt']
# for filename in sorted(os.listdir(directory)):

for filename in sorted(files[files.index(first_file):]):
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

        # to_write = [filename.split(".")[0], X.shape[1], X.shape[0], y.tolist().count(0), y.tolist().count(1)]
        to_write = [filename.split(".")[0]]

        # parameters = {'regulator': [2, 5, 8, 10, 12, 15, 20, 30, 40, 50, 70, 90, 100, 120]}
        # parameters = {'regulator': [2, 5, 8, 10, 12, 15, 20, 30, 40, 50, 70, 90, 100, 120], 'gamma': [None, 'auto', 'scale', 'nscale', 1, 0.1, 0.01, 0.001, 0.0001]}
        # parameters = {'regulator': list(map(lambda x: pow(2, x), list(range(-5, 16)))), 'gamma': [None] + list(map(lambda x: pow(2, x), list(range(-15, 4))))}
        # parameters = {'regulator': [0.01, 0.1, 1, 2, 5, 8, 10, 20, 40, 70, 100, 150], 'gamma': [None, 'auto', 'scale', 'nscale']}
        #parameters = {'regulator': [0.01, 0.1, 1, 2, 5, 8, 10, 15, 20, 30, 40, 50, 70, 90, 100, 150]}
        #parameters = {'regulator': [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 10, 100, 1000, 10000, 100000]}
        #parameters = {'regulator': [0.01, 0.1, 0.3, 0.6, 0.8, 1, 5, 7, 10, 20, 30, 50, 80, 100, 150, 1000, 10000, 100000]}
        parameters = {'regulator': [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 15, 30, 50, 100, 150]}
        #parameters = {'regulator': [0.01, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.85, 0.9, 1]}
        parameters_ada = {'n_estimators': [10, 50, 100, 200, 500, 800, 1000], 'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0]}
        parameters_ada_1 = {'n_estimators': [10, 50, 100, 200, 500, 800, 1000], 'learning_rate': [1.0]}

        print("######################################################################\n"
              "#                                START                               #\n"
              "######################################################################")

        print("Adaboost grid search")
        print("Dataset :", filename)
        n_trees, fps, fns, fit_times, train_scores, test_scores, n_iter, regulators, gammas, n_opti, n_nodes = [], [], [], [], [], [], [], [], [], 0, []
        for k in range(N_FOLDS):
            X_train, X_test, y_train, y_test = X_trains[k], X_tests[k], y_trains[k], y_tests[k]

            train_indices, valid_indices = [], []
            kf = StratifiedKFold(n_splits=N_FOLDS_TUNING)
            for train_index, test_index in kf.split(X_train, y_train):
                if X.shape[0] <= 1000:  # 3/4 tr - 1/4 te
                    train_indices.append(train_index)
                    valid_indices.append(test_index)
                else:  # 1/4 tr - 3/4 te
                    train_indices.append(test_index)
                    valid_indices.append(train_index)
            custom_cv = zip(train_indices, valid_indices)

            print("Fold", k+1, "- Search for the best regulator using grid search...", MAX_TREES)
            gd_sr = GridSearchCV(estimator=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=MAX_DEPTH, criterion='entropy')), error_score=np.nan,
                                 param_grid=parameters_ada_1, scoring='accuracy', cv=custom_cv, n_jobs=-1, verbose=0)
            gd_sr.fit(X_train, y_train)
            print()
            print("Fold", k+1, "- Running cross validation for LPBoost + DL8.5 with best regulator =", -1, "on", filename)
            print("best estimator", gd_sr.best_estimator_)
            clf = gd_sr.best_estimator_
            y_pred = clf.predict(X_test)
            n_trees.append(len(clf.estimators_))
            fit_times.append(gd_sr.refit_time_)
            train_scores.append(round(accuracy_score(y_train, clf.predict(X_train)) * 100, 2))
            test_scores.append(round(accuracy_score(y_test, y_pred) * 100, 2))
            n_iter.append(len(clf.estimators_))
            regulators.append(gd_sr.best_params_["n_estimators"])
            #gammas.append(gd_sr.best_params_["gamma"])
            gammas.append(gd_sr.best_params_["learning_rate"])
            n_nodes.append(sum([c.tree_.node_count for c in clf.estimators_]))
            n_opti = n_opti + 1
            fps.append(len([i for i in [j for j, val in enumerate(y_pred) if val == 1] if y_test[i] != 1]))
            fns.append(len([i for i in [j for j, val in enumerate(y_pred) if val == 0] if y_test[i] != 0]))
            # to_write += [n_iter[-1], n_trees[-1], fit_times[-1], True, train_scores[-1], test_scores[-1], fps[-1], fns[-1], n_nodes[-1], regulators[-1], gammas[-1]]
            print("fold :", k+1, "n_trees :", n_trees[k], "train_acc :", train_scores[k], "test acc :", test_scores[k], "n_nodes :", n_nodes[k], "regu :", regulators[k], "gamma :", gammas[k], "\n")
        # max_estimators = int(sum(n_trees)/len(n_trees))
        # max_estimators = n_trees[:]
        to_write += [np.mean(n_trees), round(float(np.mean(train_scores)), 2), round(float(np.mean(test_scores)), 2), sum(n_nodes), regulators[0], gammas[0]]
        print("Model built. Avg duration of building =", round(float(np.mean(fit_times)), 4))
        print("Number of trees =", n_trees, np.mean(n_trees))
        print("Accuracy on training set =", train_scores, round(float(np.mean(train_scores)), 4))
        print("Accuracy on test set =", test_scores, round(float(np.mean(test_scores)), 4))
        print("number of optimality :", n_opti)
        print("list of iterations :", n_iter)
        print("list of time :", fit_times)
        print("list of regulator :", regulators)
        print("list of gammas :", gammas)
        print("list of n_nodes :", n_nodes)
        print("sum false positives =", sum(fps))
        print("sum false negatives =", sum(fns), "\n\n\n")

        file_out.write(",".join(map(lambda x: str(x), to_write)) + "\n")
        file_out.flush()
        print(to_write)
file_out.close()

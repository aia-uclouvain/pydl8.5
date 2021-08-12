import os
import warnings
from sklearn.metrics import confusion_matrix
from sklearn.exceptions import FitFailedWarning
from sklearn.model_selection import train_test_split
import sys
import time
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score, cross_validate
from dl85 import DL85Booster, MODEL_LP_RATSCH, MODEL_LP_DEMIRIZ, MODEL_QP_MDBOOST, DL85Classifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
import csv

# files = ['zoo-1.txt',
#          'hepatitis.txt',
#          'lymph.txt',
#          'audiology.txt',
#          'heart-cleveland.txt',
#          'primary-tumor.txt',
#          'tic-tac-toe.txt',
#          'vote.txt',
#          'soybean.txt',
#          'anneal.txt',
#          'yeast.txt',
#          'australian-credit.txt',
#          'breast-wisconsin.txt',
#          'diabetes.txt',
#          'german-credit.txt',
#          'kr-vs-kp.txt',
#          'hypothyroid.txt',
#          'mushroom.txt',
#          'vehicle.txt',
#          'ionosphere.txt',
#          'segment.txt',
#          'splice-1.txt',
#          'pendigits.txt',
#          'letter.txt']

files = ['diabetes.txt', 'australian-credit.txt', 'breast-wisconsin.txt', "splice-1.txt", "vehicle.txt"]

# files = ['yeast.txt',
#          'diabetes.txt',
#          'german-credit.txt',
#          'mushroom.txt',
#          'ionosphere.txt',
#          'letter.txt']

FIRST_FILE = sys.argv[3] + '.txt' if len(sys.argv) > 3 else 'diabetes.txt'
MAX_ITERATIONS = int(sys.argv[2]) if len(sys.argv) > 2 else 0
MAX_DEPTH = int(sys.argv[1]) if len(sys.argv) > 1 else 3
MAX_FOLD_SIZE = 1000  # max size of each fold used for cross validation. size for tuning is timed by N_FOLDS_TUNING / N_FOLDS
N_FOLDS_TUNING = 4  # number of folds to create per fold for hyperparameter tuning
VERBOSE_LEVEL = 10
TIME_LIMIT = 0
N_FOLDS = 5  # number of folds for cross validation
MIN_SUP = 1
START = 0
END = len(files)
# START = files.index(FIRST_FILE)
# END = files.index(FIRST_FILE) + 1
# python3 file.py max_depth max_iterations first_file

# parameters = {'regulator': [.5, 1, 3, 6, 12, 25, 50, 100]}
parameters = {'regulator': [.125, .25, .375, .5, .625, .75, .875, 1]}
# parameters = {'regulator': [1, 5, 10, 15, 30, 50, 100, 150]}
# parameters = {'regulator': [0.01, 0.1, 0.5, 1, 5, 8, 10, 12, 13, 15, 17, 18, 20, 30, 50, 100]}
parameters_md = {'regulator': [1, 5, 10, 15, 30, 50, 100, 150]}
parameters_ada = {'n_estimators': [5, 10, 20, 40, 80, 100, 200, 300]}
# parameters_ada = {'n_estimators': [10, 50, 100, 200, 300, 400, 500]}

model_names = ["LP_RATSCH", "MDBOOST"]
# model_names = ["LP_DEMIRIZ"]
# model_names = ["LP_RATSCH"]
# model_names = ["MDBOOST"]
models = [MODEL_LP_RATSCH, MODEL_QP_MDBOOST]
# models = [MODEL_LP_DEMIRIZ]
# models = [MODEL_LP_RATSCH]
# models = [MODEL_QP_MDBOOST]
tree_names = ["DL85", "CART"]
# tree_names = ["DL85"]
tree_algos = [None, DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=42)]
# tree_algos = [None]
with_ada = True
# with_ada = False

directory = '../datasets'
# file_out = open("output/out_auc_opti_maxiter_" + (str(MAX_ITERATIONS) if MAX_ITERATIONS > 0 else 'none') + "_depth_" + str(MAX_DEPTH) + ".csv", "a+")
file_out = open("output/out_auc_opti_depth_" + str(MAX_DEPTH) + ".csv", "w")
file_iters = open("output/out_iters_depth_" + str(MAX_DEPTH) + ".csv", "a")
stats_writer = csv.writer(file_out, delimiter=',')
iters_writer = csv.writer(file_iters, delimiter=',')
iters_writer.writerow(["dataset", "fold", "reg", "fixed_reg", "algo", "objective", "train_acc", "test_acc", "train_auc", "test_auc", "n_iter", "n_trees", "min_margin", "avg_margin", "var_margin"])

for filename in files[START:END]:
    dataset = np.genfromtxt("../datasets/" + filename, delimiter=' ')
    filename_ = filename.split(".")[0]
    X, y = dataset[:, 1:], dataset[:, 0]
    X_trains, X_tests, y_trains, y_tests = [], [], [], []
    train_indices_list, test_indices_list = [], []
    max_data_size_for_cv = MAX_FOLD_SIZE * (N_FOLDS / (N_FOLDS - 1))
    # if we do not overflow the max size of the data to respect the max size per fold after splitting, we split in folds. Otherwise we pick the max data in stratified way
    kf = StratifiedKFold(n_splits=N_FOLDS) if X.shape[0] <= max_data_size_for_cv else StratifiedShuffleSplit(n_splits=N_FOLDS, train_size=MAX_FOLD_SIZE, random_state=0)

    for train_indices, test_indices in kf.split(X, y):
        # set training data and keep their index in initial data
        train_indices_list.append(train_indices)
        X_trains.append(X[train_indices])
        y_trains.append(y[train_indices])
        # set test data and keep their index in initial data
        test_indices_list.append(test_indices)
        X_tests.append(X[test_indices])
        y_tests.append(y[test_indices])

    # prepare the folds for grid search CV
    custom_cv_dl85 = zip(train_indices_list, test_indices_list)
    custom_cv_cart = zip(train_indices_list, test_indices_list)
    custom_cv_ada = zip(train_indices_list, test_indices_list)

    # describe the data
    print("Dataset :", filename)
    print("size of 0 :", y.tolist().count(0), "size of 1 :", y.tolist().count(1))
    print("n_feat = ", X.shape[1], "n_trans = ", X.shape[0])

    # write the description part of the data in the results
    to_write = [filename.split(".")[0], X.shape[1], X.shape[0], y.tolist().count(0), y.tolist().count(1)]

    print("######################################################################\n"
          "#                                START                               #\n"
          "######################################################################")

    # ========================= #
    #        Optiboost          #
    # ========================= #
    def get_ntrees(estimator, X, y):
        return estimator.n_estimators_

    def get_niterations(estimator, X, y):
        return estimator.n_iterations_

    def get_objective(estimator, X, y):
        return estimator.objective_

    def replacenan(l):
        for i in range(len(l)):
            if np.isnan(l[i]):
                l[i] = 0
        return list(map(lambda x: str(x), l))

    info_per_reg = {"dem": {"dl85": {}, "cart": {}}, "rat": {"dl85": {}, "cart": {}}, "md": {"dl85": {}, "cart": {}}}
    all_best_objs_per_fold = {}
    for name, base_clf in zip([x + " + " + y for x in model_names for y in tree_names], [DL85Booster(base_estimator=base, max_depth=MAX_DEPTH, model=mod, max_iterations=MAX_ITERATIONS) for mod in models for base in tree_algos]):
        print("Optiboost + {}".format(name))
        print("Dataset :", filename)
        file = open("output/" + name.replace(" + ", "_") + "_auc_depth_" + str(MAX_DEPTH) + ".csv", "a")
        n_trees, fps, fns, fit_times, train_scores, test_scores, n_iter, regulators, gammas, n_opti, n_nodes, train_aucs, test_aucs, objs = [], [], [], [], [], [], [], [], [], 0, [], [], [], []
        # build training set and validation set for the hyperparameter tuning. Use 4 folds for this task
        for k in range(N_FOLDS):
            X_train, X_test, y_train, y_test = X_trains[k], X_tests[k], y_trains[k], y_tests[k]
            print("Fold", k+1, "- Search for the best regulator using grid search...")
            param = parameters_md if "MDBOOST" in name else parameters
            # param = parameters
            gd_sr = GridSearchCV(estimator=base_clf, error_score=np.nan, param_grid=param, scoring={'ntrees': get_ntrees, 'niter': get_niterations, 'obj': get_objective, 'auc': 'roc_auc', 'acc': 'accuracy'}, refit=False, cv=N_FOLDS_TUNING, n_jobs=-1, verbose=VERBOSE_LEVEL)
            gd_sr.fit(X_train, y_train)

            rank_sorted_indices = np.argsort(gd_sr.cv_results_['rank_test_auc'])
            best_param_index = rank_sorted_indices[0]
            best_reg = gd_sr.cv_results_['params'][best_param_index]['regulator']

            mod = MODEL_LP_DEMIRIZ if "LP_DEMIRIZ" in name else MODEL_LP_RATSCH if "LP_RATSCH" in name else MODEL_QP_MDBOOST
            estim = None if "DL85" in name else DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=42)
            clf = DL85Booster(model=mod, base_estimator=estim, max_depth=MAX_DEPTH, max_iterations=MAX_ITERATIONS, regulator=best_reg)
            suffix = "lpdem_cart" if name == "LP_DEMIRIZ + CART" else "lpdem_dl85" if name == "LP_DEMIRIZ + DL85" else "lprat_cart" if name == "LP_RATSCH + CART" else "lprat_dl85" if name == "LP_RATSCH + DL85" else "mdboost_cart" if name == "MDBOOST + CART" else "mdboost_dl85" if name == "MDBOOST + DL85" else "other"
            iter_filepath = "output/{}_depth_{}_fold_{}_{}".format(filename.split(".")[0], MAX_DEPTH, k+1, suffix)
            clf.fit(X_train, y_train, X_test, y_test, iter_file=iter_filepath)

            with open(iter_filepath + ".csv") as f:
                content = f.readlines()
            os.remove(iter_filepath + ".csv")

            def add_info(line: str):
                my_list = line.strip().split(',')
                return [filename_, str(k+1), str(best_reg), "0", suffix] + my_list

            content.pop(0)
            content = [add_info(x) for x in content]
            iters_writer.writerows(content)
            file_iters.flush()

            print("Fold", k+1, "- End {} with best regulator = {} on {}".format(name, best_reg, filename))
            print("best estimator", clf)
            y_pred = clf.predict(X_test)
            n_trees.append(clf.n_estimators_)
            fit_times.append(clf.duration_)
            train_scores.append(accuracy_score(y_train, clf.predict(X_train)))
            test_scores.append(accuracy_score(y_test, y_pred))
            n_iter.append(clf.n_iterations_)
            objs.append(clf.objective_)
            regulators.append(best_reg)
            test_aucs.append(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
            train_aucs.append(roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1]))
            gammas.append(-1)
            if "CART" in name:
                n_nodes.append(sum([c.tree_.node_count for c in clf.estimators_]))
            else:  # dl85booster with dl85
                n_nodes.append(clf.get_nodes_count())
            n_opti = n_opti + 1 if clf.optimal_ else n_opti
            fps.append(len([i for i in [j for j, val in enumerate(y_pred) if val == 1] if y_test[i] != 1]))
            fns.append(len([i for i in [j for j, val in enumerate(y_pred) if val == 0] if y_test[i] != 0]))
            to_write += [n_iter[-1], n_trees[-1], fit_times[-1], clf.optimal_, train_scores[-1], test_scores[-1], fps[-1], fns[-1], n_nodes[-1], regulators[-1], gammas[-1], train_aucs[-1], test_aucs[-1], objs[-1]]
            print("Optiboost + {}".format(name), end=" ")
            print("fold :", k+1, "n_trees :", n_trees[k], "train_acc :", train_scores[k], "test acc :", test_scores[k], "n_nodes :", n_nodes[k], "regu :", regulators[k], "gamma :", gammas[k], "train_auc :", train_aucs[k], "test_auc :", test_aucs[k], "obj :", objs[k], "\n")


            # this part is about saving some metrics about the different parameters tested
            # each successive 5 lines contain for each file, the 5 metrics below gathered
            # for each fold. So, for 5 folds, each file needs 25 lines. Each line is
            # composed of the value gathered for each hyperparameter. So each line contains
            # the same value as the number of hyperparameters. In this case, there are 16 for
            # lpboost and 8 for mdboost. Values are the means on the different tuning folds.

            tech = "dem" if "LP_DEMIRIZ" in name else "rat" if "LP_RATSCH" in name else "md"
            alg = "cart" if "CART" in name else "dl85"
            info_per_reg[tech][alg][k] = {
                "best_reg": regulators[-1],  # best regulator for this tech + alg and fold
                # mean accuracy on test sets for each parameter value on folds used during hyperparameter tuning
                "acc": gd_sr.cv_results_['mean_test_acc'],  # it can be used as the test set accuracy for each of the parameters
                "auc": gd_sr.cv_results_['mean_test_auc'],  # mean auc on test sets
                "trees": gd_sr.cv_results_['mean_test_ntrees'],  # mean trees found on test sets
                "iter": gd_sr.cv_results_['mean_test_niter'],  # mean iteration numbers needed on test sets
                "obj": gd_sr.cv_results_['mean_test_obj']  # mean objective value on test sets
            }
            file.write(filename + "," + ",".join(replacenan(gd_sr.cv_results_['mean_test_ntrees'])) + "\n")
            file.write(filename + "," + ",".join(replacenan(gd_sr.cv_results_['mean_test_niter'])) + "\n")
            file.write(filename + "," + ",".join(replacenan(gd_sr.cv_results_['mean_test_acc'])) + "\n")
            file.write(filename + "," + ",".join(replacenan(gd_sr.cv_results_['mean_test_auc'])) + "\n")
            file.write(filename + "," + ",".join(replacenan(gd_sr.cv_results_['mean_test_obj'])) + "\n")
            file.flush()

        file.close()
        print("Model built. Avg duration of building =", round(float(np.mean(fit_times)), 4))
        print("Number of trees =", n_trees, np.mean(n_trees))
        print("Accuracy on training set =", train_scores, round(float(np.mean(train_scores)), 4))
        print("Accuracy on test set =", test_scores, round(float(np.mean(test_scores)), 4))
        print("number of optimality :", n_opti)
        print("list of iterations :", n_iter)
        print("list of time :", fit_times)
        print("list of regulator :", regulators)
        print("list of gammas :", gammas)
        print("list of train_aucs :", train_aucs, round(float(np.mean(train_aucs)), 4))
        print("list of test_aucs :", test_aucs, round(float(np.mean(test_aucs)), 4))
        print("list of objs :", objs, round(float(np.mean(objs)), 4))
        all_best_objs_per_fold[name] = objs
        print("list of n_nodes :", n_nodes)
        print("sum false positives =", sum(fps))
        print("sum false negatives =", sum(fns), "\n\n\n")

    # ========================= #
    #   Optiboost + other reg   #
    # ========================= #
    for name in [x + " + " + y for x in model_names for y in tree_names]:
        print("Optiboost + other reg + {}".format(name))
        print("Dataset :", filename)
        # file = open(name.replace(" + ", "_"), "a+")
        n_trees, fps, fns, fit_times, train_scores, test_scores, n_iter, regulators, gammas, n_opti, n_nodes, train_aucs, test_aucs, objs = [], [], [], [], [], [], [], [], [], 0, [], [], [], []
        # build training set and validation set for the hyperparameter tuning. Use 4 folds for this task
        other_name = "LP_DEMIRIZ + DL85" if name == "LP_DEMIRIZ + CART" else "LP_DEMIRIZ + CART" if name == "LP_DEMIRIZ + DL85" else "LP_RATSCH + DL85" if name == "LP_RATSCH + CART" else "LP_RATSCH + CART" if name == "LP_RATSCH + DL85" else "MDBOOST + DL85" if name == "MDBOOST + CART" else "MDBOOST + CART" if name ==  "MDBOOST + DL85" else None
        old_objs = all_best_objs_per_fold[other_name]

        for k in range(N_FOLDS):
            X_train, X_test, y_train, y_test = X_trains[k], X_tests[k], y_trains[k], y_tests[k]
            print("Fold", k+1, "- Search for the best regulator using grid search...")

            tech, suffix, mod = ("dem", "lpdem_", MODEL_LP_DEMIRIZ) if "LP_DEMIRIZ" in name else ("rat", "lprat_", MODEL_LP_RATSCH) if "LP_RATSCH" in name else ("md", "mdboost_", MODEL_QP_MDBOOST)
            alg, estim, suffix = ("cart", DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=42), suffix + "cart") if "CART" in name else ("dl85", None, suffix + "dl85")
            reg = info_per_reg[tech][alg][k]["best_reg"]

            clf = DL85Booster(base_estimator=estim, max_depth=MAX_DEPTH, model=mod, max_iterations=MAX_ITERATIONS, regulator=reg)
            # clf.fit(X_train, y_train)
            iter_filepath = "output/{}_fixed_reg_depth_{}_fold_{}_{}".format(filename.split(".")[0], MAX_DEPTH, k+1, suffix)
            clf.fit(X_train, y_train, X_test, y_test, iter_file=iter_filepath)
            print("Fold", k+1, "- End {} with best regulator = {} on {}".format(name, reg, filename))

            with open(iter_filepath + ".csv") as f:
                content = f.readlines()
            os.remove(iter_filepath + ".csv")

            def add_info(line: str):
                my_list = line.strip().split(',')
                return [filename_, str(k+1), str(best_reg), "1", suffix] + my_list

            content.pop(0)
            content = [add_info(x) for x in content]
            iters_writer.writerows(content)
            file_iters.flush()

            y_pred = clf.predict(X_test)
            n_trees.append(clf.n_estimators_)
            fit_times.append(clf.duration_)
            train_scores.append(accuracy_score(y_train, clf.predict(X_train)))
            test_scores.append(accuracy_score(y_test, y_pred))
            n_iter.append(clf.n_iterations_)
            objs.append(clf.objective_)
            regulators.append(reg)
            gammas.append(-1)
            test_aucs.append(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
            train_aucs.append(roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1]))
            if "CART" in name:
                n_nodes.append(sum([c.tree_.node_count for c in clf.estimators_]))
            else:  # dl85booster with dl85
                n_nodes.append(clf.get_nodes_count())
            n_opti = n_opti + 1 if clf.optimal_ else n_opti
            fps.append(len([i for i in [j for j, val in enumerate(y_pred) if val == 1] if y_test[i] != 1]))
            fns.append(len([i for i in [j for j, val in enumerate(y_pred) if val == 0] if y_test[i] != 0]))
            to_write += [n_iter[-1], n_trees[-1], fit_times[-1], clf.optimal_, train_scores[-1], test_scores[-1], fps[-1], fns[-1], n_nodes[-1], regulators[-1], gammas[-1], train_aucs[-1], test_aucs[-1], objs[-1]]
            print("Optiboost + other reg + {}".format(name), end=" ")
            print("fold :", k+1, "n_trees :", n_trees[k], "train_acc :", train_scores[k], "test acc :", test_scores[k], "n_nodes :", n_nodes[k], "regu :", regulators[k], "gamma :", gammas[k], "train_auc :", train_aucs[k], "test_auc :", test_aucs[k], "\n")
        print("Model built. Avg duration of building =", round(float(np.mean(fit_times)), 4))
        print("Number of trees =", n_trees, np.mean(n_trees))
        print("Accuracy on training set =", train_scores, round(float(np.mean(train_scores)), 4))
        print("Accuracy on test set =", test_scores, round(float(np.mean(test_scores)), 4))
        print("number of optimality :", n_opti)
        print("list of iterations :", n_iter)
        print("list of time :", fit_times)
        print("list of regulator :", regulators)
        print("list of gammas :", gammas)
        print("list of train_aucs :", train_aucs, round(float(np.mean(train_aucs)), 4))
        print("list of test_aucs :", test_aucs, round(float(np.mean(test_aucs)), 4))
        print("list of objs :", objs, round(float(np.mean(objs)), 4))
        print("list of old objs :", old_objs, round(float(np.mean(old_objs)), 4))
        print("list of n_nodes :", n_nodes)
        print("sum false positives =", sum(fps))
        print("sum false negatives =", sum(fns), "\n\n\n")

    if with_ada is True:
        # ============================================ #
        #         Adaboost | Random | Gradient         #
        # ============================================ #
        def get_ada_trees(clf):
            l = lambda t: tuple(zip(t.feature, t.children_left, t.children_right, t.threshold))
            return len(set(list(map(lambda tree: l(tree.tree_), clf.estimators_))))

        for name, base_clf in zip(["Adaboost + CART"],
                                  [AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=42))]):
            print("{} grid search".format(name))
            print("Dataset :", filename)
            n_trees, fps, fns, fit_times, train_scores, test_scores, n_iter, regulators, gammas, n_opti, n_nodes, train_aucs, test_aucs = [], [], [], [], [], [], [], [], [], 0, [], [], []
            # build training set and validation set for the hyperparameter tuning. Use 4 folds for this task
            for k in range(N_FOLDS):
                X_train, X_test, y_train, y_test = X_trains[k], X_tests[k], y_trains[k], y_tests[k]
                print("Fold", k+1, "- Search for the best regulator using grid search...")
                gd_sr = GridSearchCV(estimator=base_clf, error_score=np.nan, param_grid=parameters_ada, scoring={'auc': 'roc_auc', 'acc': 'accuracy'}, refit='auc', cv=N_FOLDS_TUNING, n_jobs=-1, verbose=0)
                gd_sr.fit(X_train, y_train)
                print("Fold", k+1, "- End CV + {} with best regulator = {} on {}".format(name, -1, filename))
                print("best estimator", gd_sr.best_estimator_)
                clf = gd_sr.best_estimator_
                y_pred = clf.predict(X_test)
                n_trees.append(get_ada_trees(clf))
                fit_times.append(gd_sr.refit_time_)
                train_scores.append(accuracy_score(y_train, clf.predict(X_train)))
                test_scores.append(accuracy_score(y_test, y_pred))
                n_iter.append(len(clf.estimators_))
                regulators.append(gd_sr.best_params_["n_estimators"])
                gammas.append(-1)
                n_nodes.append(sum([c.tree_.node_count for c in clf.estimators_]))
                test_aucs.append(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
                train_aucs.append(roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1]))
                n_opti = n_opti + 1
                fps.append(len([i for i in [j for j, val in enumerate(y_pred) if val == 1] if y_test[i] != 1]))
                fns.append(len([i for i in [j for j, val in enumerate(y_pred) if val == 0] if y_test[i] != 0]))
                to_write += [n_iter[-1], n_trees[-1], fit_times[-1], True, train_scores[-1], test_scores[-1], fps[-1], fns[-1], n_nodes[-1], regulators[-1], gammas[-1], train_aucs[-1], test_aucs[-1], -1]
                print("fold :", k+1, "n_trees :", n_trees[k], "train_acc :", train_scores[k], "test acc :", test_scores[k], "n_nodes :", n_nodes[k], "regu :", regulators[k], "gamma :", gammas[k], "train_auc :", train_aucs[k], "test_auc :", test_aucs[k], "\n")
            print("Model built. Avg duration of building =", round(float(np.mean(fit_times)), 4))
            print("Number of trees =", n_trees, np.mean(n_trees))
            print("Accuracy on training set =", train_scores, round(float(np.mean(train_scores)), 4))
            print("Accuracy on test set =", test_scores, round(float(np.mean(test_scores)), 4))
            print("number of optimality :", n_opti)
            print("list of iterations :", n_iter)
            print("list of time :", fit_times)
            print("list of regulator :", regulators)
            print("list of gammas :", gammas)
            print("list of train_aucs :", train_aucs, round(float(np.mean(train_aucs)), 4))
            print("list of test_aucs :", test_aucs, round(float(np.mean(test_aucs)), 4))
            print("list of n_nodes :", n_nodes)
            print("sum false positives =", sum(fps))
            print("sum false negatives =", sum(fns), "\n\n")

        # ===================================== #
        #         Adaboost fixed n_trees        #
        # ===================================== #
        n_estims = 100
        print("Adaboost + CART {}".format(n_estims))
        print("Dataset :", filename)
        clf_results = cross_validate(estimator=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=MAX_DEPTH), n_estimators=n_estims),
                                     X=X, y=y, scoring='accuracy', cv=custom_cv_ada, n_jobs=-1, verbose=VERBOSE_LEVEL, return_train_score=True, return_estimator=True, error_score=np.nan)
        n_trees = [get_ada_trees(c) for c in clf_results['estimator']]
        test_aucs = [roc_auc_score(y_tests[k], clf_results['estimator'][k].predict_proba(X_tests[k])[:, 1]) for k in range(N_FOLDS)]
        train_aucs = [roc_auc_score(y_trains[k], clf_results['estimator'][k].predict_proba(X_trains[k])[:, 1]) for k in range(N_FOLDS)]
        n_nodes = [sum([dectree.tree_.node_count for dectree in ada_estim.estimators_]) for ada_estim in clf_results['estimator']]
        fps = [len([i for i in [j for j, val in enumerate(clf_results['estimator'][k].predict(X_tests[k])) if val == 1] if y_tests[k][i] != 1]) for k in range(N_FOLDS)]
        fns = [len([i for i in [j for j, val in enumerate(clf_results['estimator'][k].predict(X_tests[k])) if val == 0] if y_tests[k][i] != 0]) for k in range(N_FOLDS)]
        print("Model built. Avg duration of building =", round(float(np.mean(clf_results['fit_time'])), 4))
        print("Avg number of trees =", round(float(np.mean(n_trees)), 4))
        print("Accuracy on training set =", clf_results['train_score'], round(float(np.mean(clf_results['train_score'])), 4))
        print("Avg accuracy on test set =", clf_results['test_score'], round(float(np.mean(clf_results['test_score'])), 4))
        print("list of time :", clf_results['fit_time'])
        print("list of train_aucs :", train_aucs, round(float(np.mean(train_aucs)), 4))
        print("list of test_aucs :", test_aucs, round(float(np.mean(test_aucs)), 4))
        print("sum false positives =", sum(fps))
        print("sum false negatives =", sum(fns), "\n\n\n")
        tmp_to_write = [[n_trees[k], n_trees[k], clf_results['fit_time'][k], True, clf_results['train_score'][k], clf_results['test_score'][k], fps[k], fns[k], n_nodes[k], n_estims, -1, train_aucs[k], test_aucs[k], -1] for k in range(N_FOLDS)]
        to_write += [val for sublist in tmp_to_write for val in sublist]

    stats_writer.writerow(list(map(lambda x: str(x), to_write)))
    # file_out.write(";".join(map(lambda x: str(x), to_write)) + "\n")
    file_out.flush()
    print(to_write)
file_out.close()

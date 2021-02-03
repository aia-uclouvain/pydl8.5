from sklearn.svm import SVC
from sklearn.datasets import load_iris, load_digits, fetch_openml
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, StratifiedKFold, \
    StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelBinarizer, KBinsDiscretizer, Binarizer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.utils import check_random_state
from dl85 import DL85Boostera, DL85Booster, MODEL_RATSCH, MODEL_DEMIRIZ, MODEL_AGLIN, DL85Classifier
import time
import numpy as np
from sklearn.metrics import confusion_matrix
import subprocess

depth, time_limit, N_FOLDS = 2, 0, 5

# dataset = np.genfromtxt("../datasets/paper_test.txt", delimiter=" ")
# dataset = np.genfromtxt("../datasets/zoo-1.txt", delimiter=" ")
# dataset = np.genfromtxt("../datasets/kr-vs-kp.txt", delimiter=" ")
# dataset = np.genfromtxt("../datasets/hepatitis.txt", delimiter=" ")
# dataset = np.genfromtxt("../datasets/audiology.txt", delimiter=" ")
dataset = np.genfromtxt("../datasets/australian-credit.txt", delimiter=" ")
# X, y = dataset[:, 1:], dataset[:, 0]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=525)
X, y = dataset[:, 1:], dataset[:, 0]
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
        # train_indices.append(train_index[:800])
        # test_indices.append(np.concatenate((train_index[-800:], test_index)))
        # X_trains.append(X[train_index[:800]])
        # y_trains.append(y[train_index[:800]])
        # X_tests.append(X[np.concatenate((train_index[-800:], test_index))])
        # y_tests.append(y[np.concatenate((train_index[-800:], test_index))])
custom_cv_dl85 = zip(train_indices, test_indices)
custom_cv_cart = zip(train_indices, test_indices)
custom_cv_opti = zip(train_indices, test_indices)
custom_cv_ada = zip(train_indices, test_indices)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_test, y_train, y_test = X_trains[0], X_tests[0], y_trains[0], y_tests[0]
# np.savetxt("../datasets/hp.txt", np.concatenate((y_train.reshape(-1, 1), X_train), axis=1), fmt="%d", delimiter=" ")

from math import log10 as ln
from math import exp


def ada(clf, x, y, n):
    s_w = [1 / x.shape[0]] * x.shape[0]
    clf.fit(x, y, sample_weight=s_w)
    err = 0
    predd = clf.predict(x)
    for i, pred in enumerate(predd):
        if pred != y[i]:
            err += s_w[i]
    t_w = 0.5 * ln((1 - err) / err)

    for i in range(n) and err != 0.5:
        for i, ww in enumerate(s_w):
            s_w[i] = ww * exp(-t_w[i] * y[i] * predd[i])
        s_w = list(map(lambda x: x / sum(s_w), s_w))
        clf.fit(x, y, sample_weight=s_w)
        err = 0
        predd = clf.predict(x)
        for i, pred in enumerate(predd):
            if pred != y[i]:
                err += s_w[i]
        t_w = 0.5 * ln((1 - err) / err)


# clf = DL85Boostera(max_depth=depth, regulator=0.1, max_iterations=-1, quiet=False, gamma='nscale', model='cvxpy')
# start = time.perf_counter()
# print("Model building...")
# clf.fit(X_train, y_train)
# print("Model built in ", time.perf_counter() - start, "second(s)")
# y_pred = clf.predict(X_test)
# print("Accuracy DL8.5 on training set =", accuracy_score(y_train, clf.predict(X_train)))
# print("Accuracy DL8.5 on test set =", accuracy_score(y_test, y_pred))


# clf = DL85Booster(max_depth=depth, regulator=0.3, quiet=False, model=MODEL_AGLIN)
# # clf = DL85Boostera(max_depth=depth, regulator=0, model=5)
# start = time.perf_counter()
# print("Model building...")
# clf.fit(X_train, y_train)
# print("Model built in ", time.perf_counter() - start, "second(s)")
# y_pred = clf.predict(X_test)
# print("Accuracy DL8.5 on training set =", accuracy_score(y_train, clf.predict(X_train)))
# print("Accuracy DL8.5 on test set =", accuracy_score(y_test, y_pred))
# print(clf.n_estimators_)


regg = 15
clf_results = cross_validate(
    estimator=DL85Boostera(max_depth=depth, regulator=regg, max_iterations=5, quiet=True, gamma=None), X=X, y=y,
    scoring='accuracy',
    cv=custom_cv_opti, n_jobs=-1, verbose=10, return_train_score=True, return_estimator=True, error_score=np.nan)
n_iter = [clf_results['estimator'][k].n_iterations_ for k in range(N_FOLDS)]
n_trees = [clf_results['estimator'][k].n_estimators_ for k in range(N_FOLDS)]
fps = [len([i for i in [j for j, val in enumerate(clf_results['estimator'][k].predict(X_tests[k])) if val == 1] if
            y_tests[k][i] != 1]) for k in range(N_FOLDS)]
fns = [len([i for i in [j for j, val in enumerate(clf_results['estimator'][k].predict(X_tests[k])) if val == 0] if
            y_tests[k][i] != 0]) for k in range(N_FOLDS)]
print("Model built. Avg duration of building =", round(float(np.mean(clf_results['fit_time'])), 4))
print("Avg number of trees =", n_trees, round(float(np.mean(n_trees)), 4))
print("Accuracy on training set =", clf_results['train_score'], round(float(np.mean(clf_results['train_score'])), 4))
print("Avg accuracy on test set =", clf_results['test_score'], round(float(np.mean(clf_results['test_score'])), 4))
print("list of time :", clf_results['fit_time'])
n_nodes = [clf_results['estimator'][k].get_nodes_count() for k in range(N_FOLDS)]
print("list of n_nodes :", n_nodes)
print("sum false positives =", sum(fps))
print("sum false negatives =", sum(fns), "\n\n\n")

clf_results = cross_validate(
    estimator=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=depth), n_estimators=5), X=X, y=y,
    scoring='accuracy',
    cv=custom_cv_ada, n_jobs=-1, verbose=10, return_train_score=True, return_estimator=True, error_score=np.nan)
n_iter = [len(clf_results['estimator'][k].estimators_) for k in range(N_FOLDS)]
n_trees = [len(clf_results['estimator'][k].estimators_) for k in range(N_FOLDS)]
fps = [len([i for i in [j for j, val in enumerate(clf_results['estimator'][k].predict(X_tests[k])) if val == 1] if
            y_tests[k][i] != 1]) for k in range(N_FOLDS)]
fns = [len([i for i in [j for j, val in enumerate(clf_results['estimator'][k].predict(X_tests[k])) if val == 0] if
            y_tests[k][i] != 0]) for k in range(N_FOLDS)]
print("Model built. Avg duration of building =", round(float(np.mean(clf_results['fit_time'])), 4))
print("Avg number of trees =", n_trees, round(float(np.mean(n_trees)), 4))
print("Accuracy on training set =", clf_results['train_score'], round(float(np.mean(clf_results['train_score'])), 4))
print("Avg accuracy on test set =", clf_results['test_score'], round(float(np.mean(clf_results['test_score'])), 4))
print("list of time :", clf_results['fit_time'])
n_nodes = [sum([c.tree_.node_count for c in clf_results['estimator'][k].estimators_]) for k in range(N_FOLDS)]
print("list of n_nodes :", n_nodes)
print("sum false positives =", sum(fps))
print("sum false negatives =", sum(fns), "\n\n\n")

# clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=depth), n_estimators=clf.n_estimators_)
# start = time.perf_counter()
# print("Model building...")
# clf.fit(X_train, y_train)
# print("Model built in", time.perf_counter() - start, "second(s)")
# y_pred = clf.predict(X_test)
# print("Accuracy Adaboost on training set =", accuracy_score(y_train, clf.predict(X_train)))
# print("Accuracy Adaboost on test set =", accuracy_score(y_test, y_pred))


clf = DecisionTreeClassifier(max_depth=depth, criterion='entropy')
start = time.perf_counter()
print("Model building...")
clf.fit(X_train, y_train, sample_weight=[1 / X_train.shape[0]] * X_train.shape[0])
print("Model built in", time.perf_counter() - start, "second(s)")
y_pred = clf.predict(X_test)
print("Accuracy Cart on training set =", accuracy_score(y_train, clf.predict(X_train)))
print("Accuracy Cart on test set =", accuracy_score(y_test, y_pred))

clf = DL85Classifier(max_depth=depth, print_output=True)
start = time.perf_counter()
print("Model building...")
clf.fit(X_train, y_train)
print("Model built in", time.perf_counter() - start, "second(s)")
y_pred = clf.predict(X_test)
print("Accuracy DL85 on training set =", accuracy_score(y_train, clf.predict(X_train)))
print("Accuracy DL85 on test set =", accuracy_score(y_test, y_pred))

clf = DL85Classifier(max_depth=depth, print_output=True)
start = time.perf_counter()
print("Model building...")
clf.fit(X_train, y_train, sample_weight=[1 / X_train.shape[0]] * X_train.shape[0])
print("Model built in", time.perf_counter() - start, "second(s)")
y_pred = clf.predict(X_test)
print("Accuracy DL85 on training set =", accuracy_score(y_train, clf.predict(X_train)))
print("Accuracy DL85 on test set =", accuracy_score(y_test, y_pred))

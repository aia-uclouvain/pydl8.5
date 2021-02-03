from sklearn.svm import SVC
from sklearn.datasets import load_iris, load_digits, fetch_openml
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelBinarizer, KBinsDiscretizer, Binarizer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.utils import check_random_state
from dl85 import DL85Boostera, DL85Classifier
import time
import numpy as np
from sklearn.metrics import confusion_matrix
import subprocess

depth, time_limit, N_FOLDS = 2, 0, 5

# dataset = np.genfromtxt("../datasets/paper_test.txt", delimiter=" ")
# dataset = np.genfromtxt("../datasets/zoo-1.txt", delimiter=" ")
# dataset = np.genfromtxt("../datasets/kr-vs-kp.txt", delimiter=" ")
dataset = np.genfromtxt("../datasets/kr-vs-kp.txt", delimiter=" ")
X, y = dataset[:, 1:], dataset[:, 0]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=525)

i, j = 3, 1
n_train = 90

X_trainss, X_tests, y_trainss, y_tests = [], [], [], []
kf = StratifiedKFold(n_splits=N_FOLDS)
train_indices, test_indices = [], []
for train_index, test_index in kf.split(X, y):
    X_trainss.append(X[train_index[:n_train]])
    y_trainss.append(y[train_index[:n_train]])
    X_tests.append(X[np.concatenate((train_index[-n_train:], test_index))])
    y_tests.append(y[np.concatenate((train_index[-n_train:], test_index))])

X_trains, X_valids, y_trains, y_valids = [], [], [], []
kf = StratifiedKFold(n_splits=N_FOLDS - 1)
for train_index, test_index in kf.split(X_trainss[i], y_trainss[i]):
    if X.shape[0] <= 1500:  # 3/4 tr - 1/4 te
        X_trains.append(X[train_index])
        y_trains.append(y[train_index])
        X_valids.append(X[test_index])
        y_valids.append(y[test_index])
    else:  # 1/4 tr - 3/4 te
        X_trains.append(X[test_index])
        y_trains.append(y[test_index])
        X_valids.append(X[train_index])
        y_valids.append(y[train_index])


# clf = DL85Classifier(max_depth=depth, print_output=True, verbose=True, quiet=False)
# clf.fit(X_trains[j], y_trains[j])
#
# print(y_trains[j])
# print(len(y_trains[j]))


clf = DL85Boostera(max_depth=depth, regulator=2.5, max_iterations=5, quiet=True, gamma='nscale', model='cvxpy', print_output=True)
start = time.perf_counter()
print("Model building...")
clf.fit(X_trains[j], y_trains[j])
print("Model built in ", time.perf_counter() - start, "second(s)")
y_pred = clf.predict(X_tests[i])
print("Accuracy DL8.5 on training set =", accuracy_score(y_trains[j], clf.predict(X_trains[j])))
print("Accuracy DL8.5 on test set =", accuracy_score(y_tests[i], y_pred))


# clf_results = cross_validate(estimator=DL85Boostera(max_depth=depth, regulator=0.03125, max_iterations=-1, quiet=True, gamma=None), X=X, y=y, scoring='accuracy',
#                              cv=N_FOLDS, n_jobs=-1, verbose=10, return_train_score=True, return_estimator=True, error_score=np.nan)
# n_trees = [1 for k in range(N_FOLDS)]
# # fps = [len([i for i in [j for j, val in enumerate(clf_results['estimator'][k].predict(X_tests[k])) if val == 1] if y_tests[k][i] != 1]) for k in range(N_FOLDS)]
# # fns = [len([i for i in [j for j, val in enumerate(clf_results['estimator'][k].predict(X_tests[k])) if val == 0] if y_tests[k][i] != 0]) for k in range(N_FOLDS)]
# print("Model built. Avg duration of building =", round(float(np.mean(clf_results['fit_time'])), 4))
# print("Avg number of trees =", round(float(np.mean(n_trees)), 4))
# print("Accuracy on training set =", clf_results['train_score'], round(float(np.mean(clf_results['train_score'])), 4))
# print("Avg accuracy on test set =", clf_results['test_score'], round(float(np.mean(clf_results['test_score'])), 4))
# print("list of time :", clf_results['fit_time'])
# # print("sum false positives =", sum(fps))
# # print("sum false negatives =", sum(fns), "\n\n\n")


# clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=depth), n_estimators=clf.n_estimators_)
# start = time.perf_counter()
# print("Model building...")
# clf.fit(X_train, y_train)
# print("Model built in", time.perf_counter() - start, "second(s)")
# y_pred = clf.predict(X_test)
# print("Accuracy Adaboost on training set =", accuracy_score(y_train, clf.predict(X_train)))
# print("Accuracy Adaboost on test set =", accuracy_score(y_test, y_pred))
#
#
# clf = DecisionTreeClassifier(max_depth=depth)
# start = time.perf_counter()
# print("Model building...")
# clf.fit(X_train, y_train)
# print("Model built in", time.perf_counter() - start, "second(s)")
# y_pred = clf.predict(X_test)
# print("Accuracy Cart on training set =", accuracy_score(y_train, clf.predict(X_train)))
# print("Accuracy Cart on test set =", accuracy_score(y_test, y_pred))

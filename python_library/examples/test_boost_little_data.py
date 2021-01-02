from sklearn.svm import SVC
from sklearn.datasets import load_iris, load_digits, fetch_openml
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelBinarizer, KBinsDiscretizer, Binarizer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.utils import check_random_state
from dl85 import DL85Boostera
import time
import numpy as np
from sklearn.metrics import confusion_matrix
import subprocess

depth, time_limit, N_FOLDS = 2, 0, 5

# dataset = np.genfromtxt("../datasets/paper_test.txt", delimiter=" ")
dataset = np.genfromtxt("../datasets/zoo-1.txt", delimiter=" ")
# dataset = np.genfromtxt("../datasets/kr-vs-kp.txt", delimiter=" ")
X, y = dataset[:, 1:], dataset[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

clf = DL85Boostera(max_depth=depth, regulator=1, max_iterations=-1, quiet=False, gamma='nscale')
start = time.perf_counter()
print("Model building...")
clf.fit(X_train, y_train)
print("Model built in ", time.perf_counter() - start, "second(s)")
y_pred = clf.predict(X_test)
print("Accuracy DL8.5 on training set =", accuracy_score(y_train, clf.predict(X_train)))
print("Accuracy DL8.5 on test set =", accuracy_score(y_test, y_pred))


clf_results = cross_validate(estimator=DL85Boostera(max_depth=depth, regulator=0.03125, max_iterations=-1, quiet=True, gamma=None), X=X, y=y, scoring='accuracy',
                             cv=N_FOLDS, n_jobs=-1, verbose=10, return_train_score=True, return_estimator=True, error_score=np.nan)
n_trees = [1 for k in range(N_FOLDS)]
# fps = [len([i for i in [j for j, val in enumerate(clf_results['estimator'][k].predict(X_tests[k])) if val == 1] if y_tests[k][i] != 1]) for k in range(N_FOLDS)]
# fns = [len([i for i in [j for j, val in enumerate(clf_results['estimator'][k].predict(X_tests[k])) if val == 0] if y_tests[k][i] != 0]) for k in range(N_FOLDS)]
print("Model built. Avg duration of building =", round(float(np.mean(clf_results['fit_time'])), 4))
print("Avg number of trees =", round(float(np.mean(n_trees)), 4))
print("Accuracy on training set =", clf_results['train_score'], round(float(np.mean(clf_results['train_score'])), 4))
print("Avg accuracy on test set =", clf_results['test_score'], round(float(np.mean(clf_results['test_score'])), 4))
print("list of time :", clf_results['fit_time'])
# print("sum false positives =", sum(fps))
# print("sum false negatives =", sum(fns), "\n\n\n")


clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=depth), n_estimators=clf.n_estimators_)
start = time.perf_counter()
print("Model building...")
clf.fit(X_train, y_train)
print("Model built in", time.perf_counter() - start, "second(s)")
y_pred = clf.predict(X_test)
print("Accuracy Adaboost on training set =", accuracy_score(y_train, clf.predict(X_train)))
print("Accuracy Adaboost on test set =", accuracy_score(y_test, y_pred))


clf = DecisionTreeClassifier(max_depth=depth)
start = time.perf_counter()
print("Model building...")
clf.fit(X_train, y_train)
print("Model built in", time.perf_counter() - start, "second(s)")
y_pred = clf.predict(X_test)
print("Accuracy Cart on training set =", accuracy_score(y_train, clf.predict(X_train)))
print("Accuracy Cart on test set =", accuracy_score(y_test, y_pred))

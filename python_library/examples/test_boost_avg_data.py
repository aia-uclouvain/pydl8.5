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

depth, time_limit, N_FOLDS = 1, 0, 5

dataset = np.genfromtxt("../datasets/anneal.txt", delimiter=" ")
# dataset = np.genfromtxt("../datasets/paper.txt", delimiter=" ")
X = dataset[:, 1:]
y = dataset[:, 0]


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=0)
clf = DL85Boostera(max_depth=depth, min_sup=1, max_iterations=100000, regulator=120, time_limit=time_limit, quiet=False)
start = time.perf_counter()
print("Model building...")
# clf.fit(X, y)
clf.fit(X_train, y_train)
duration = time.perf_counter() - start
print("Model built. Duration of building =", round(duration, 4))
y_pred = clf.predict(X_test)
print("Confusion Matrix below")
print(confusion_matrix(y_test, y_pred))
print("Accuracy DL8.5 on training set =", round(clf.accuracy_, 4))
print("Accuracy DL8.5 on test set =", round(accuracy_score(y_test, y_pred), 4))
print(clf.problem)
print(clf.regulator, clf.n_estimators_)
for c in clf.estimators_:
    print(c.tree_)


print()
print("Adaboost")
clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=depth), n_estimators=clf.n_estimators_)
# clf = DecisionTreeClassifier(max_depth=depth)
# clf = DL85Booster(max_depth=depth, min_sup=1, regulator=0.007137981694607998, time_limit=time_limit)
start = time.perf_counter()
print("Model building...")
clf.fit(X_train, y_train)
duration = time.perf_counter() - start
print("Model built. Duration of building =", round(duration, 4))
y_pred = clf.predict(X_test)
print("Confusion Matrix below")
print(confusion_matrix(y_test, y_pred))
print("Accuracy DL8.5 on training set =", round(accuracy_score(y_train, clf.predict(X_train)), 4))
print("Accuracy DL8.5 on test set =", round(accuracy_score(y_test, y_pred), 4), "\n\n\n")


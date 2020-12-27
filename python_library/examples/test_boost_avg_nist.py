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

train = np.genfromtxt("../datasets/boosting/nist/optdigits.tra", delimiter=",")
test = np.genfromtxt("../datasets/boosting/nist/optdigits.tes", delimiter=",")
# split features and target
X_train, y_train = train[:, :-1], train[:, -1]
X_test, y_test = test[:, :-1], test[:, -1]
# select a slice of training data
# X_train, _, y_train, _ = train_test_split(X_train, y_train, stratify=y_train, train_size=500)
# convert values to int
X_train, y_train = X_train.astype('int32'), y_train.astype('int32')
X_test, y_test = X_test.astype('int32'), y_test.astype('int32')
# binarize the features
enc = Binarizer(threshold=1)
X_train = enc.fit_transform(X_train)
X_test = enc.fit_transform(X_test)
# create 2 classes even or odd
# biner = lambda x: x % 2
# y_train = biner(y_train)
# y_test = biner(y_test)
# another way to create more complex binarization
# def binner(x):
#    if x in [5, 2]:
#        return 0
#    elif x in [3, 8]:
#        return 1
#    elif x in [1, 7, 4]:
#        return 2
#    elif x in [0, 9, 6]:
#        return 3


def binner(x):
    if x in [0, 1, 2, 3]:
        return 0
    elif x in [4, 5, 6]:
        return 1
    elif x in [7, 8, 9]:
        return 2


y_train = np.array(list(map(binner, y_train)))
y_test = np.array(list(map(binner, y_test)))
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)
print(set(y_train), set(y_test))


clf = DL85Boostera(max_depth=depth, min_sup=1, max_iterations=100000, regulator=5, time_limit=time_limit, quiet=False)
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
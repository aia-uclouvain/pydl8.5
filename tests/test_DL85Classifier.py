from sklearn.utils.estimator_checks import check_estimator
from sklearn.model_selection import train_test_split

from pydl85 import Cache_Type, Wipe_Type
from pydl85.supervised.classifiers import DL85Classifier
import numpy as np
from random import randrange
import os
from sklearn.metrics import accuracy_score

# set the current file path as working directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


def test_predict():
    dataset = np.genfromtxt("../datasets/anneal.txt", delimiter=' ')
    X = dataset[:, 1:]
    y = dataset[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    clf1 = DL85Classifier(max_depth=randrange(1, 4), min_sup=randrange(1, X_train.shape[0] // 4))
    clf1.fit(X_train, y_train)

    # if clf1.sol_size in [8, 9]:
    y_pred1 = clf1.predict(X_test)

    def is_class(y_pred):
        for i in y_pred:
            if i not in list(set(list(clf1.classes_))):
                return False
        return True

    assert len(y_pred1) == X_test.shape[0] and is_class(y_pred1) is True  # list(set(y_pred1)) == list(set(list(clf1.classes_)))


def test_depth_2():
    solutions = [137, 10, 87, 22, 177, 267, 60, 16, 70, 32, 418, 599, 22, 252, 153, 58, 9, 55, 508, 282, 75, 17, 437, 0]
    mypath = "../datasets"
    onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f)) and "compas" not in f and "txt" in f]
    onlyfiles = sorted(onlyfiles)
    depth = 2

    for i, file in enumerate(onlyfiles):
        dataset = np.genfromtxt(os.path.join(mypath, file), delimiter=' ')
        X, y = dataset[:, 1:], dataset[:, 0]

        clf = DL85Classifier(max_depth=depth, time_limit=600)
        clf.fit(X, y)
        y_pred = clf.predict(X)
        assert clf.depth_ <= depth
        assert clf.size_ <= 2 ** (depth+1) - 1
        assert clf.error_ <= solutions[i]
        assert clf.error_ == int(X.shape[0] - X.shape[0] * accuracy_score(y, y_pred))


def test_depth_3():
    solutions = [112, 5, 73, 15, 162, 236, 41, 10, 61, 22, 198, 369, 12, 8, 47, 46, 0, 29, 224, 216, 26, 12, 403, 0]
    mypath = "../datasets"
    onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f)) and "compas" not in f and "txt" in f]
    onlyfiles = sorted(onlyfiles)
    depth = 3

    for i, file in enumerate(onlyfiles):
        if file in ["ionosphere.txt", "letter.txt", "mushroom.txt", "pendigits.txt", "splice-1.txt", "vehicle.txt"]:
            continue
        dataset = np.genfromtxt(os.path.join(mypath, file), delimiter=' ')
        X, y = dataset[:, 1:], dataset[:, 0]

        clf = DL85Classifier(max_depth=depth, time_limit=600)
        clf.fit(X, y)
        y_pred = clf.predict(X)
        assert clf.depth_ <= depth
        assert clf.size_ <= 2 ** (depth+1) - 1
        assert clf.error_ <= solutions[i]
        assert clf.error_ == int(X.shape[0] - X.shape[0] * accuracy_score(y, y_pred))


def test_user_tids_error_class():
    dataset = np.genfromtxt("../datasets/anneal.txt", delimiter=' ')
    X, y = dataset[:, 1:], dataset[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    def error(tids, y):
        classes, supports = np.unique(y.take(list(tids)), return_counts=True)
        maxindex = np.argmax(supports)
        return sum(supports) - supports[maxindex], classes[maxindex]

    clf = DL85Classifier(max_depth=2, error_function=lambda tids: error(tids, y_train), time_limit=600)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    assert clf.error_ == 114.0
    assert round(clf.accuracy_, 4) == 0.8243
    assert round(accuracy_score(y_test, y_pred), 4) == 0.8466
    assert str(clf.tree_) == "{'feat': 5, 'left': {'feat': 32, 'left': {'value': 1, 'error': 44}, 'right': {'value': 0, 'error': 2}}, 'right': {'feat': 46, 'left': {'value': 1, 'error': 68}, 'right': {'value': 0, 'error': 0}}, 'proba': [0.2357473035439137, 0.7642526964560863]}"


def test_user_sups_error_class():
    dataset = np.genfromtxt("../datasets/anneal.txt", delimiter=' ')
    X, y = dataset[:, 1:], dataset[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    def error(sup_iter):
        supports = list(sup_iter)
        maxindex = np.argmax(supports)
        return sum(supports) - supports[maxindex], maxindex

    clf = DL85Classifier(max_depth=2, fast_error_function=error, time_limit=600)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy DL8.5 on training set =", round(clf.accuracy_, 4))
    print("Accuracy DL8.5 on test set =", round(accuracy_score(y_test, y_pred), 4))

    assert clf.error_ == 114.0
    assert round(clf.accuracy_, 4) == 0.8243
    assert round(accuracy_score(y_test, y_pred), 4) == 0.8466
    assert str(clf.tree_) == "{'feat': 5, 'left': {'feat': 32, 'left': {'value': 1, 'error': 44}, 'right': {'value': 0, 'error': 2}}, 'right': {'feat': 46, 'left': {'value': 1, 'error': 68}, 'right': {'value': 0, 'error': 0}}, 'proba': [0.2357473035439137, 0.7642526964560863]}"


def test_memory_limitation():
    dataset = np.genfromtxt("../datasets/anneal.txt", delimiter=' ')
    X, y = dataset[:, 1:], dataset[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    clf = DL85Classifier(max_depth=4, cache_type=Cache_Type.Cache_TrieItemset, maxcachesize=5000, wipe_factor=0.4, wipe_type=Wipe_Type.Reuses)
    clf.fit(X_train, y_train)

    assert clf.error_ == 74

# check_estimator(DL85Classifier())

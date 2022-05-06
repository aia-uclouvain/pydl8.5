from sklearn.utils.estimator_checks import check_estimator
from sklearn.model_selection import train_test_split
from ..classifier import DL85Classifier
import numpy as np
from random import randrange
from os.path import join
from sklearn.metrics import accuracy_score

dev = "../../../../"
prod = ""
prefix = prod
# prefix = dev


def test_fit():
    dataset = np.genfromtxt(prefix + "datasets/anneal.txt", delimiter=' ').astype('int32')
    X, y = dataset[:, 1:], dataset[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    clf1 = DL85Classifier(max_depth=randrange(1, 4), min_sup=randrange(1, X_train.shape[0] // 4))
    clf1.fit(X_train, y_train)
    assert clf1.sol_size in [6, 10]


def test_predict():
    dataset = np.genfromtxt(prefix + "datasets/anneal.txt", delimiter=' ').astype('int32')
    X, y = dataset[:, 1:], dataset[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    clf1 = DL85Classifier(max_depth=randrange(1, 4), min_sup=randrange(1, X_train.shape[0] // 4))
    clf1.fit(X_train, y_train)

    if clf1.sol_size == 10:
        y_pred1 = clf1.predict(X_test)

        def is_class(y_pred):
            for i in y_pred:
                if i not in list(set(list(clf1.classes_))):
                    return False
            return True

        assert y_pred1.shape[0] == X_test.shape[0] and is_class(y_pred1) is True  # list(set(y_pred1)) == list(set(list(clf1.classes_)))
    else:
        assert clf1.sol_size == 6


def test_depth_2():
    solutions = [137, 10, 87, 22, 177, 267, 60, 16, 70, 32, 418, 599, 22, 252, 153, 58, 9, 55, 508, 282, 75, 17, 437, 0]
    mypath = prefix + "datasets"
    datasets = ['anneal.txt', 'audiology.txt', 'australian-credit.txt', 'breast-wisconsin.txt', 'diabetes.txt', 'german-credit.txt', 'heart-cleveland.txt', 'hepatitis.txt', 'hypothyroid.txt', 'ionosphere.txt', 'kr-vs-kp.txt', 'letter.txt', 'lymph.txt', 'mushroom.txt', 'pendigits.txt', 'primary-tumor.txt', 'segment.txt', 'soybean.txt', 'splice-1.txt', 'tic-tac-toe.txt', 'vehicle.txt', 'vote.txt', 'yeast.txt', 'zoo-1.txt']
    depth = 2

    for i, file in enumerate(datasets):
        dataset = np.genfromtxt(join(mypath, file), delimiter=' ').astype('int32')
        X, y = dataset[:, 1:], dataset[:, 0]
        clf = DL85Classifier(max_depth=depth, time_limit=600)
        clf.fit(X, y)
        y_pred = clf.predict(X)
        assert clf.depth_ <= depth
        assert clf.size_ <= 2 ** (depth+1) - 1
        assert clf.error_ == solutions[i]
        assert clf.error_ == int(X.shape[0] - X.shape[0] * accuracy_score(y, y_pred))


def test_depth_3():
    solutions = [112, 5, 73, 15, 162, 236, 41, 10, 61, 22, 198, 369, 12, 8, 47, 46, 0, 29, 224, 216, 26, 12, 403, 0]
    mypath = prefix + "datasets"
    datasets = ['anneal.txt', 'audiology.txt', 'australian-credit.txt', 'breast-wisconsin.txt', 'diabetes.txt', 'german-credit.txt', 'heart-cleveland.txt', 'hepatitis.txt', 'hypothyroid.txt', 'ionosphere.txt', 'kr-vs-kp.txt', 'letter.txt', 'lymph.txt', 'mushroom.txt', 'pendigits.txt', 'primary-tumor.txt', 'segment.txt', 'soybean.txt', 'splice-1.txt', 'tic-tac-toe.txt', 'vehicle.txt', 'vote.txt', 'yeast.txt', 'zoo-1.txt']
    depth = 3

    for i, file in enumerate(datasets):
        if file in ["ionosphere.txt", "letter.txt", "mushroom.txt", "pendigits.txt", "splice-1.txt", "vehicle.txt"]:
            continue
        dataset = np.genfromtxt(join(mypath, file), delimiter=' ').astype('int32')
        X, y = dataset[:, 1:], dataset[:, 0]
        clf = DL85Classifier(max_depth=depth, time_limit=600)
        clf.fit(X, y)
        y_pred = clf.predict(X)
        assert clf.depth_ <= depth
        assert clf.size_ <= 2 ** (depth+1) - 1
        assert clf.error_ == solutions[i]
        assert clf.error_ == int(X.shape[0] - X.shape[0] * accuracy_score(y, y_pred))


def test_depth_4():
    solutions = [91, 1, 56, 7, 137, 204, 25, 3, 53, 11, 144, 550, 3, 0, 14, 34, 0, 14, 141, 137, 13, 5, 366, 0]
    mypath = prefix + "datasets"
    datasets = ['anneal.txt', 'audiology.txt', 'australian-credit.txt', 'breast-wisconsin.txt', 'diabetes.txt', 'german-credit.txt', 'heart-cleveland.txt', 'hepatitis.txt', 'hypothyroid.txt', 'ionosphere.txt', 'kr-vs-kp.txt', 'letter.txt', 'lymph.txt', 'mushroom.txt', 'pendigits.txt', 'primary-tumor.txt', 'segment.txt', 'soybean.txt', 'splice-1.txt', 'tic-tac-toe.txt', 'vehicle.txt', 'vote.txt', 'yeast.txt', 'zoo-1.txt']
    depth = 4

    for i, file in enumerate(datasets):
        if file not in ["hepatitis.txt", "lymph.txt", "mushroom.txt", "primary-tumor.txt", "segment.txt", "soybean.txt", "tic-tac-toe.txt", "vote.txt", "zoo-1.txt"]:
            continue
        dataset = np.genfromtxt(join(mypath, file), delimiter=' ').astype('int32')
        X, y = dataset[:, 1:], dataset[:, 0]
        clf = DL85Classifier(max_depth=depth, time_limit=600)
        clf.fit(X, y)
        y_pred = clf.predict(X)
        assert clf.depth_ <= depth
        assert clf.size_ <= 2 ** (depth+1) - 1
        assert clf.error_ == solutions[i]
        assert clf.error_ == int(X.shape[0] - X.shape[0] * accuracy_score(y, y_pred))


# check_estimator(DL85Classifier())

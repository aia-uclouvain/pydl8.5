from sklearn.utils.estimator_checks import check_estimator
from sklearn.model_selection import train_test_split
from ..classifier import DL85Classifier
import numpy as np
from random import randrange


def test_fit():
    dataset = np.genfromtxt("datasets/anneal.txt", delimiter=' ')
    X = dataset[:, 1:]
    y = dataset[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    clf1 = DL85Classifier(max_depth=randrange(1, 4), min_sup=randrange(1, X_train.shape[0] // 4))
    clf1.fit(X_train, y_train)

    assert clf1.sol_size in [4, 5, 8, 9]


def test_predict():
    dataset = np.genfromtxt("datasets/anneal.txt", delimiter=' ')
    X = dataset[:, 1:]
    y = dataset[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    clf1 = DL85Classifier(max_depth=randrange(1, 4), min_sup=randrange(1, X_train.shape[0] // 4))
    clf1.fit(X_train, y_train)

    if clf1.sol_size in [8, 9]:
        y_pred1 = clf1.predict(X_test)

        def is_class(y_pred):
            for i in y_pred:
                if i not in list(set(list(clf1.classes_))):
                    return False
            return True

        assert len(y_pred1) == X_test.shape[0] and is_class(y_pred1) is True  # list(set(y_pred1)) == list(set(list(clf1.classes_)))
    else:
        assert clf1.sol_size in [4, 5]


check_estimator(DL85Classifier)

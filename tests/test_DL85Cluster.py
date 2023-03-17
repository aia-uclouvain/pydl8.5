from sklearn.utils.estimator_checks import check_estimator
from sklearn.model_selection import train_test_split
from pydl85.unsupervised import DL85Cluster
import numpy as np


def test_default_clustering():
    dataset = np.genfromtxt("../datasets/anneal.txt", delimiter=' ')
    X = dataset[:, 1:]
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=0)

    clf = DL85Cluster(max_depth=2, time_limit=600)
    clf.fit(X_train)
    assert str(clf.tree_) == "{'feat': 82, 'left': {'feat': 58, 'left': {'value': 0.39, 'error': 343.26001}, 'right': {'value': 0.51, 'error': 564.900024}}, 'right': {'feat': 72, 'left': {'value': 0.43, 'error': 424.809998}, 'right': {'value': 0.47, 'error': 299.459991}}, 'proba': None}"


def test_default_predictive_clustering():
    dataset = np.genfromtxt("../datasets/anneal.txt", delimiter=' ')
    X = dataset[:, 1:]
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=0)

    X_desc = X_train[:X_test.shape[0], :]
    X_err = X_test
    clf = DL85Cluster(max_depth=2, time_limit=600)
    clf.fit(X_desc, X_err)
    assert str(clf.tree_) == "{'feat': 46, 'left': {'feat': 16, 'left': {'value': 0.43, 'error': 462.079987}, 'right': {'value': 0.62, 'error': 35.57}}, 'right': {'feat': 72, 'left': {'value': 0.6, 'error': 11.13}, 'right': {'value': 0.0, 'error': 0.0}}, 'proba': None}"


# check_estimator(DL85Cluster())

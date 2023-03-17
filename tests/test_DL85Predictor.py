from sklearn.utils.estimator_checks import check_estimator
from sklearn.model_selection import train_test_split
from pydl85.predictors import DL85Predictor
import numpy as np


def test_user_sups_error_class():
    dataset = np.genfromtxt("../datasets/anneal.txt", delimiter=' ')
    X, y = dataset[:, 1:], dataset[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    def error(tids):
        classes, supports = np.unique(y_train.take(list(tids)), return_counts=True)
        maxindex = np.argmax(supports)
        return sum(supports) - supports[maxindex], classes[maxindex]

    clf = DL85Predictor(max_depth=2, error_function=error, time_limit=600)
    clf.fit(X_train)

    assert clf.error_ == 114.0
    assert round(clf.accuracy_, 4) == 0.8243
    assert str(clf.tree_) == "{'feat': 5, 'left': {'feat': 32, 'left': {'value': 1, 'error': 44}, 'right': {'value': 0, 'error': 2}}, 'right': {'feat': 46, 'left': {'value': 1, 'error': 68}, 'right': {'value': 0, 'error': 0}}, 'proba': None}"


# check_estimator(DL85Predictor())

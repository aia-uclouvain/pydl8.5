"""
======================
Default DL85Booster
======================

"""
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from pydl85 import DL85Booster, Boosting_Model

dataset = np.genfromtxt("../datasets/anneal.txt", delimiter=' ')
X, y = dataset[:, 1:], dataset[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

models = [Boosting_Model.MODEL_LP_DEMIRIZ, Boosting_Model.MODEL_LP_RATSCH, Boosting_Model.MODEL_QP_MDBOOST]
regulators = [15] * len(models)
model_names = ['MODEL_LP_DEMIRIZ', 'MODEL_LP_RATSCH', 'MODEL_QP_MDBOOST']
depth = 1

print("######################################################################\n"
      "#                     DL8.5 boosting classifier                      #\n"
      "######################################################################")
for i, model in enumerate(models):
    print("<<=== Optiboost ===>>")
    db_clf = DL85Booster(max_depth=depth, regulator=regulators[i], model=model)
    start = time.perf_counter()
    print("Model building...")
    db_clf.fit(X_train, y_train)
    duration = time.perf_counter() - start
    print("Model built. Duration of building =", round(duration, 4))
    print("Number of trees =", db_clf.n_estimators_)
    y_pred = db_clf.predict(X_test)
    print("Confusion Matrix below")
    print(confusion_matrix(y_test, y_pred))
    print("Accuracy DL85Booster +", model_names[i], "on training set =", round(accuracy_score(y_train, db_clf.predict(X_train)), 4))
    print("Accuracy DL85Booster +", model_names[i], "on test set =", round(accuracy_score(y_test, y_pred), 4), "\n")

    print("<<=== AdaBoost + CART ===>>")
    ab_clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=depth), n_estimators=db_clf.n_estimators_)
    start = time.perf_counter()
    print("Model building...")
    ab_clf.fit(X_train, y_train)
    duration = time.perf_counter() - start
    print("Model built. Duration of building =", round(duration, 4))
    y_pred = ab_clf.predict(X_test)
    print("Confusion Matrix below")
    print(confusion_matrix(y_test, y_pred))
    print("Accuracy AdaBoost on training set =", round(accuracy_score(y_train, ab_clf.predict(X_train)), 4))
    print("Accuracy AdaBoost on test set =", round(accuracy_score(y_test, y_pred), 4))
    print("\n\n")

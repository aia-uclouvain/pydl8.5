"""
======================
Default DL85Booster
======================

"""
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from pydl85 import DL85Booster, MODEL_LP_RATSCH, MODEL_LP_DEMIRIZ, MODEL_QP_MDBOOST
import time
import numpy as np
from sklearn.metrics import confusion_matrix

dataset = np.genfromtxt("../datasets/anneal.txt", delimiter=' ')
X, y = dataset[:, 1:], dataset[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

models = [MODEL_LP_DEMIRIZ, MODEL_LP_RATSCH, MODEL_QP_MDBOOST]
reguls = [15] * len(models)
model_names = ['MODEL_LP_DEMIRIZ', 'MODEL_LP_RATSCH', 'MODEL_QP_MDBOOST']
depth = 1

print("######################################################################\n"
      "#                     DL8.5 boosting classifier                      #\n"
      "######################################################################")
for i, mod in enumerate(models):
    print("<<=== Optiboost ===>>")
    clf = DL85Booster(max_depth=depth, regulator=reguls[i], model=mod)
    start = time.perf_counter()
    print("Model building...")
    clf.fit(X_train, y_train)
    duration = time.perf_counter() - start
    print("Model built. Duration of building =", round(duration, 4))
    print("Number of trees =", clf.n_estimators_)
    y_pred = clf.predict(X_test)
    print("Confusion Matrix below")
    print(confusion_matrix(y_test, y_pred))
    print("Accuracy DL85Booster +", model_names[i], "on training set =", round(accuracy_score(y_train, clf.predict(X_train)), 4))
    print("Accuracy DL85Booster +", model_names[i], "on test set =", round(accuracy_score(y_test, y_pred), 4), "\n")

    print("<<=== AdaBoost + CART ===>>")
    ab = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=depth), n_estimators=clf.n_estimators_)
    start = time.perf_counter()
    print("Model building...")
    ab.fit(X_train, y_train)
    duration = time.perf_counter() - start
    print("Model built. Duration of building =", round(duration, 4))
    y_pred = ab.predict(X_test)
    print("Confusion Matrix below")
    print(confusion_matrix(y_test, y_pred))
    print("Accuracy AdaBoost on training set =", round(accuracy_score(y_train, ab.predict(X_train)), 4))
    print("Accuracy AdaBoost on test set =", round(accuracy_score(y_test, y_pred), 4))
    print("\n\n")

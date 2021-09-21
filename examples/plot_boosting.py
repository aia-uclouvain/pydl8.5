"""
=================================
Default DL85Booster
=================================
"""
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from dl85 import DL85Booster, MODEL_QP_MDBOOST, MODEL_LP_DEMIRIZ
import time
import numpy as np
from sklearn.metrics import confusion_matrix

dataset = np.genfromtxt("../datasets/tic-tac-toe.txt", delimiter=' ')
X, y = dataset[:, 1:], dataset[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

depth = 1
# params = {'model': MODEL_QP_MDBOOST, 'regulator': 100, 'name': 'MDBoost'}
params = {'model': MODEL_LP_DEMIRIZ, 'regulator': .9, 'name': 'LPBoost'}

print("######################################################################\n"
      "#                     DL8.5 boosting classifier                      #\n"
      "######################################################################")
print("<<=== Optiboost ===>>")
clf = DL85Booster(max_depth=depth, model=params['model'], regulator=params['regulator'])
start = time.perf_counter()
print("Model building...")
clf.fit(X_train, y_train)
duration = time.perf_counter() - start
print("Model built. Duration of building =", round(duration, 4))
print("Number of trees =", clf.n_estimators_)
y_pred = clf.predict(X_test)
print("Confusion Matrix below")
print(confusion_matrix(y_test, y_pred))
print("Accuracy DL85Booster +", params['name'], "on training set =", round(accuracy_score(y_train, clf.predict(X_train)), 4))
print("Accuracy DL85Booster +", params['name'], "on test set =", round(accuracy_score(y_test, y_pred), 4), "\n")

print("<<=== AdaBoost + CART ===>>")
ab = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=depth), n_estimators=clf.n_estimators_)
start = time.perf_counter()
print("Model building...")
ab.fit(X, y)
duration = time.perf_counter() - start
print("Model built. Duration of building =", round(duration, 4))
print("Number of trees =", clf.n_estimators_)
y_pred = ab.predict(X_test)
print("Confusion Matrix below")
print(confusion_matrix(y_test, y_pred))
print("Accuracy AdaBoost on training set =", round(accuracy_score(y_train, ab.predict(X_train)), 4))
print("Accuracy AdaBoost on test set =", round(accuracy_score(y_test, y_pred), 4))
print("\n\n")

"""
======================
Default DL85Classifier
======================

"""
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import time
from dl85 import DL85Booster, DL85Classifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

dataset = np.genfromtxt("../datasets/anneal.txt", delimiter=' ')
X = dataset[:, 1:]
y = dataset[:, 0]
X = X.astype('int32')
y = y.astype('int32')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("######################################################################\n"
      "#                       DL8.5 default booster                        #\n"
      "######################################################################")

MAX_DEPTH = 2
MIN_SUP = 1
REGULATOR = 1
# REGULATOR = 0  # use value lower of equal to 0 for default value

clf = DL85Booster(max_depth=MAX_DEPTH, min_sup=MIN_SUP, time_limit=80, max_estimators=0, regulator=REGULATOR)
start = time.perf_counter()
print("Model building...")
clf.fit(X_train, y_train)
duration = time.perf_counter() - start
print("Model built. Duration of building =", round(duration, 4))
y_pred = clf.predict(X_test)
print("Confusion Matrix below")
print(confusion_matrix(y_test, y_pred))
print("Accuracy DL8.5Booster on training set =", round(clf.accuracy_, 4))
print("Accuracy DL8.5Booster on test set =", round(accuracy_score(y_test, y_pred), 4), "\n\n")


print("Random Forest Classifier")
rf = RandomForestClassifier(max_depth=MAX_DEPTH, min_samples_leaf=MIN_SUP, n_estimators=sum(w > 0 for w in clf.tree_weights))
start = time.perf_counter()
print("Model building...")
rf.fit(X_train, y_train)
duration = time.perf_counter() - start
print("Model built. Duration of building =", round(duration, 4))
y_pred = rf.predict(X_test)
print("Confusion Matrix below")
print(confusion_matrix(y_test, y_pred))
# print("Accuracy DL8.5 on training set =", round(rf.accuracy_, 4))
print("Accuracy RF on test set =", round(accuracy_score(y_test, y_pred), 4), "\n\n")


print("Gradient Boosting Classifier")
gb = GradientBoostingClassifier(max_depth=MAX_DEPTH, min_samples_leaf=MIN_SUP, n_estimators=sum(w > 0 for w in clf.tree_weights))
start = time.perf_counter()
print("Model building...")
gb.fit(X_train, y_train)
duration = time.perf_counter() - start
print("Model built. Duration of building =", round(duration, 4))
y_pred = gb.predict(X_test)
print("Confusion Matrix below")
print(confusion_matrix(y_test, y_pred))
# print("Accuracy DL8.5 on training set =", round(rf.accuracy_, 4))
print("Accuracy GB on test set =", round(accuracy_score(y_test, y_pred), 4), "\n\n")


print("DL8.5 Classifier")
clf = DL85Classifier(max_depth=MAX_DEPTH, min_sup=MIN_SUP, time_limit=80)
start = time.perf_counter()
print("Model building...")
clf.fit(X_train, y_train)
duration = time.perf_counter() - start
print("Model built. Duration of building =", round(duration, 4))
y_pred = clf.predict(X_test)
print("Confusion Matrix below")
print(confusion_matrix(y_test, y_pred))
print("Accuracy DL8.5 on training set =", round(clf.accuracy_, 4))
print("Accuracy DL8.5 on test set =", round(accuracy_score(y_test, y_pred), 4), "\n\n")

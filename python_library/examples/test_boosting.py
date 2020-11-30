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
from sklearn.model_selection import GridSearchCV
import time
from dl85 import DL85Booster, DL85Classifier, BOOST_SVM2, BOOST_SVM1
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

dataset = np.genfromtxt("../datasets/paper.txt", delimiter=' ')
X = dataset[:, 1:]
y = dataset[:, 0]
X = X.astype('int32')
y = y.astype('int32')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("######################################################################\n"
      "#                       DL8.5 default booster                        #\n"
      "######################################################################")

MAX_DEPTH = 1
MIN_SUP = 1
REGULATOR = 1
# REGULATOR = 0  # use value lower of equal to 0 for default value

print("DL8.5")
clf = DL85Classifier(max_depth=MAX_DEPTH, min_sup=MIN_SUP, time_limit=80, print_output=True)
start = time.perf_counter()
print("Model building...")
clf.fit(X, y)
duration = time.perf_counter() - start
print("Model built. Duration of building =", round(duration, 4))
y_pred = clf.predict(X)
print(y_pred)
print("Confusion Matrix below")
print(confusion_matrix(y, y_pred))
print("Accuracy DL8.5 on training set =", round(clf.accuracy_, 4))
print("Accuracy DL8.5 on test set =", round(accuracy_score(y, y_pred), 4), "\n\n")


print("LPBoost + DL8.5")
clf = DL85Booster(max_depth=MAX_DEPTH, min_sup=MIN_SUP, time_limit=80, max_estimators=0, regulator=REGULATOR, model=BOOST_SVM1)
start = time.perf_counter()
print("Model building...")
clf.fit(X, y)
duration = time.perf_counter() - start
print("Model built. Duration of building =", round(duration, 4))
y_pred = clf.predict(X)
print(y_pred)
print("Confusion Matrix below")
print(confusion_matrix(y, y_pred))
print("Accuracy DL8.5Booster on training set =", round(clf.accuracy_, 4))
print("Accuracy DL8.5Booster on test set =", round(accuracy_score(y, y_pred), 4))
print(clf.n_estimators_)
print(clf.n_iterations_)
for i, j in zip(clf.estimator_weights_, clf.estimators_):
      print(i, j.tree_)
print("\n\n")


print("AdaBoost + CART")
ab = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=MAX_DEPTH, min_samples_leaf=MIN_SUP), n_estimators=2)
start = time.perf_counter()
print("Model building...")
ab.fit(X, y)
duration = time.perf_counter() - start
print("Model built. Duration of building =", round(duration, 4))
y_pred = ab.predict(X)
print("Confusion Matrix below")
print(confusion_matrix(y, y_pred))
# print("Accuracy AdaBoost on training set =", round(accuracy_score(y, ab.predict(X)), 4))
print("Accuracy AdaBoost on test set =", round(accuracy_score(y, y_pred), 4))
print(len(ab.estimator_weights_))
print(ab.estimator_weights_)
print("\n\n")

# print("LPBoost + DL8.5")
# clf = DL85Booster(max_depth=MAX_DEPTH, min_sup=MIN_SUP, time_limit=80, max_estimators=0, regulator=REGULATOR)
# start = time.perf_counter()
# print("Model building...")
# clf.fit(X_train, y_train)
# duration = time.perf_counter() - start
# print("Model built. Duration of building =", round(duration, 4))
# y_pred = clf.predict(X_test)
# print("Confusion Matrix below")
# print(confusion_matrix(y_test, y_pred))
# print("Accuracy DL8.5Booster on training set =", round(clf.accuracy_, 4))
# print("Accuracy DL8.5Booster on test set =", round(accuracy_score(y_test, y_pred), 4), "\n\n")
#
#
# print("LPBoost + CART")
# clf1 = DL85Booster(base_estimator=DecisionTreeClassifier(max_depth=MAX_DEPTH, min_samples_leaf=MIN_SUP), time_limit=80, max_estimators=0, regulator=REGULATOR)
# start = time.perf_counter()
# print("Model building...")
# clf1.fit(X_train, y_train)
# duration = time.perf_counter() - start
# print("Model built. Duration of building =", round(duration, 4))
# y_pred = clf1.predict(X_test)
# print("Confusion Matrix below")
# print(confusion_matrix(y_test, y_pred))
# print("Accuracy CartBooster on training set =", round(clf1.accuracy_, 4))
# print("Accuracy CartBooster on test set =", round(accuracy_score(y_test, y_pred), 4), "\n\n")
#
#
# print("AdaBoost + CART")
# ab = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=MAX_DEPTH, min_samples_leaf=MIN_SUP), n_estimators=clf.n_estimators_)
# start = time.perf_counter()
# print("Model building...")
# ab.fit(X_train, y_train)
# duration = time.perf_counter() - start
# print("Model built. Duration of building =", round(duration, 4))
# y_pred = ab.predict(X_test)
# print("Confusion Matrix below")
# print(confusion_matrix(y_test, y_pred))
# print("Accuracy AdaBoost on training set =", round(accuracy_score(y_train, ab.predict(X_train)), 4))
# print("Accuracy AdaBoost on test set =", round(accuracy_score(y_test, y_pred), 4), "\n\n")
#
#
# print("AdaBoost + DL8.5")
# abd = AdaBoostClassifier(base_estimator=DL85Classifier(max_depth=MAX_DEPTH, min_sup=MIN_SUP), algorithm="SAMME", n_estimators=clf.n_estimators_)
# start = time.perf_counter()
# print("Model building...")
# abd.fit(X_train, y_train)
# duration = time.perf_counter() - start
# print("Model built. Duration of building =", round(duration, 4))
# y_pred = abd.predict(X_test)
# print("Confusion Matrix below")
# print(confusion_matrix(y_test, y_pred))
# print("Accuracy AdaBoost+DL8.5 on training set =", round(accuracy_score(y_train, abd.predict(X_train)), 4))
# print("Accuracy AdaBoost+DL8.5  on test set =", round(accuracy_score(y_test, y_pred), 4), "\n\n")
#
#
# print("Gradient Boosting")
# gb = GradientBoostingClassifier(max_depth=MAX_DEPTH, min_samples_leaf=MIN_SUP, n_estimators=clf.n_estimators_)
# start = time.perf_counter()
# print("Model building...")
# gb.fit(X_train, y_train)
# duration = time.perf_counter() - start
# print("Model built. Duration of building =", round(duration, 4))
# y_pred = gb.predict(X_test)
# print("Confusion Matrix below")
# print(confusion_matrix(y_test, y_pred))
# print("Accuracy GB on test set =", round(accuracy_score(y_train, gb.predict(X_train)), 4))
# print("Accuracy GB on test set =", round(accuracy_score(y_test, y_pred), 4), "\n\n")
#
#
# print("Random Forest")
# rf = RandomForestClassifier(max_depth=MAX_DEPTH, min_samples_leaf=MIN_SUP, n_estimators=clf.n_estimators_)
# start = time.perf_counter()
# print("Model building...")
# rf.fit(X_train, y_train)
# duration = time.perf_counter() - start
# print("Model built. Duration of building =", round(duration, 4))
# y_pred = rf.predict(X_test)
# print("Confusion Matrix below")
# print(confusion_matrix(y_test, y_pred))
# print("Accuracy RF on training set =", round(accuracy_score(y_train, rf.predict(X_train)), 4))
# print("Accuracy RF on test set =", round(accuracy_score(y_test, y_pred), 4), "\n\n")
#
#
# print("DL8.5 Classifier")
# clf = DL85Classifier(max_depth=MAX_DEPTH, min_sup=MIN_SUP, time_limit=80)
# start = time.perf_counter()
# print("Model building...")
# clf.fit(X_train, y_train)
# duration = time.perf_counter() - start
# print("Model built. Duration of building =", round(duration, 4))
# y_pred = clf.predict(X_test)
# print("Confusion Matrix below")
# print(confusion_matrix(y_test, y_pred))
# print("Accuracy DL8.5 on training set =", round(clf.accuracy_, 4))
# print("Accuracy DL8.5 on test set =", round(accuracy_score(y_test, y_pred), 4))

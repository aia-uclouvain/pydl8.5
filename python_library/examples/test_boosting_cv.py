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
from dl85 import DL85Booster, DL85Classifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

dataset = np.genfromtxt("../datasets/anneal.txt", delimiter=' ')
X = dataset[:, 1:]
y = dataset[:, 0]
X = X.astype('int32')
y = y.astype('int32')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("######################################################################\n"
      "#                       DL8.5 default booster                        #\n"
      "######################################################################")

N_FOLDS = 5
MAX_DEPTH = 1
MIN_SUP = 1
REGULATOR = 0.5
# REGULATOR = 0  # use value lower of equal to 0 for default value

print("LPBoost + DL8.5")
clf = DL85Booster(max_depth=MAX_DEPTH, min_sup=MIN_SUP, time_limit=80, max_estimators=0, regulator=REGULATOR)
start = time.perf_counter()
print("Model building...")
scores = cross_val_score(clf, X, y, cv=N_FOLDS)
duration = time.perf_counter() - start
print("Model built. Duration of building =", round(duration, 4))
print("Average accuracy DL8.5Booster on test set =", round(np.mean(scores), 4), "\n\n")


print("LPBoost + DL8.5 manual cv")
kf = KFold(n_splits=5, random_state=42, shuffle=True)
training_accuracies = []
test_accuracies = []
max_trees = []
start = time.perf_counter()
print("Model building...")
for train_index, test_index in kf.split(X):
    data_train = X[train_index]
    target_train = y[train_index]
    data_test = X[test_index]
    target_test = y[test_index]
    clf = DL85Booster(max_depth=MAX_DEPTH, min_sup=MIN_SUP, time_limit=80, max_estimators=0, regulator=REGULATOR)
    clf.fit(data_train, target_train)
    max_trees.append(clf.n_estimators_)
    preds = clf.predict(data_test)
    training_accuracies.append(clf.accuracy_)
    test_accuracies.append(accuracy_score(target_test, preds))
duration = time.perf_counter() - start
print("Model built. Duration of building =", round(duration, 4))
print("Average accuracy on training set =", round(np.mean(training_accuracies), 4))
print("Average accuracy on test set =", round(np.mean(test_accuracies), 4), "\n\n\n")


print("LPBoost + CART")
clf1 = DL85Booster(base_estimator=DecisionTreeClassifier(max_depth=MAX_DEPTH, min_samples_leaf=MIN_SUP), time_limit=80,
                   max_estimators=0, regulator=REGULATOR)
start = time.perf_counter()
print("Model building...")
scores = cross_val_score(clf1, X, y, cv=N_FOLDS)
duration = time.perf_counter() - start
print("Model built. Duration of building =", round(duration, 4))
print("Average accuracy CartBooster on test set =", round(np.mean(scores), 4), "\n\n")

print("AdaBoost + CART")
ab = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=MAX_DEPTH, min_samples_leaf=MIN_SUP),
                        n_estimators=max(max_trees))
start = time.perf_counter()
print("Model building...")
scores = cross_val_score(ab, X, y, cv=N_FOLDS)
duration = time.perf_counter() - start
print("Model built. Duration of building =", round(duration, 4))
print("Average accuracy AdaBoost on test set =", round(np.mean(scores), 4), "\n\n")

print("AdaBoost + DL8.5")
abd = AdaBoostClassifier(base_estimator=DL85Classifier(max_depth=MAX_DEPTH, min_sup=MIN_SUP), algorithm="SAMME",
                         n_estimators=max(max_trees))
start = time.perf_counter()
print("Model building...")
scores = cross_val_score(abd, X, y, cv=N_FOLDS)
duration = time.perf_counter() - start
print("Model built. Duration of building =", round(duration, 4))
print("Accuracy AdaBoost+DL8.5  on test set =", round(np.mean(scores), 4), "\n\n")

print("Gradient Boosting")
gb = GradientBoostingClassifier(max_depth=MAX_DEPTH, min_samples_leaf=MIN_SUP, n_estimators=max(max_trees))
start = time.perf_counter()
print("Model building...")
scores = cross_val_score(gb, X, y, cv=N_FOLDS)
duration = time.perf_counter() - start
print("Model built. Duration of building =", round(duration, 4))
print("Average accuracy GB on test set =", round(np.mean(scores), 4), "\n\n")

print("Random Forest")
rf = RandomForestClassifier(max_depth=MAX_DEPTH, min_samples_leaf=MIN_SUP, n_estimators=max(max_trees))
start = time.perf_counter()
print("Model building...")
scores = cross_val_score(rf, X, y, cv=N_FOLDS)
duration = time.perf_counter() - start
print("Model built. Duration of building =", round(duration, 4))
print("Average accuracy RF on test set =", round(np.mean(scores), 4), "\n\n")

print("DL8.5 Classifier")
clf2 = DL85Classifier(max_depth=MAX_DEPTH, min_sup=MIN_SUP, time_limit=80)
start = time.perf_counter()
print("Model building...")
scores = cross_val_score(clf2, X, y, cv=N_FOLDS)
duration = time.perf_counter() - start
print("Model built. Duration of building =", round(duration, 4))
print("Average accuracy DL8.5 on test set =", round(np.mean(scores), 4))

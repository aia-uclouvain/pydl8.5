"""
======================
Default DL85Classifier
======================

"""
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import time
from dl85 import DL85Booster, DL85Classifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

filename = "australian-credit"
dataset = np.genfromtxt("../datasets/" + filename + ".txt", delimiter=' ')
X = dataset[:, 1:]
y = dataset[:, 0]
X = X.astype('int32')
y = y.astype('int32')

print("Dataset :", filename)
print("size of 0 :", y.tolist().count(0), "size of 1 :", y.tolist().count(1))

N_FOLDS, MAX_DEPTH, MIN_SUP, MAX_TREES = 5, 2, 1, 10

X_trains, y_trains = [], []
X_tests, y_tests = [], []
# kf = KFold(n_splits=N_FOLDS, random_state=42, shuffle=True)
kf = StratifiedKFold(n_splits=N_FOLDS)
for train_index, test_index in kf.split(X, y):
    X_trains.append(X[train_index])
    y_trains.append(y[train_index])
    X_tests.append(X[test_index])
    y_tests.append(y[test_index])

parameters = {'regulator': np.linspace(0.1, 1, 10).tolist()}

print("######################################################################\n"
      "#                                START                               #\n"
      "######################################################################")

print("DL8.5")
train_acc, test_acc = [], []
n_trees, fps, fns = [], [], []
start = time.perf_counter()
print("Model building...")
for k in range(N_FOLDS):
    X_train, y_train, X_test, y_test = X_trains[k], y_trains[k], X_tests[k], y_tests[k]
    clf = DL85Classifier(max_depth=MAX_DEPTH, min_sup=MIN_SUP, time_limit=80)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    train_acc.append(accuracy_score(y_train, clf.predict(X_train)))
    test_acc.append(accuracy_score(y_test, preds))
    n_trees.append(1)
    fps.append(len([i for i in [j for j, val in enumerate(preds) if val == 1] if y_test[i] != 1]))
    fns.append(len([i for i in [j for j, val in enumerate(preds) if val == 0] if y_test[i] != 0]))
duration = time.perf_counter() - start
print("Model built. Avg duration of building =", round(duration / N_FOLDS, 4))
print("Avg number of trees =", round(float(np.mean(n_trees)), 4))
print("Avg accuracy on training set =", round(float(np.mean(train_acc)), 4))
print("Avg accuracy on test set =", round(float(np.mean(test_acc)), 4))
print("sum false positives =", sum(fps))
print("sum false negatives =", sum(fns), "\n\n\n")


print("LPBoost + DL8.5")
print("Search for the best regulator using grid search...", MAX_TREES)
# each regulator is tested without constraint on trees numbers
gd_sr = GridSearchCV(estimator=DL85Booster(max_depth=MAX_DEPTH, min_sup=MIN_SUP, time_limit=80, max_estimators=MAX_TREES),
                     param_grid=parameters, scoring='accuracy', cv=N_FOLDS, n_jobs=1)
gd_sr.fit(X, y)
print("Done")
train_acc, test_acc = [], []
n_trees, fps, fns = [], [], []
start = time.perf_counter()
print("Model building...")
for k in range(N_FOLDS):
    X_train, y_train, X_test, y_test = X_trains[k], y_trains[k], X_tests[k], y_tests[k]
    clf = DL85Booster(max_depth=MAX_DEPTH, min_sup=MIN_SUP, time_limit=80, regulator=gd_sr.best_params_["regulator"],
                      max_estimators=MAX_TREES)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    train_acc.append(clf.accuracy_)
    test_acc.append(accuracy_score(y_test, preds))
    n_trees.append(clf.n_estimators_)
    fps.append(len([i for i in [j for j, val in enumerate(preds) if val == 1] if y_test[i] != 1]))
    fns.append(len([i for i in [j for j, val in enumerate(preds) if val == 0] if y_test[i] != 0]))
duration = time.perf_counter() - start
max_estimators = int(sum(n_trees)/len(n_trees))
print("Model built. Avg duration of building =", round(duration / N_FOLDS, 4))
print("Avg number of trees =", round(float(np.mean(n_trees)), 4))
print("Avg accuracy on training set =", round(float(np.mean(train_acc)), 4))
print("Avg accuracy on test set =", round(float(np.mean(test_acc)), 4))
print("sum false positives =", sum(fps))
print("sum false negatives =", sum(fns), "\n\n\n")


print("LPBoost + CART")
print("Search for the best regulator using grid search...")
# each regulator is tested with max_trees fixed by LPBoost+DL8.5
gd_sr = GridSearchCV(estimator=DL85Booster(base_estimator=DecisionTreeClassifier(max_depth=MAX_DEPTH,
                     min_samples_leaf=MIN_SUP), time_limit=80, max_estimators=max_estimators), param_grid=parameters,
                     scoring='accuracy', cv=N_FOLDS, n_jobs=-1)
gd_sr.fit(X, y)
print("Done")
train_acc, test_acc = [], []
n_trees, fps, fns = [], [], []
start = time.perf_counter()
print("Model building...")
for k in range(N_FOLDS):
    X_train, y_train, X_test, y_test = X_trains[k], y_trains[k], X_tests[k], y_tests[k]
    clf = DL85Booster(base_estimator=DecisionTreeClassifier(max_depth=MAX_DEPTH, min_samples_leaf=MIN_SUP),
                      max_estimators=max_estimators, time_limit=80, regulator=gd_sr.best_params_["regulator"])
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    train_acc.append(clf.accuracy_)
    test_acc.append(accuracy_score(y_test, preds))
    n_trees.append(clf.n_estimators_)
    fps.append(len([i for i in [j for j, val in enumerate(preds) if val == 1] if y_test[i] != 1]))
    fns.append(len([i for i in [j for j, val in enumerate(preds) if val == 0] if y_test[i] != 0]))
duration = time.perf_counter() - start
print("Model built. Avg duration of building =", round(duration / N_FOLDS, 4))
print("Avg number of trees =", round(float(np.mean(n_trees)), 4))
print("Avg accuracy on training set =", round(float(np.mean(train_acc)), 4))
print("Avg accuracy on test set =", round(float(np.mean(test_acc)), 4))
print("sum false positives =", sum(fps))
print("sum false negatives =", sum(fns), "\n\n\n")


print("Adaboost + CART")
train_acc, test_acc = [], []
n_trees, fps, fns = [], [], []
start = time.perf_counter()
print("Model building...")
for k in range(N_FOLDS):
    X_train, y_train, X_test, y_test = X_trains[k], y_trains[k], X_tests[k], y_tests[k]
    clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=MAX_DEPTH, min_samples_leaf=MIN_SUP),
                             n_estimators=max_estimators)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    train_acc.append(accuracy_score(y_train, clf.predict(X_train)))
    test_acc.append(accuracy_score(y_test, preds))
    n_trees.append(len(clf.estimators_))
    fps.append(len([i for i in [j for j, val in enumerate(preds) if val == 1] if y_test[i] != 1]))
    fns.append(len([i for i in [j for j, val in enumerate(preds) if val == 0] if y_test[i] != 0]))
duration = time.perf_counter() - start
print("Model built. Avg duration of building =", round(duration / N_FOLDS, 4))
print("Avg number of trees =", round(float(np.mean(n_trees)), 4))
print("Avg accuracy on training set =", round(float(np.mean(train_acc)), 4))
print("Avg accuracy on test set =", round(float(np.mean(test_acc)), 4))
print("sum false positives =", sum(fps))
print("sum false negatives =", sum(fns), "\n\n\n")


print("Gradient Boosting")
train_acc, test_acc = [], []
n_trees, fps, fns = [], [], []
start = time.perf_counter()
print("Model building...")
for k in range(N_FOLDS):
    X_train, y_train, X_test, y_test = X_trains[k], y_trains[k], X_tests[k], y_tests[k]
    clf = GradientBoostingClassifier(max_depth=MAX_DEPTH, min_samples_leaf=MIN_SUP, n_estimators=max_estimators)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    train_acc.append(accuracy_score(y_train, clf.predict(X_train)))
    test_acc.append(accuracy_score(y_test, preds))
    n_trees.append(clf.n_estimators_)
    fps.append(len([i for i in [j for j, val in enumerate(preds) if val == 1] if y_test[i] != 1]))
    fns.append(len([i for i in [j for j, val in enumerate(preds) if val == 0] if y_test[i] != 0]))
duration = time.perf_counter() - start
print("Model built. Avg duration of building =", round(duration / N_FOLDS, 4))
print("Avg number of trees =", round(float(np.mean(n_trees)), 4))
print("Avg accuracy on training set =", round(float(np.mean(train_acc)), 4))
print("Avg accuracy on test set =", round(float(np.mean(test_acc)), 4))
print("sum false positives =", sum(fps))
print("sum false negatives =", sum(fns), "\n\n\n")


print("Random Forest")
train_acc, test_acc = [], []
n_trees, fps, fns = [], [], []
start = time.perf_counter()
print("Model building...")
for k in range(N_FOLDS):
    X_train, y_train, X_test, y_test = X_trains[k], y_trains[k], X_tests[k], y_tests[k]
    clf = RandomForestClassifier(max_depth=MAX_DEPTH, min_samples_leaf=MIN_SUP, n_estimators=max_estimators)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    train_acc.append(accuracy_score(y_train, clf.predict(X_train)))
    test_acc.append(accuracy_score(y_test, preds))
    n_trees.append(len(clf.estimators_))
    fps.append(len([i for i in [j for j, val in enumerate(preds) if val == 1] if y_test[i] != 1]))
    fns.append(len([i for i in [j for j, val in enumerate(preds) if val == 0] if y_test[i] != 0]))
duration = time.perf_counter() - start
print("Model built. Avg duration of building =", round(duration / N_FOLDS, 4))
print("Avg number of trees =", round(float(np.mean(n_trees)), 4))
print("Avg accuracy on training set =", round(float(np.mean(train_acc)), 4))
print("Avg accuracy on test set =", round(float(np.mean(test_acc)), 4))
print("sum false positives =", sum(fps))
print("sum false negatives =", sum(fns), "\n\n\n")


print("Adaboost + DL8.5")
train_acc, test_acc = [], []
n_trees, fps, fns = [], [], []
start = time.perf_counter()
print("Model building...")
for k in range(N_FOLDS):
    X_train, y_train, X_test, y_test = X_trains[k], y_trains[k], X_tests[k], y_tests[k]
    clf = AdaBoostClassifier(base_estimator=DL85Classifier(max_depth=MAX_DEPTH, min_sup=MIN_SUP), algorithm="SAMME",
                             n_estimators=max_estimators)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    train_acc.append(accuracy_score(y_train, clf.predict(X_train)))
    test_acc.append(accuracy_score(y_test, preds))
    n_trees.append(len(clf.estimators_))
    fps.append(len([i for i in [j for j, val in enumerate(preds) if val == 1] if y_test[i] != 1]))
    fns.append(len([i for i in [j for j, val in enumerate(preds) if val == 0] if y_test[i] != 0]))
duration = time.perf_counter() - start
print("Model built. Avg duration of building =", round(duration / N_FOLDS, 4))
print("Avg number of trees =", round(float(np.mean(n_trees)), 4))
print("Avg accuracy on training set =", round(float(np.mean(train_acc)), 4))
print("Avg accuracy on test set =", round(float(np.mean(test_acc)), 4))
print("sum false positives =", sum(fps))
print("sum false negatives =", sum(fns))

from sklearn.svm import SVC
from sklearn.datasets import load_iris, load_digits, fetch_openml
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelBinarizer, KBinsDiscretizer, Binarizer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from dl85 import DL85Booster, DL85Classifier, BOOST_SVM2
import xgboost as xgb
import time
import numpy as np
import pandas as pd
import sys


depth, time_limit = 2, 900


file = open("../output/nist_" + str(depth) + ".txt", "w")


train = np.genfromtxt("../datasets/mnist_train.csv", delimiter=",")
test = np.genfromtxt("../datasets/mnist_test.csv", delimiter=",")
# split features and target
X_train, y_train = train[:, 1:], train[:, 0]
X_test, y_test = test[:, 1:], test[:, 0]
# select a slice of training data
X_train, _, y_train, _ = train_test_split(X_train, y_train, stratify=y_train, train_size=20000)
# convert values to int
X_train, y_train = X_train.astype('int32'), y_train.astype('int32')
X_test, y_test = X_test.astype('int32'), y_test.astype('int32')
# binarize the features
enc = Binarizer(threshold=10)
X_train = enc.fit_transform(X_train)
X_test = enc.fit_transform(X_test)
# create 2 classes even or odd
biner = lambda x: x % 2
y_train = biner(y_train)
y_test = biner(y_test)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)
print(set(y_train), set(y_test))


print("mnist")


print()
print("dl85 depth=1")
clf = DL85Classifier(max_depth=1, min_sup=1, time_limit=time_limit)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("train_acc:", accuracy_score(y_train, clf.predict(X_train)), "test_acc:", accuracy_score(y_test, y_pred))


print()
print("cart depth=1")
clf = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("train_acc:", accuracy_score(y_train, clf.predict(X_train)), "test_acc:", accuracy_score(y_test, y_pred))


print()
print("dl85 depth=2")
clf = DL85Classifier(max_depth=depth, min_sup=1, time_limit=time_limit)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("train_acc:", accuracy_score(y_train, clf.predict(X_train)), "test_acc:", accuracy_score(y_test, y_pred))


print()
print("cart depth=2")
clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("train_acc:", accuracy_score(y_train, clf.predict(X_train)), "test_acc:", accuracy_score(y_test, y_pred))
# sys.exit(0)


print()
print("lpdl85 2")
clf = DL85Booster(max_depth=depth, time_limit=0, model=BOOST_SVM2, regulator=1000000, quiet=False)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("train_acc:", accuracy_score(y_train, clf.predict(X_train)), "test_acc:", accuracy_score(y_test, y_pred))
sys.exit(0)


print()
print("lpdl8 d" + str(depth))
parameters = {'regulator': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 10, 100, 1000, 10000, 100000, 1000000]}
grid = GridSearchCV(estimator=DL85Booster(max_depth=depth, min_sup=1, time_limit=time_limit, max_estimators=0, model=BOOST_SVM2),
                    param_grid=parameters, scoring='accuracy', cv=4, n_jobs=-1, verbose=10)
grid.fit(X_train, y_train)
max_trees = grid.best_estimator_.n_estimators_
print("max_trees:", max_trees)
y_pred = grid.predict(X_test)
train_acc, test_acc = accuracy_score(y_train, grid.predict(X_train)), accuracy_score(y_test, y_pred)
print("train_acc:", train_acc, "test_acc:", test_acc, "n_trees", max_trees, "regulator", grid.best_params_['regulator'])


print()
print("AdaBoost non-linear max_trees")
clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=depth), n_estimators=max_trees)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
train_acc, test_acc = accuracy_score(y_train, clf.predict(X_train)), accuracy_score(y_test, y_pred)
print("train_acc:", train_acc, "test_acc:", test_acc, "n_trees", len(clf.estimators_))


print()
print("AdaBoost non-linear")
param_grid_ada = {'n_estimators': [50, 100, 1000]}
grid = GridSearchCV(estimator=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=depth)), param_grid=param_grid_ada, refit=True, verbose=10, cv=4, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)
print(grid.best_estimator_)
y_pred = grid.predict(X_test)
train_acc, test_acc = accuracy_score(y_train, grid.predict(X_train)), accuracy_score(y_test, y_pred)
print("train_acc:", train_acc, "test_acc:", test_acc, "n_trees", len(grid.best_estimator_.estimators_))


print()
print("XGBoost non-linear maxtrees")
param_grid_xgb = {'objective': ['binary:logistic', 'binary:logitraw']}
grid = GridSearchCV(estimator=xgb.XGBClassifier(n_estimators=max_trees, max_depth=depth), param_grid=param_grid_xgb, refit=True, verbose=10, cv=4, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)
print(grid.best_estimator_)
y_pred = grid.predict(X_test)
train_acc, test_acc = accuracy_score(y_train, grid.predict(X_train)), accuracy_score(y_test, y_pred)
print("train_acc:", train_acc, "test_acc:", test_acc, "n_trees", grid.best_estimator_.n_estimators, "objective", grid.best_params_['objective'])


print()
print("XGBoost non-linear")
param_grid_xgb = {'n_estimators': [50, 100, 1000], 'objective': ['binary:logistic', 'binary:logitraw']}
grid = GridSearchCV(estimator=xgb.XGBClassifier(max_depth=depth), param_grid=param_grid_xgb, refit=True, verbose=10, cv=4, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)
print(grid.best_estimator_)
y_pred = grid.predict(X_test)
train_acc, test_acc = accuracy_score(y_train, grid.predict(X_train)), accuracy_score(y_test, y_pred)
print("train_acc:", train_acc, "test_acc:", test_acc, "n_trees", grid.best_estimator_.n_estimators, "objective", grid.best_params_['objective'])


print()
print("RF non-linear maxtrees")
clf = RandomForestClassifier(max_depth=depth, n_estimators=max_trees)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
train_acc, test_acc = accuracy_score(y_train, clf.predict(X_train)), accuracy_score(y_test, y_pred)
print("train_acc:", train_acc, "test_acc:", test_acc, "n_trees", len(clf.estimators_))


print()
print("RF non-linear")
param_grid_rf = {'n_estimators': [50, 100, 1000]}
grid = GridSearchCV(estimator=RandomForestClassifier(max_depth=depth), param_grid=param_grid_rf, refit=True, verbose=10, cv=4, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)
print(grid.best_estimator_)
y_pred = grid.predict(X_test)
train_acc, test_acc = accuracy_score(y_train, grid.predict(X_train)), accuracy_score(y_test, y_pred)
print("train_acc:", train_acc, "test_acc:", test_acc, "n_trees", len(grid.best_estimator_.estimators_))


print()
print("GB non-linear maxtrees")
clf = GradientBoostingClassifier(max_depth=depth, n_estimators=max_trees)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
train_acc, test_acc = accuracy_score(y_train, clf.predict(X_train)), accuracy_score(y_test, y_pred)
print("train_acc:", train_acc, "test_acc:", test_acc, "n_trees", clf.n_estimators)


print()
print("GB non-linear")
param_grid_gb = {'n_estimators': [50, 100, 1000]}
grid = GridSearchCV(estimator=GradientBoostingClassifier(max_depth=depth), param_grid=param_grid_gb, refit=True, verbose=10, cv=4, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)
print(grid.best_estimator_)
y_pred = grid.predict(X_test)
train_acc, test_acc = accuracy_score(y_train, grid.predict(X_train)), accuracy_score(y_test, y_pred)
print("train_acc:", train_acc, "test_acc:", test_acc, "n_trees", grid.best_estimator_.n_estimators)



#print("SVM linear")
#param_grid_linear = {'C': [0.1, 1, 10, 100, 1000], 'kernel': ['linear']}
#grid = GridSearchCV(SVC(max_iter=1000), param_grid_linear, refit=True, verbose=10, cv=4, scoring='accuracy', n_jobs=-1)
#grid.fit(X_train, y_train)
#print(grid.best_estimator_)
#y_pred = grid.predict(X_test)
#print("train_fold:", accuracy_score(y_train, grid.predict(X_train)), "test_fold:", accuracy_score(y_test, y_pred))
## print(grid.cv_results_)


#print()
#print("SVM non-linear")
#best_train, best_test = [], []
#param_grid_poly = {'C': [0.1, 1, 10, 100, 1000], 'kernel': ['poly', 'rbf'], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'degree': [2, 3, 4]}
#grid = GridSearchCV(SVC(max_iter=1000), param_grid_poly, refit=True, verbose=10, cv=4, scoring='accuracy', n_jobs=-1)
#grid.fit(X_train, y_train)
#print(grid.best_estimator_)
#y_pred = grid.predict(X_test)
#print("train_fold:", accuracy_score(y_train, grid.predict(X_train)), "test_fold:", accuracy_score(y_test, y_pred))
## print(grid.cv_results_)

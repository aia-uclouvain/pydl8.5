from sklearn.svm import SVC
from sklearn.datasets import load_iris, load_digits, fetch_openml
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelBinarizer, KBinsDiscretizer, Binarizer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.utils import check_random_state
from dl85 import DL85Booster, DL85Classifier, BOOST_SVM2
import xgboost as xgb
import time
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from sklearn import tree
import graphviz
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from geopy.extra.rate_limiter import RateLimiter

depth, time_limit, N_FOLDS = 2, 900, 5


# split features and target
# dataset = np.genfromtxt("../../datasets/continuous/car_evaluation.csv", delimiter=";", skip_header=1)
# X, y = dataset[:, :-1], dataset[:, -1]
# enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
# X = enc.fit_transform(X)
dataset = np.genfromtxt("../datasets/kr-vs-kp.txt", delimiter=" ")
X, y = dataset[:, 1:], dataset[:, 0]
X, y = X.astype('int32'), y.astype('int32')
print(X.shape, y.shape)
print(set(y))


kf = StratifiedKFold(n_splits=5)
X_trains, X_tests, y_trains, y_tests = [], [], [], []
for train_index, test_index in kf.split(X, y):
    X_trains.append(X[train_index])
    y_trains.append(y[train_index])
    X_tests.append(X[test_index])
    y_tests.append(y[test_index])

print()
print("dl85 depth=1")
clf_results = cross_validate(estimator=DL85Classifier(max_depth=1, min_sup=1, time_limit=time_limit), X=X, y=y, scoring='accuracy',
                             cv=N_FOLDS, n_jobs=-1, verbose=10, return_train_score=True, return_estimator=True, error_score=np.nan)
print("train_accs:", clf_results['train_score'], np.mean(clf_results['train_score']))
print("test_fold:", clf_results['test_score'], np.mean(clf_results['test_score']))


print()
print("cart depth=1")
clf_results = cross_validate(estimator=DecisionTreeClassifier(max_depth=1, min_samples_leaf=1), X=X, y=y, scoring='accuracy',
                             cv=N_FOLDS, n_jobs=-1, verbose=10, return_train_score=True, return_estimator=True, error_score=np.nan)
print("train_accs:", clf_results['train_score'], np.mean(clf_results['train_score']))
print("test_fold:", clf_results['test_score'], np.mean(clf_results['test_score']))


print()
print("dl85 depth=2")
clf_results = cross_validate(estimator=DL85Classifier(max_depth=depth, min_sup=1, time_limit=time_limit), X=X, y=y, scoring='accuracy',
                             cv=N_FOLDS, n_jobs=-1, verbose=10, return_train_score=True, return_estimator=True, error_score=np.nan)
print("train_accs:", clf_results['train_score'], np.mean(clf_results['train_score']))
print("test_fold:", clf_results['test_score'], np.mean(clf_results['test_score']))
#
# for i, clf in enumerate(clf_results['estimator']):
#     dot = clf.export_graphviz()
#     graph = graphviz.Source(dot, format="png")
#     graph.render('dl85_trees_' + str(i))


print()
print("cart depth=2")
clf_results = cross_validate(estimator=DecisionTreeClassifier(max_depth=2, min_samples_leaf=1), X=X, y=y, scoring='accuracy',
                             cv=N_FOLDS, n_jobs=-1, verbose=10, return_train_score=True, return_estimator=True, error_score=np.nan)
print("train_accs:", clf_results['train_score'], np.mean(clf_results['train_score']))
print("test_fold:", clf_results['test_score'], np.mean(clf_results['test_score']))


# for i, clf in enumerate(clf_results['estimator']):
#     dot = tree.export_graphviz(clf)
#     graph = graphviz.Source(dot, format="png")
#     graph.render('cart_trees_' + str(i))
# sys.exit(0)


print()
print("cart depth=3")
clf_results = cross_validate(estimator=DecisionTreeClassifier(max_depth=3, min_samples_leaf=1), X=X, y=y, scoring='accuracy',
                             cv=N_FOLDS, n_jobs=-1, verbose=10, return_train_score=True, return_estimator=True, error_score=np.nan)
print("train_accs:", clf_results['train_score'], np.mean(clf_results['train_score']))
print("test_fold:", clf_results['test_score'], np.mean(clf_results['test_score']))



print()
print("cart depth=4")
clf_results = cross_validate(estimator=DecisionTreeClassifier(max_depth=4, min_samples_leaf=1), X=X, y=y, scoring='accuracy',
                             cv=N_FOLDS, n_jobs=-1, verbose=10, return_train_score=True, return_estimator=True, error_score=np.nan)
print("train_accs:", clf_results['train_score'], np.mean(clf_results['train_score']))
print("test_fold:", clf_results['test_score'], np.mean(clf_results['test_score']))


print()
print("SVM linear")
param_grid_linear = {'C': [0.1, 1, 10, 100, 1000], 'kernel': ['linear']}
grid = GridSearchCV(SVC(max_iter=1000), param_grid_linear, refit=False, verbose=10, cv=4, scoring='accuracy', n_jobs=-1, return_train_score=True)
grid.fit(X, y)
print(grid.best_params_)
print("train:", grid.cv_results_["mean_train_score"][grid.best_index_], "test", grid.cv_results_["mean_test_score"][grid.best_index_])


print()
print("SVM non-linear")
param_grid_poly = {'C': [0.1, 1, 10, 100, 1000], 'kernel': ['poly', 'rbf'], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'degree': [2, 3, 4]}
grid = GridSearchCV(SVC(max_iter=1000), param_grid_poly, refit=False, verbose=10, cv=4, scoring='accuracy', n_jobs=-1, return_train_score=True)
grid.fit(X, y)
print(grid.best_params_)
print("train:", grid.cv_results_["mean_train_score"][grid.best_index_], "test", grid.cv_results_["mean_test_score"][grid.best_index_])


print()
print("Ada 1")
param_grid_poly = {'n_estimators': [50, 100, 1000]}
grid = GridSearchCV(AdaBoostClassifier(DecisionTreeClassifier(max_depth=1)), param_grid_poly, refit=False, verbose=10,
                    cv=4, scoring='accuracy', n_jobs=-1, return_train_score=True)
grid.fit(X, y)
print(grid.best_params_)
print("train:", grid.cv_results_["mean_train_score"][grid.best_index_], "test", grid.cv_results_["mean_test_score"][grid.best_index_])


print()
print("Ada 2")
param_grid_poly = {'n_estimators': [50, 100, 1000]}
grid = GridSearchCV(AdaBoostClassifier(DecisionTreeClassifier(max_depth=2)), param_grid_poly, refit=False, verbose=10,
                    cv=4, scoring='accuracy', n_jobs=-1, return_train_score=True)
grid.fit(X, y)
print(grid.best_params_)
print("train:", grid.cv_results_["mean_train_score"][grid.best_index_], "test", grid.cv_results_["mean_test_score"][grid.best_index_])


# print()
# print("Ada 1")
# clf_results = cross_validate(estimator=AdaBoostClassifier(DecisionTreeClassifier(max_depth=1)), X=X, y=y, scoring='accuracy',
#                              cv=N_FOLDS, n_jobs=-1, verbose=10, return_train_score=True, return_estimator=True, error_score=np.nan)
# print("train_accs:", clf_results['train_score'], np.mean(clf_results['train_score']))
# print("test_fold:", clf_results['test_score'], np.mean(clf_results['test_score']))
#
#
# print()
# print("Ada 2")
# clf_results = cross_validate(estimator=AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_leaf=1)), X=X, y=y, scoring='accuracy',
#                              cv=N_FOLDS, n_jobs=-1, verbose=10, return_train_score=True, return_estimator=True, error_score=np.nan)
# print("train_accs:", clf_results['train_score'], np.mean(clf_results['train_score']))
# print("test_fold:", clf_results['test_score'], np.mean(clf_results['test_score']))

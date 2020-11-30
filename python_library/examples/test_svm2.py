from sklearn.svm import SVC
from sklearn.datasets import load_iris, load_digits, fetch_openml
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelBinarizer, KBinsDiscretizer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from dl85 import DL85Booster, DL85Classifier, BOOST_SVM2
import xgboost as xgb
import time
import numpy as np
import pandas as pd
import sys

# dataset = load_digits(3)
# X = dataset.data
# y = dataset.target
# print(X.shape)
# print(y.shape)
# enc = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='uniform')
# X = enc.fit_transform(X)
# enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
# X = enc.fit_transform(X)


X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
random_state = check_random_state(0)
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
y = y[permutation]
X = X.reshape((X.shape[0], -1))
X = X.astype('int32')
y = y.astype('int32')
print(X.shape)
print(y.shape)


# data = pd.read_csv("../datasets/matchmaker.csv", delimiter=",", header=None)
# for col in [1, 2, 3, 4, 6, 7, 8, 9]:
#     data[col] = data[col].astype('category')
# cat_columns = data.select_dtypes(['category']).columns
# data[cat_columns] = data[cat_columns].apply(lambda x: x.cat.codes)
# dataset = data.to_numpy()
# X = dataset[:, :-1]
# y = dataset[:, -1]
# enc = KBinsDiscretizer(n_bins=8, encode='ordinal', strategy='uniform')
# X = enc.fit_transform(X)
# enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
# X = enc.fit_transform(X)
# enc = LabelBinarizer()
# y = enc.fit_transform(y)
# y = y.ravel()
# np.savetxt("../datasets/matchmake.txt", np.concatenate((y.reshape(X.shape[0], 1), X), axis=1), fmt="%d", delimiter=" ")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape)
print(X.shape)



print()
print("lpdl8 d1")
best_train, best_test = [], []
parameters = {'regulator': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 10, 100, 1000, 10000, 100000, 1000000]}
kf = StratifiedKFold(n_splits=5)
for train_index, test_index in kf.split(X, y):
    X_train, y_train, X_test, y_test = X[train_index], y[train_index], X[test_index], y[test_index]
    grid = GridSearchCV(estimator=DL85Booster(max_depth=1, min_sup=1, time_limit=180, max_estimators=0, model=BOOST_SVM2),
                        param_grid=parameters, scoring='accuracy', cv=4, n_jobs=-1, verbose=0)
    # GridSearchCV(SVC(), param_grid_poly, refit=True, verbose=0, cv=4)
    grid.fit(X_train, y_train)
    print(grid.best_estimator_)
    y_pred = grid.predict(X_test)
    best_train.append(accuracy_score(y_train, grid.predict(X_train)))
    best_test.append(accuracy_score(y_test, y_pred))
    print("train_fold:", best_train[-1], "test_fold:", best_test[-1])
    # print(grid.cv_results_)
print()
print("train:", best_train, np.mean(best_train))
print("test:", best_test, np.mean(best_test))


print("lpdl8 d2")
best_train, best_test = [], []
parameters = {'regulator': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 10, 100, 1000, 10000, 100000, 1000000]}
kf = StratifiedKFold(n_splits=5)
for train_index, test_index in kf.split(X, y):
    X_train, y_train, X_test, y_test = X[train_index], y[train_index], X[test_index], y[test_index]
    grid = GridSearchCV(estimator=DL85Booster(max_depth=2, min_sup=1, time_limit=180, max_estimators=0, model=BOOST_SVM2),
                        param_grid=parameters, scoring='accuracy', cv=4, n_jobs=-1, verbose=0)
    # GridSearchCV(SVC(), param_grid_poly, refit=True, verbose=0, cv=4)
    grid.fit(X_train, y_train)
    print(grid.best_estimator_)
    y_pred = grid.predict(X_test)
    best_train.append(accuracy_score(y_train, grid.predict(X_train)))
    best_test.append(accuracy_score(y_test, y_pred))
    print("train_fold:", best_train[-1], "test_fold:", best_test[-1])
    # print(grid.cv_results_)
print()
print("train:", best_train, np.mean(best_train))
print("test:", best_test, np.mean(best_test))

print("SVM linear")
best_train, best_test = [], []
param_grid_linear = {'C': [0.1, 1, 10, 100, 1000], 'kernel': ['linear']}
kf = StratifiedKFold(n_splits=5)
for train_index, test_index in kf.split(X, y):
    X_train, y_train, X_test, y_test = X[train_index], y[train_index], X[test_index], y[test_index]
    grid = GridSearchCV(SVC(), param_grid_linear, refit=True, verbose=0, cv=4)
    grid.fit(X_train, y_train)
    print(grid.best_estimator_)
    y_pred = grid.predict(X_test)
    best_train.append(accuracy_score(y_train, grid.predict(X_train)))
    best_test.append(accuracy_score(y_test, y_pred))
    print("train_fold:", best_train[-1], "test_fold:", best_test[-1])
    # print(grid.cv_results_)
print()
print("train:", best_train, np.mean(best_train))
print("test:", best_test, np.mean(best_test))


print()
print("SVM non-linear")
best_train, best_test = [], []
param_grid_poly = {'C': [0.1, 1, 10, 100, 1000], 'kernel': ['poly', 'rbf'], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'degree': [2, 3, 4]}
kf = StratifiedKFold(n_splits=5)
for train_index, test_index in kf.split(X, y):
    X_train, y_train, X_test, y_test = X[train_index], y[train_index], X[test_index], y[test_index]
    grid = GridSearchCV(SVC(), param_grid_poly, refit=True, verbose=0, cv=4)
    grid.fit(X_train, y_train)
    print(grid.best_estimator_)
    y_pred = grid.predict(X_test)
    best_train.append(accuracy_score(y_train, grid.predict(X_train)))
    best_test.append(accuracy_score(y_test, y_pred))
    print("train_fold:", best_train[-1], "test_fold:", best_test[-1])
    # print(grid.cv_results_)
print()
print("train:", best_train, np.mean(best_train))
print("test:", best_test, np.mean(best_test))


print()
print("AdaBoost linear")
best_train, best_test = [], []
param_grid_ada = {'n_estimators': [50, 100, 1000]}
kf = StratifiedKFold(n_splits=5)
for train_index, test_index in kf.split(X, y):
    X_train, y_train, X_test, y_test = X[train_index], y[train_index], X[test_index], y[test_index]
    grid = GridSearchCV(estimator=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1)), param_grid=param_grid_ada, refit=True, verbose=0, cv=4)
    grid.fit(X_train, y_train)
    print(grid.best_estimator_)
    y_pred = grid.predict(X_test)
    best_train.append(accuracy_score(y_train, grid.predict(X_train)))
    best_test.append(accuracy_score(y_test, y_pred))
    print("train_fold:", best_train[-1], "test_fold:", best_test[-1])
    # print(grid.cv_results_)
print()
print("train:", best_train, np.mean(best_train))
print("test:", best_test, np.mean(best_test))


print()
print("AdaBoost non-linear")
best_train, best_test = [], []
param_grid_ada = {'base_estimator': [DecisionTreeClassifier(max_depth=2), DecisionTreeClassifier(max_depth=3)], 'n_estimators': [50, 100, 1000]}
kf = StratifiedKFold(n_splits=5)
for train_index, test_index in kf.split(X, y):
    X_train, y_train, X_test, y_test = X[train_index], y[train_index], X[test_index], y[test_index]
    grid = GridSearchCV(estimator=AdaBoostClassifier(), param_grid=param_grid_ada, refit=True, verbose=0, cv=4)
    grid.fit(X_train, y_train)
    print(grid.best_estimator_)
    y_pred = grid.predict(X_test)
    best_train.append(accuracy_score(y_train, grid.predict(X_train)))
    best_test.append(accuracy_score(y_test, y_pred))
    print("train_fold:", best_train[-1], "test_fold:", best_test[-1])
    # print(grid.cv_results_)
print()
print("train:", best_train, np.mean(best_train))
print("test:", best_test, np.mean(best_test))


print()
print("XGBoost linear")
best_train, best_test = [], []
param_grid_xgb = {'n_estimators': [50, 100, 1000], 'objective': ['binary:logistic', 'binary:logitraw']}
kf = StratifiedKFold(n_splits=5)
for train_index, test_index in kf.split(X, y):
    X_train, y_train, X_test, y_test = X[train_index], y[train_index], X[test_index], y[test_index]
    grid = GridSearchCV(estimator=xgb.XGBClassifier(max_depth=1), param_grid=param_grid_xgb, refit=True, verbose=0, cv=4)
    grid.fit(X_train, y_train)
    print(grid.best_estimator_)
    y_pred = grid.predict(X_test)
    best_train.append(accuracy_score(y_train, grid.predict(X_train)))
    best_test.append(accuracy_score(y_test, y_pred))
    print("train_fold:", best_train[-1], "test_fold:", best_test[-1])
    # print(grid.cv_results_)
print()
print("train:", best_train, np.mean(best_train))
print("test:", best_test, np.mean(best_test))


print()
print("XGBoost non-linear")
best_train, best_test = [], []
param_grid_xgb = {'max_depth': [2, 3], 'n_estimators': [50, 100, 1000], 'objective': ['binary:logistic', 'binary:logitraw']}
kf = StratifiedKFold(n_splits=5)
for train_index, test_index in kf.split(X, y):
    X_train, y_train, X_test, y_test = X[train_index], y[train_index], X[test_index], y[test_index]
    grid = GridSearchCV(estimator=xgb.XGBClassifier(), param_grid=param_grid_xgb, refit=True, verbose=0, cv=4)
    grid.fit(X_train, y_train)
    print(grid.best_estimator_)
    y_pred = grid.predict(X_test)
    best_train.append(accuracy_score(y_train, grid.predict(X_train)))
    best_test.append(accuracy_score(y_test, y_pred))
    print("train_fold:", best_train[-1], "test_fold:", best_test[-1])
    # print(grid.cv_results_)
print()
print("train:", best_train, np.mean(best_train))
print("test:", best_test, np.mean(best_test))


# print()
# print("lpdl8 d1")
# best_train, best_test = [], []
# parameters = {'regulator': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 10, 100, 1000, 10000, 100000, 1000000]}
# kf = StratifiedKFold(n_splits=5)
# for train_index, test_index in kf.split(X, y):
#     X_train, y_train, X_test, y_test = X[train_index], y[train_index], X[test_index], y[test_index]
#     grid = GridSearchCV(estimator=DL85Booster(max_depth=1, min_sup=1, time_limit=180, max_estimators=0, model=BOOST_SVM2),
#                         param_grid=parameters, scoring='accuracy', cv=4, n_jobs=-1, verbose=0)
#     # GridSearchCV(SVC(), param_grid_poly, refit=True, verbose=0, cv=4)
#     grid.fit(X_train, y_train)
#     print(grid.best_estimator_)
#     y_pred = grid.predict(X_test)
#     best_train.append(accuracy_score(y_train, grid.predict(X_train)))
#     best_test.append(accuracy_score(y_test, y_pred))
#     print("train_fold:", best_train[-1], "test_fold:", best_test[-1])
#     # print(grid.cv_results_)
# print()
# print("train:", best_train, np.mean(best_train))
# print("test:", best_test, np.mean(best_test))
#
#
# print("lpdl8 d2")
# best_train, best_test = [], []
# parameters = {'regulator': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 10, 100, 1000, 10000, 100000, 1000000]}
# kf = StratifiedKFold(n_splits=5)
# for train_index, test_index in kf.split(X, y):
#     X_train, y_train, X_test, y_test = X[train_index], y[train_index], X[test_index], y[test_index]
#     grid = GridSearchCV(estimator=DL85Booster(max_depth=2, min_sup=1, time_limit=180, max_estimators=0, model=BOOST_SVM2),
#                         param_grid=parameters, scoring='accuracy', cv=4, n_jobs=-1, verbose=0)
#     # GridSearchCV(SVC(), param_grid_poly, refit=True, verbose=0, cv=4)
#     grid.fit(X_train, y_train)
#     print(grid.best_estimator_)
#     y_pred = grid.predict(X_test)
#     best_train.append(accuracy_score(y_train, grid.predict(X_train)))
#     best_test.append(accuracy_score(y_test, y_pred))
#     print("train_fold:", best_train[-1], "test_fold:", best_test[-1])
#     # print(grid.cv_results_)
# print()
# print("train:", best_train, np.mean(best_train))
# print("test:", best_test, np.mean(best_test))


# clf_results = cross_validate(estimator=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=100), X=X, y=y, scoring='accuracy', cv=5, n_jobs=-1, verbose=0, return_train_score=True, return_estimator=True, error_score=np.nan)
# print("train:", clf_results['train_score'], np.mean(clf_results['train_score']))
# print("test:", clf_results['test_score'], np.mean(clf_results['test_score']))

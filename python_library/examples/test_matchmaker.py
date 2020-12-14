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
import matplotlib.pyplot as plt
from sklearn import tree

depth, time_limit, N_FOLDS = 2, 900, 5

# read csv with pandas
data = pd.read_csv("../datasets/matchmaker.csv", delimiter=",", header=None)
# convert string features as category
for col in [1, 2, 3, 4, 6, 7, 8, 9]:
    data[col] = data[col].astype('category')
# transform each category into int
cat_columns = data.select_dtypes(['category']).columns
data[cat_columns] = data[cat_columns].apply(lambda x: x.cat.codes)
# convert the int dataframe to numpy
dataset = data.to_numpy()
# split features and target
X = dataset[:, :-1]
y = dataset[:, -1]
# for each feature, create at most 8 categories of equal width for discretization
enc = KBinsDiscretizer(n_bins=8, encode='ordinal', strategy='uniform')
X = enc.fit_transform(X)
# then use one-hot encoding to binarize the dataset
enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
X = enc.fit_transform(X)
# convert values to int
X, y = X.astype('int32'), y.astype('int32')
print(X.shape, y.shape)
print(set(y))
# np.savetxt("../datasets/matchmaker.txt", np.concatenate((y.reshape(X.shape[0], 1), X), axis=1), fmt="%d", delimiter=" ")

print("matchmaker")

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

for clf in clf_results['estimator']:
    print(clf.tree_)


print()
print("cart depth=2")
clf_results = cross_validate(estimator=DecisionTreeClassifier(max_depth=depth, min_samples_leaf=1, criterion='entropy'), X=X, y=y, scoring='accuracy',
                             cv=N_FOLDS, n_jobs=-1, verbose=10, return_train_score=True, return_estimator=True, error_score=np.nan)
print("train_accs:", clf_results['train_score'], np.mean(clf_results['train_score']))
print("test_fold:", clf_results['test_score'], np.mean(clf_results['test_score']))

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=300)
for i, clf in enumerate(clf_results['estimator']):
    tree.plot_tree(clf, filled=True)
    fig.savefig('imagenames_' + str(i) + '.png')
sys.exit(0)


print()
print("lpdl8 d" + str(depth))
parameters = {'regulator': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 10, 100, 1000, 10000, 100000, 1000000]}
n_trees, train_accs, test_accs, reguls = [], [], [], []
for k in range(N_FOLDS):
    X_train, X_test, y_train, y_test = X_trains[k], X_tests[k], y_trains[k], y_tests[k]
    grid = GridSearchCV(estimator=DL85Booster(max_depth=depth, min_sup=1, time_limit=time_limit, max_estimators=0, model=BOOST_SVM2),
                        param_grid=parameters, scoring='accuracy', cv=4, n_jobs=-1, verbose=10)
    grid.fit(X_train, y_train)
    n_trees.append(grid.best_estimator_.n_estimators_)
    y_pred = grid.predict(X_test)
    train_accs.append(accuracy_score(y_train, grid.predict(X_train)))
    test_accs.append(accuracy_score(y_test, y_pred))
    reguls.append(grid.best_params_['regulator'])
print('n_trees:', n_trees, np.mean(n_trees))
print("train_accs:", train_accs, np.mean(train_accs))
print("test_fold:", test_accs, np.mean(test_accs))
print("regulator", reguls)
max_trees = int(np.mean(n_trees))


print()
print("AdaBoost non-linear max_trees")
clf_results = cross_validate(estimator=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=depth), n_estimators=max_trees), X=X, y=y, scoring='accuracy',
                             cv=N_FOLDS, n_jobs=-1, verbose=10, return_train_score=True, return_estimator=True, error_score=np.nan)
n_trees = list(map(lambda x: len(x.estimators_), clf_results['estimator']))
print('n_trees:', n_trees, np.mean(n_trees))
print("train_accs:", clf_results['train_score'], np.mean(clf_results['train_score']))
print("test_fold:", clf_results['test_score'], np.mean(clf_results['test_score']))


print()
print("AdaBoost non-linear")
param_grid_ada = {'n_estimators': [50, 100, 1000]}
n_trees, train_accs, test_accs = [], [], []
for k in range(N_FOLDS):
    X_train, X_test, y_train, y_test = X_trains[k], X_tests[k], y_trains[k], y_tests[k]
    grid = GridSearchCV(estimator=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=depth)),
                        param_grid=param_grid_ada, refit=True, verbose=10, cv=4, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    n_trees.append(len(grid.best_estimator_.estimators_))
    y_pred = grid.predict(X_test)
    train_accs.append(accuracy_score(y_train, grid.predict(X_train)))
    test_accs.append(accuracy_score(y_test, y_pred))
print('n_trees:', n_trees, np.mean(n_trees))
print("train_accs:", train_accs, np.mean(train_accs))
print("test_fold:", test_accs, np.mean(test_accs))


print()
print("XGBoost non-linear maxtrees")
param_grid_xgb = {'objective': ['binary:logistic', 'binary:logitraw']}
n_trees, train_accs, test_accs, obj = [], [], [], []
for k in range(N_FOLDS):
    X_train, X_test, y_train, y_test = X_trains[k], X_tests[k], y_trains[k], y_tests[k]
    grid = GridSearchCV(estimator=xgb.XGBClassifier(n_estimators=max_trees, max_depth=depth),
                        param_grid=param_grid_xgb, refit=True, verbose=10, cv=4, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    n_trees.append(grid.best_estimator_.n_estimators)
    y_pred = grid.predict(X_test)
    obj.append(grid.best_params_['objective'])
    train_accs.append(accuracy_score(y_train, grid.predict(X_train)))
    test_accs.append(accuracy_score(y_test, y_pred))
print('n_trees:', n_trees, np.mean(n_trees))
print("train_accs:", train_accs, np.mean(train_accs))
print("test_fold:", test_accs, np.mean(test_accs))
print("objective:", obj)


print()
print("XGBoost non-linear")
param_grid_xgb = {'n_estimators': [50, 100, 1000], 'objective': ['binary:logistic', 'binary:logitraw']}
n_trees, train_accs, test_accs, obj = [], [], [], []
for k in range(N_FOLDS):
    X_train, X_test, y_train, y_test = X_trains[k], X_tests[k], y_trains[k], y_tests[k]
    grid = GridSearchCV(estimator=xgb.XGBClassifier(max_depth=depth), param_grid=param_grid_xgb,
                        refit=True, verbose=10, cv=4, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    n_trees.append(grid.best_estimator_.n_estimators)
    y_pred = grid.predict(X_test)
    obj.append(grid.best_params_['objective'])
    train_accs.append(accuracy_score(y_train, grid.predict(X_train)))
    test_accs.append(accuracy_score(y_test, y_pred))
print('n_trees:', n_trees, np.mean(n_trees))
print("train_accs:", train_accs, np.mean(train_accs))
print("test_fold:", test_accs, np.mean(test_accs))
print("objective:", obj)


print()
print("RF non-linear maxtrees")
clf_results = cross_validate(estimator=RandomForestClassifier(max_depth=depth, n_estimators=max_trees), X=X, y=y, scoring='accuracy',
                             cv=N_FOLDS, n_jobs=-1, verbose=10, return_train_score=True, return_estimator=True, error_score=np.nan)
n_trees = list(map(lambda x: len(x.estimators_), clf_results['estimator']))
print('n_trees:', n_trees, np.mean(n_trees))
print("train_accs:", clf_results['train_score'], np.mean(clf_results['train_score']))
print("test_fold:", clf_results['test_score'], np.mean(clf_results['test_score']))


print()
print("RF non-linear")
param_grid_rf = {'n_estimators': [50, 100, 1000]}
n_trees, train_accs, test_accs = [], [], []
for k in range(N_FOLDS):
    X_train, X_test, y_train, y_test = X_trains[k], X_tests[k], y_trains[k], y_tests[k]
    grid = GridSearchCV(estimator=RandomForestClassifier(max_depth=depth), param_grid=param_grid_rf,
                        refit=True, verbose=10, cv=4, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    n_trees.append(len(grid.best_estimator_.estimators_))
    y_pred = grid.predict(X_test)
    train_accs.append(accuracy_score(y_train, grid.predict(X_train)))
    test_accs.append(accuracy_score(y_test, y_pred))
print('n_trees:', n_trees, np.mean(n_trees))
print("train_accs:", train_accs, np.mean(train_accs))
print("test_fold:", test_accs, np.mean(test_accs))


print()
print("GB non-linear maxtrees")
clf_results = cross_validate(estimator=GradientBoostingClassifier(max_depth=depth, n_estimators=max_trees), X=X, y=y, scoring='accuracy',
                             cv=N_FOLDS, n_jobs=-1, verbose=10, return_train_score=True, return_estimator=True, error_score=np.nan)
n_trees = list(map(lambda x: x.n_estimators, clf_results['estimator']))
print('n_trees:', n_trees, np.mean(n_trees))
print("train_accs:", clf_results['train_score'], np.mean(clf_results['train_score']))
print("test_fold:", clf_results['test_score'], np.mean(clf_results['test_score']))


print()
print("GB non-linear")
param_grid_gb = {'n_estimators': [50, 100, 1000]}
n_trees, train_accs, test_accs = [], [], []
for k in range(N_FOLDS):
    X_train, X_test, y_train, y_test = X_trains[k], X_tests[k], y_trains[k], y_tests[k]
    grid = GridSearchCV(estimator=GradientBoostingClassifier(max_depth=depth), param_grid=param_grid_gb,
                        refit=True, verbose=10, cv=4, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    n_trees.append(grid.best_estimator_.n_estimators)
    y_pred = grid.predict(X_test)
    train_accs.append(accuracy_score(y_train, grid.predict(X_train)))
    test_accs.append(accuracy_score(y_test, y_pred))
print('n_trees:', n_trees, np.mean(n_trees))
print("train_accs:", train_accs, np.mean(train_accs))
print("test_fold:", test_accs, np.mean(test_accs))



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

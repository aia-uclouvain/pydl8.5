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
# import graphviz
# from geopy.geocoders import Nominatim
# from geopy.distance import geodesic
# from geopy.extra.rate_limiter import RateLimiter

depth, time_limit, N_FOLDS = 1, 0, 5

# # read csv with pandas
# data = pd.read_csv("../datasets/matchmaker.csv", delimiter=",", header=0)
# # replace missing values by no interests
# data.interests_1.replace(np.nan, '', inplace=True)
# data.interests_2.replace(np.nan, '', inplace=True)
# # create a column to compute the number of common interests
# data['interests_1'] = data['interests_1'].str.split(":", expand=False)
# data['interests_2'] = data['interests_2'].str.split(":", expand=False)
# data['n_common_hobby'] = [len(set(a).intersection(b)) for a, b in zip(data.interests_1, data.interests_2)]
# # remove interests columns
# data.drop('interests_1', axis=1, inplace=True)
# data.drop('interests_2', axis=1, inplace=True)
# # convert yes/no columns to 0/1
# data["smoker_1"] = pd.Series(np.where(data.smoker_1.values == 'yes', 1, 0), data.index)
# data["smoker_2"] = pd.Series(np.where(data.smoker_2.values == 'yes', 1, 0), data.index)
# data["want_children_1"] = pd.Series(np.where(data.want_children_1.values == 'yes', 1, 0), data.index)
# data["want_children_2"] = pd.Series(np.where(data.want_children_2.values == 'yes', 1, 0), data.index)
# # compute distance between the two locations (it will take time)
# geolocator = Nominatim(user_agent="my_app")
# delay_geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, error_wait_seconds=5)
# # data['distance'] = [int(geodesic(delay_geocode(a)[1], delay_geocode(b)[1]).km) for a, b in zip(data.location_1, data.location_2)]
# data["location_1"] = data.location_1.apply(delay_geocode)
# print("get coords a")
# data["location_1"] = data.location_1.apply(lambda loc: tuple(loc.point)[:-1] if loc else None)
# data["location_2"] = data.location_2.apply(delay_geocode)
# print("get coords b")
# data["location_2"] = data.location_2.apply(lambda loc: tuple(loc.point)[:-1] if loc else None)
# data["distance"] = [geodesic(a, b).km if a is not None and b is not None else np.nan for a, b in zip(data.location_1, data.location_2)]
# # remove locations columns
# data.drop('location_1', axis=1, inplace=True)
# data.drop('location_2', axis=1, inplace=True)
# # re-arrange the columns to make the class be the last one
# data = data[[c for c in data if c not in ['match']] + ['match']]
# # convert the int dataframe to numpy
# dataset = data.to_numpy()
# # export the processed dataset to csv
# np.savetxt("../datasets/clean_mm.csv", dataset, fmt='%d', delimiter=",")
# sys.exit(0)

file = open("../output/kr-vs-kp_" + str(depth) + ".txt", "w")

# split features and target
dataset = np.genfromtxt("../datasets/kr-vs-kp.txt", delimiter=" ")
X = dataset[:, 1:]
y = dataset[:, 0]
# # for each feature, create at most 8 categories of equal width for discretization
# enc = KBinsDiscretizer(n_bins=8, encode='ordinal', strategy='uniform')
# X = enc.fit_transform(X)
# # then use one-hot encoding to binarize the dataset
# enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
# X = enc.fit_transform(X)
# convert values to int
X, y = X.astype('int32'), y.astype('int32')
print(X.shape, y.shape)
print(set(y))
file.write(str(X.shape) + " " + str(y.shape) + "\n")
file.write(str(set(y)) + "\n")
# np.savetxt("../datasets/matchmaker.txt", np.concatenate((y.reshape(X.shape[0], 1), X), axis=1), fmt="%d", delimiter=" ")

print("kr-vs-kp")

kf = StratifiedKFold(n_splits=5)
X_trains, X_tests, y_trains, y_tests = [], [], [], []
for train_index, test_index in kf.split(X, y):
    X_trains.append(X[train_index])
    y_trains.append(y[train_index])
    X_tests.append(X[test_index])
    y_tests.append(y[test_index])

for d in range(1, depth+1):
    print()
    print("dl85 depth =", d)
    file.write("\ndl85 depth=" + str(d) + "\n")
    clf_results = cross_validate(estimator=DL85Classifier(max_depth=d, min_sup=1, time_limit=time_limit), X=X, y=y, scoring='accuracy',
                                 cv=N_FOLDS, n_jobs=-1, verbose=10, return_train_score=True, return_estimator=True, error_score=np.nan)
    print("train_accs:", clf_results['train_score'], np.mean(clf_results['train_score']))
    print("test_fold:", clf_results['test_score'], np.mean(clf_results['test_score']))
    file.write("train_accs: " + str(clf_results['train_score']) + " " + str(np.mean(clf_results['train_score'])) + "\n")
    file.write("test_fold: " + str(clf_results['test_score']) + " " + str(np.mean(clf_results['test_score'])) + "\n")
    file.flush()

    print()
    print("cart depth =", d)
    file.write("\ncart depth=" + str(d) + "\n")
    clf_results = cross_validate(estimator=DecisionTreeClassifier(max_depth=d, min_samples_leaf=1), X=X, y=y, scoring='accuracy',
                                 cv=N_FOLDS, n_jobs=-1, verbose=10, return_train_score=True, return_estimator=True, error_score=np.nan)
    print("train_accs:", clf_results['train_score'], np.mean(clf_results['train_score']))
    print("test_fold:", clf_results['test_score'], np.mean(clf_results['test_score']))
    file.write("train_accs: " + str(clf_results['train_score']) + " " + str(np.mean(clf_results['train_score'])) + "\n")
    file.write("test_fold: " + str(clf_results['test_score']) + " " + str(np.mean(clf_results['test_score'])) + "\n")
    file.flush()


# print()
# print("dl85 depth=2")
# clf_results = cross_validate(estimator=DL85Classifier(max_depth=depth, min_sup=1, time_limit=time_limit), X=X, y=y, scoring='accuracy',
#                              cv=N_FOLDS, n_jobs=-1, verbose=10, return_train_score=True, return_estimator=True, error_score=np.nan)
# print("train_accs:", clf_results['train_score'], np.mean(clf_results['train_score']))
# print("test_fold:", clf_results['test_score'], np.mean(clf_results['test_score']))

# for i, clf in enumerate(clf_results['estimator']):
#     dot = clf.export_graphviz()
#     graph = graphviz.Source(dot, format="png")
#     graph.render('dl85_trees_' + str(i))


# print()
# print("cart depth=2")
# clf_results = cross_validate(estimator=DecisionTreeClassifier(max_depth=depth, min_samples_leaf=1, criterion='entropy'), X=X, y=y, scoring='accuracy',
#                              cv=N_FOLDS, n_jobs=-1, verbose=10, return_train_score=True, return_estimator=True, error_score=np.nan)
# print("train_accs:", clf_results['train_score'], np.mean(clf_results['train_score']))
# print("test_fold:", clf_results['test_score'], np.mean(clf_results['test_score']))


# for i, clf in enumerate(clf_results['estimator']):
#     dot = tree.export_graphviz(clf)
#     graph = graphviz.Source(dot, format="png")
#     graph.render('cart_trees_' + str(i))
# sys.exit(0)


print()
print("lpdl8 d" + str(depth))
file.write("\n\nlpdl85 depth=" + str(depth) + "\n")
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
    print("best params:", grid.best_params_)
    print("train_acc for fold", k, "=", train_accs[-1])
    print("test_acc for fold", k, "=", test_accs[-1])
    file.write("best params:" + str(grid.best_params_) + "\n")
    file.write("train_acc for fold" + " " + str(k) + " = " + str(train_accs[-1]) + "\n")
    file.write("test_acc for fold" + " " + str(k) + " = " + str(test_accs[-1]) + "\n")
    file.write("n_trees for fold" + " " + str(k) + " = " + str(n_trees[-1]) + "\n\n")
    file.flush()
    print()
print('n_trees:', n_trees, np.mean(n_trees))
print("train_accs:", train_accs, np.mean(train_accs))
print("test_accs:", test_accs, np.mean(test_accs))
print("regulator", reguls)
file.write('n_trees: ' + str(n_trees) + " " + str(np.mean(n_trees)) + "\n")
file.write("train_accs: " + str(train_accs) + " " + str(np.mean(train_accs)) + "\n")
file.write("test_accs: " + str(test_accs) + " " + str(np.mean(test_accs)) + "\n")
file.write("regulator: " + str(reguls) + "\n")
file.flush()
max_trees = int(np.mean(n_trees))
time.sleep(5)


print("\n")
print("AdaBoost non-linear max_trees")
file.write("\n\nAdaBoost non-linear max_trees\n")
clf_results = cross_validate(estimator=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=depth), n_estimators=max_trees), X=X, y=y, scoring='accuracy',
                             cv=N_FOLDS, n_jobs=-1, verbose=10, return_train_score=True, return_estimator=True, error_score=np.nan)
n_trees = list(map(lambda x: len(x.estimators_), clf_results['estimator']))
print('n_trees:', n_trees, np.mean(n_trees))
print("train_accs:", clf_results['train_score'], np.mean(clf_results['train_score']))
print("test_accs:", clf_results['test_score'], np.mean(clf_results['test_score']))
file.write('n_trees: ' + str(n_trees) + " " + str(np.mean(n_trees)) + "\n")
file.write("train_accs: " + str(clf_results['train_score']) + " " + str(np.mean(clf_results['train_score'])) + "\n")
file.write("test_accs: " + str(clf_results['test_score']) + " " + str(np.mean(clf_results['test_score'])) + "\n")
file.flush()
time.sleep(5)


print("\n")
print("AdaBoost non-linear")
file.write("\n\nAdaBoost non-linear\n")
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
    print("best params:", grid.best_params_)
    print("train_acc for fold", k, "=", train_accs[-1])
    print("test_acc for fold", k, "=", test_accs[-1])
    print("n_trees for fold", k, "=", n_trees[-1])
    file.write("best params:" + str(grid.best_params_) + "\n")
    file.write("train_acc for fold" + " " + str(k) + " = " + str(train_accs[-1]) + "\n")
    file.write("test_acc for fold" + " " + str(k) + " = " + str(test_accs[-1]) + "\n")
    file.write("n_trees for fold" + " " + str(k) + " = " + str(n_trees[-1]) + "\n\n")
    file.flush()
    print()
print('n_trees:', n_trees, np.mean(n_trees))
print("train_accs:", train_accs, np.mean(train_accs))
print("test_accs:", test_accs, np.mean(test_accs))
file.write('n_trees: ' + str(n_trees) + " " + str(np.mean(n_trees)) + "\n")
file.write("train_accs: " + str(train_accs) + " " + str(np.mean(train_accs)) + "\n")
file.write("test_accs: " + str(test_accs) + " " + str(np.mean(test_accs)) + "\n")
file.flush()
time.sleep(5)


print("\n")
print("XGBoost non-linear maxtrees")
file.write("\n\nXGBoost non-linear maxtrees\n")
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
    print("best params:", grid.best_params_)
    print("train_acc for fold", k, "=", train_accs[-1])
    print("test_acc for fold", k, "=", test_accs[-1])
    print("n_trees for fold", k, "=", n_trees[-1])
    file.write("best params:" + str(grid.best_params_) + "\n")
    file.write("train_acc for fold" + " " + str(k) + " = " + str(train_accs[-1]) + "\n")
    file.write("test_acc for fold" + " " + str(k) + " = " + str(test_accs[-1]) + "\n")
    file.write("n_trees for fold" + " " + str(k) + " = " + str(n_trees[-1]) + "\n\n")
    file.flush()
    print()
print('n_trees:', n_trees, np.mean(n_trees))
print("train_accs:", train_accs, np.mean(train_accs))
print("test_fold:", test_accs, np.mean(test_accs))
print("objective:", obj)
file.write('n_trees: ' + str(n_trees) + " " + str(np.mean(n_trees)) + "\n")
file.write("train_accs: " + str(train_accs) + " " + str(np.mean(train_accs)) + "\n")
file.write("test_accs: " + str(test_accs) + " " + str(np.mean(test_accs)) + "\n")
file.write("objective: " + str(obj) + "\n")
file.flush()
time.sleep(5)


print("\n")
print("XGBoost non-linear")
file.write("\n\nXGBoost non-linear\n")
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
    print("best params:", grid.best_params_)
    print("train_acc for fold", k, "=", train_accs[-1])
    print("test_acc for fold", k, "=", test_accs[-1])
    print("n_trees for fold", k, "=", n_trees[-1])
    file.write("best params:" + str(grid.best_params_) + "\n")
    file.write("train_acc for fold" + " " + str(k) + " = " + str(train_accs[-1]) + "\n")
    file.write("test_acc for fold" + " " + str(k) + " = " + str(test_accs[-1]) + "\n")
    file.write("n_trees for fold" + " " + str(k) + " = " + str(n_trees[-1]) + "\n\n")
    file.flush()
    print()
print('n_trees:', n_trees, np.mean(n_trees))
print("train_accs:", train_accs, np.mean(train_accs))
print("test_accs:", test_accs, np.mean(test_accs))
print("objective:", obj)
file.write('n_trees: ' + str(n_trees) + " " + str(np.mean(n_trees)) + "\n")
file.write("train_accs: " + str(train_accs) + " " + str(np.mean(train_accs)) + "\n")
file.write("test_accs: " + str(test_accs) + " " + str(np.mean(test_accs)) + "\n")
file.write("objective: " + str(obj) + "\n")
file.flush()
time.sleep(5)


print("\n")
print("RF non-linear maxtrees")
file.write("\n\nRF non-linear maxtrees\n")
clf_results = cross_validate(estimator=RandomForestClassifier(max_depth=depth, n_estimators=max_trees), X=X, y=y, scoring='accuracy',
                             cv=N_FOLDS, n_jobs=-1, verbose=10, return_train_score=True, return_estimator=True, error_score=np.nan)
n_trees = list(map(lambda x: len(x.estimators_), clf_results['estimator']))
print('n_trees:', n_trees, np.mean(n_trees))
print("train_accs:", clf_results['train_score'], np.mean(clf_results['train_score']))
print("test_accs:", clf_results['test_score'], np.mean(clf_results['test_score']))
file.write('n_trees: ' + str(n_trees) + " " + str(np.mean(n_trees)) + "\n")
file.write("train_accs: " + str(clf_results['train_score']) + " " + str(np.mean(clf_results['train_score'])) + "\n")
file.write("test_accs: " + str(clf_results['test_score']) + " " + str(np.mean(clf_results['test_score'])) + "\n")
file.flush()
time.sleep(5)


print("\n")
print("RF non-linear")
file.write("\n\nRF non-linear\n")
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
    print("best params:", grid.best_params_)
    print("train_acc for fold", k, "=", train_accs[-1])
    print("test_acc for fold", k, "=", test_accs[-1])
    print("n_trees for fold", k, "=", n_trees[-1])
    file.write("best params:" + str(grid.best_params_) + "\n")
    file.write("train_acc for fold" + " " + str(k) + " = " + str(train_accs[-1]) + "\n")
    file.write("test_acc for fold" + " " + str(k) + " = " + str(test_accs[-1]) + "\n")
    file.write("n_trees for fold" + " " + str(k) + " = " + str(n_trees[-1]) + "\n\n")
    file.flush()
    print()
print('n_trees:', n_trees, np.mean(n_trees))
print("train_accs:", train_accs, np.mean(train_accs))
print("test_fold:", test_accs, np.mean(test_accs))
file.write('n_trees: ' + str(n_trees) + " " + str(np.mean(n_trees)) + "\n")
file.write("train_accs: " + str(train_accs) + " " + str(np.mean(train_accs)) + "\n")
file.write("test_accs: " + str(test_accs) + " " + str(np.mean(test_accs)) + "\n")
file.flush()
time.sleep(5)


print("\n")
print("GB non-linear maxtrees")
file.write("\n\nGB non-linear maxtrees\n")
clf_results = cross_validate(estimator=GradientBoostingClassifier(max_depth=depth, n_estimators=max_trees), X=X, y=y, scoring='accuracy',
                             cv=N_FOLDS, n_jobs=-1, verbose=10, return_train_score=True, return_estimator=True, error_score=np.nan)
n_trees = list(map(lambda x: x.n_estimators, clf_results['estimator']))
print('n_trees:', n_trees, np.mean(n_trees))
print("train_accs:", clf_results['train_score'], np.mean(clf_results['train_score']))
print("test_accs:", clf_results['test_score'], np.mean(clf_results['test_score']))
file.write('n_trees: ' + str(n_trees) + " " + str(np.mean(n_trees)) + "\n")
file.write("train_accs: " + str(clf_results['train_score']) + " " + str(np.mean(clf_results['train_score'])) + "\n")
file.write("test_accs: " + str(clf_results['test_score']) + " " + str(np.mean(clf_results['test_score'])) + "\n")
file.flush()
time.sleep(5)


print("\n")
print("GB non-linear")
file.write("\n\nGB non-linear\n")
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
    print("best params:", grid.best_params_)
    print("train_acc for fold", k, "=", train_accs[-1])
    print("test_acc for fold", k, "=", test_accs[-1])
    print("n_trees for fold", k, "=", n_trees[-1])
    file.write("best params:" + str(grid.best_params_) + "\n")
    file.write("train_acc for fold" + " " + str(k) + " = " + str(train_accs[-1]) + "\n")
    file.write("test_acc for fold" + " " + str(k) + " = " + str(test_accs[-1]) + "\n")
    file.write("n_trees for fold" + " " + str(k) + " = " + str(n_trees[-1]) + "\n\n")
    file.flush()
    print()
print('n_trees:', n_trees, np.mean(n_trees))
print("train_accs:", train_accs, np.mean(train_accs))
print("test_accs:", test_accs, np.mean(test_accs))
file.write('n_trees: ' + str(n_trees) + " " + str(np.mean(n_trees)) + "\n")
file.write("train_accs: " + str(train_accs) + " " + str(np.mean(train_accs)) + "\n")
file.write("test_accs: " + str(test_accs) + " " + str(np.mean(test_accs)) + "\n")
file.flush()
time.sleep(5)


print("\n")
print("SVM linear")
file.write("\n\nSVM linear\n")
param_grid_linear = {'C': [0.1, 1, 10, 100, 1000], 'kernel': ['linear']}
train_accs, test_accs = [], []
for k in range(N_FOLDS):
    X_train, X_test, y_train, y_test = X_trains[k], X_tests[k], y_trains[k], y_tests[k]
    grid = GridSearchCV(SVC(max_iter=1000), param_grid_linear, refit=True, verbose=10, cv=4, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    train_accs.append(accuracy_score(y_train, grid.predict(X_train)))
    test_accs.append(accuracy_score(y_test, y_pred))
    print("best params:", grid.best_params_)
    print("train_acc for fold", k, "=", train_accs[-1])
    print("test_acc for fold", k, "=", test_accs[-1])
    file.write("best params:" + str(grid.best_params_) + "\n")
    file.write("train_acc for fold" + " " + str(k) + " = " + str(train_accs[-1]) + "\n")
    file.write("test_acc for fold" + " " + str(k) + " = " + str(test_accs[-1]) + "\n\n")
    file.flush()
    print()
print("train_accs:", train_accs, np.mean(train_accs))
print("test_accs:", test_accs, np.mean(test_accs))
file.write("train_accs: " + str(train_accs) + " " + str(np.mean(train_accs)) + "\n")
file.write("test_accs: " + str(test_accs) + " " + str(np.mean(test_accs)) + "\n")
file.flush()
time.sleep(5)


print("\n")
print("SVM non-linear")
file.write("\n\nSVM non-linear\n")
param_grid_poly = {'C': [0.1, 1, 10, 100, 1000], 'kernel': ['poly', 'rbf'], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'degree': [2, 3, 4]}
train_accs, test_accs = [], []
for k in range(N_FOLDS):
    X_train, X_test, y_train, y_test = X_trains[k], X_tests[k], y_trains[k], y_tests[k]
    grid = GridSearchCV(SVC(max_iter=1000), param_grid_poly, refit=True, verbose=10, cv=4, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    train_accs.append(accuracy_score(y_train, grid.predict(X_train)))
    test_accs.append(accuracy_score(y_test, y_pred))
    print("best params:", grid.best_params_)
    print("train_acc for fold", k, "=", train_accs[-1])
    print("test_acc for fold", k, "=", test_accs[-1])
    file.write("best params:" + str(grid.best_params_) + "\n")
    file.write("train_acc for fold" + " " + str(k) + " = " + str(train_accs[-1]) + "\n")
    file.write("test_acc for fold" + " " + str(k) + " = " + str(test_accs[-1]) + "\n\n")
    file.flush()
    print()
print("train_accs:", train_accs, np.mean(train_accs))
print("test_accs:", test_accs, np.mean(test_accs))
file.write("train_accs: " + str(train_accs) + " " + str(np.mean(train_accs)) + "\n")
file.write("test_accs: " + str(test_accs) + " " + str(np.mean(test_accs)) + "\n")
file.flush()

file.close()

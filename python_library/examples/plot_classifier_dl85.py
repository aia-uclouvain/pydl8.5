"""
======================
Default DL85Classifier
======================

"""
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import time
from dl85 import DL85Classifier
from sklearn.tree import DecisionTreeClassifier
import random

dataset = np.genfromtxt("../../datasets/soybean.txt", delimiter=' ')
X = dataset[:, 1:]
y = dataset[:, 0]
X = X.astype('int32')
y = y.astype('int32')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


print("######################################################################\n"
      "#                      DL8.5 default classifier                      #\n"
      "######################################################################")
clf = DL85Classifier(max_depth=2, time_limit=600, desc=True, verbose=True)
start = time.perf_counter()
print("Model building...")
clf.fit(X, y)
duration = time.perf_counter() - start
print("Model built. Duration of building =", round(duration, 4))
y_pred = clf.predict(X_test)
print("Confusion Matrix below")
print(confusion_matrix(y_test, y_pred))
print("Accuracy DL8.5 on training set =", round(clf.accuracy_, 4))
print("Accuracy DL8.5 on test set =", round(accuracy_score(y_test, y_pred), 4), "\n\n\n")


print("##############################################################\n"
      "#     DL8.5 classifier : Manual cross-validation (5-fold)    #\n"
      "##############################################################")
kf = KFold(n_splits=5, random_state=42, shuffle=True)
training_accuracies = []
test_accuracies = []
start = time.perf_counter()
print("Model building...")
for train_index, test_index in kf.split(X):
    data_train = X[train_index]
    target_train = y[train_index]
    data_test = X[test_index]
    target_test = y[test_index]
    clf = DL85Classifier(max_depth=2, time_limit=600)
    clf.fit(data_train, target_train)
    preds = clf.predict(data_test)
    training_accuracies.append(clf.accuracy_)
    test_accuracies.append(accuracy_score(target_test, preds))
duration = time.perf_counter() - start
print("Model built. Duration of building =", round(duration, 4))
print("Average accuracy on training set =", round(np.mean(training_accuracies), 4))
print("Average accuracy on test set =", round(np.mean(test_accuracies), 4), "\n\n\n")


print("##############################################################\n"
      "#   DL8.5 classifier : Automatic cross-validation (5-fold)   #\n"
      "##############################################################")
clf = DL85Classifier(max_depth=2, time_limit=600)
start = time.perf_counter()
print("Model building...")
scores = cross_val_score(clf, X, y, cv=5)
duration = time.perf_counter() - start
print("Model built. Duration of building =", round(duration, 4))
print("Average accuracy on test set =", round(np.mean(scores), 4))

# clf = DecisionTreeClassifier(max_depth=2)
# # clf = DL85Classifier(max_depth=2)
# start = time.perf_counter()
# print("Model building...")
# l = []
# for i in range(X_train.shape[0]):
#     l.append(random.uniform(0, 1))
# print(l)
# clf.fit(X_train, y_train, sample_weight=l)
# duration = time.perf_counter() - start
# print("Model built. Duration of building =", round(duration, 4))
# y_pred = clf.predict(X_test)
# print("Confusion Matrix below")
# print(confusion_matrix(y_test, y_pred))
# print("Accuracy DL8.5 on training set =", round(accuracy_score(y_train, clf.predict(X_train)), 4))
# print("Accuracy DL8.5 on test set =", round(accuracy_score(y_test, y_pred), 4), "\n\n\n")
#
# # clf = DecisionTreeClassifier(max_depth=1)
# # clf = DL85Classifier(max_depth=2)
# start = time.perf_counter()
# print("Model building...")
# # l = []
# # for i in range(X_train.shape[0]):
# #     l.append(random.uniform(0, 1))
# for i in range(400):
#     l[i] = 0
#     # l[random.randint(0, 500)] = 0
# print(l)
# print(X_train.shape)
# clf.fit(X_train, y_train, sample_weight=l)
# duration = time.perf_counter() - start
# print("Model built. Duration of building =", round(duration, 4))
# y_pred = clf.predict(X_test)
# print("Confusion Matrix below")
# print(confusion_matrix(y_test, y_pred))
# print("Accuracy DL8.5 on training set =", round(accuracy_score(y_train, clf.predict(X_train)), 4))
# print("Accuracy DL8.5 on test set =", round(accuracy_score(y_test, y_pred), 4), "\n\n\n")
#
# from sklearn.model_selection import GridSearchCV
#
#
# def custom_scoring(estimator, X, y):
#     print(type(estimator))
#     print(X.shape)
#     print(y.shape)
#     return estimator.n_features_
#
#
# def custom_scorer(y_true, y_pred):
#     return len(y_pred)
#
# # The scorers can be either be one of the predefined metric strings or a scorer
# # callable, like the one returned by make_scorer
# scoring = {'AB': custom_scoring, 'AUC': make_scorer(custom_scorer), 'Acc': 'accuracy'}
#
# # Setting refit='AUC', refits an estimator on the whole dataset with the
# # parameter setting that has the best cross-validated AUC score.
# # That estimator is made available at ``gs.best_estimator_`` along with
# # parameters like ``gs.best_score_``, ``gs.best_params_`` and
# # ``gs.best_index_``
# gs = GridSearchCV(DecisionTreeClassifier(random_state=42),
#                   param_grid={'min_samples_split': [2, 3, 4]},
#                   scoring=scoring, refit='Acc', return_train_score=True, cv=2)
# gs.fit(X, y)
# results = gs.cv_results_
# print(gs.cv_results_)

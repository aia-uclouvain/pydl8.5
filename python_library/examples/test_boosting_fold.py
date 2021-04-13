from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from dl85 import DL85Booster, MODEL_LP_RATSCH, MODEL_LP_DEMIRIZ, MODEL_QP_MDBOOST, DL85Classifier
import time
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from scipy import stats
import math
import matplotlib.pyplot as plt
from sklearn import metrics
import sys

filepath = "../datasets/pendigits.txt"
# filepath = "../datasets/anneal.txt"
dataset = np.genfromtxt(filepath, delimiter=' ')
# dataset = np.genfromtxt("../datasets/paper.txt", delimiter=' ')
X = dataset[:, 1:]
y = dataset[:, 0]
X = X.astype('int32')
y = y.astype('int32')
N_FOLDS = 5

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_trains, X_tests, y_trains, y_tests = [], [], [], []
train_indices, test_indices = [], []

max_train_size = 1000
max_for_kfold = max_train_size / ((N_FOLDS - 1) / N_FOLDS)
kf = None
if X.shape[0] <= max_for_kfold:
    kf = StratifiedKFold(n_splits=N_FOLDS)
else:
    kf = StratifiedShuffleSplit(n_splits=N_FOLDS, train_size=max_train_size, random_state=0)

for train_index, test_index in kf.split(X, y):
    # set training data and keep their index in initial data
    train_indices.append(train_index)
    X_trains.append(X[train_index])
    y_trains.append(y[train_index])
    # set test data and keep their index in initial data
    test_indices.append(test_index)
    X_tests.append(X[test_index])
    y_tests.append(y[test_index])

# for train_index, test_index in kf.split(X, y):
#     # for each fold, use 80% for training and 20% for testing
#     if X.shape[0] <= 1000:
#         # set training data and keep their index in initial data
#         train_indices.append(train_index)
#         X_trains.append(X[train_index])
#         y_trains.append(y[train_index])
#         # set test data and keep their index in initial data
#         test_indices.append(test_index)
#         X_tests.append(X[test_index])
#         y_tests.append(y[test_index])
#     # when the dataset size is greater than 1000, just keep randomly 800 instances from the initially 80%
#     # planned for training. add the remaining to the 20% planned for testing to build the real test set
#     else:
#         kk = StratifiedShuffleSplit(n_splits=2, train_size=800, random_state=0)
#         for tr_i, te_i in kk.split(X[train_index], y[train_index]):
#             # use the train_index list and tr_i index to retrieve the training index in the initial data
#             train_indices.append(train_index[tr_i])
#             X_trains.append(X[train_index[tr_i]])
#             y_trains.append(y[train_index[tr_i]])
#             # make like for training but add test_index not involved to get the new test set
#             test_indices.append(np.concatenate((train_index[te_i], test_index)))
#             X_tests.append(X[np.concatenate((train_index[te_i], test_index))])
#             y_tests.append(y[np.concatenate((train_index[te_i], test_index))])
#             break

# kf = StratifiedKFold(n_splits=5)
# for train_index, test_index in kf.split(X, y):
#     # for each fold, use 80% for training and 20% for testing
#     if X.shape[0] <= 1000:
#         # set training data and keep their index in initial data
#         train_indices.append(train_index)
#         X_trains.append(X[train_index])
#         y_trains.append(y[train_index])
#         # set test data and keep their index in initial data
#         test_indices.append(test_index)
#         X_tests.append(X[test_index])
#         y_tests.append(y[test_index])
#     # when the dataset size is greater than 1000, just keep randomly 800 instances from the initially 80%
#     # planned for training. add the remaining to the 20% planned for testing to build the real test set
#     else:
#         kk = StratifiedShuffleSplit(n_splits=2, train_size=800, random_state=0)
#         for tr_i, te_i in kk.split(X[train_index], y[train_index]):
#             # use the train_index list and tr_i index to retrieve the training index in the initial data
#             train_indices.append(train_index[tr_i])
#             X_trains.append(X[train_index[tr_i]])
#             y_trains.append(y[train_index[tr_i]])
#             # make like for training but add test_index not involved to get the new test set
#             test_indices.append(np.concatenate((train_index[te_i], test_index)))
#             X_tests.append(X[np.concatenate((train_index[te_i], test_index))])
#             y_tests.append(y[np.concatenate((train_index[te_i], test_index))])
#             break

# models = [MODEL_LP_DEMIRIZ, MODEL_LP_RATSCH, MODEL_QP_MDBOOST]
# models = [MODEL_LP_DEMIRIZ, MODEL_QP_MDBOOST]
models = [MODEL_LP_DEMIRIZ]
# reguls = [100, 100]
reguls = [100]
# reguls = [15] * len(models)
model_names = ['MODEL_LP_DEMIRIZ', 'MODEL_QP_MDBOOST']
# model_names = ['MODEL_LP_DEMIRIZ', 'MODEL_LP_RATSCH', 'MODEL_QP_MDBOOST']
depth = 1
max_iter = 0
k = 4
X_train, X_test, y_train, y_test = X_trains[k], X_tests[k], y_trains[k], y_tests[k]

print("######################################################################\n"
      "#                     DL8.5 boosting classifier                      #\n"
      "######################################################################")
print(filepath)
for i, mod in enumerate(models):
    print("<<=== Optiboost ===>>")
    # clf = DL85Booster(max_depth=depth, regulator=reguls[i], model=mod, verbose=False, quiet=False, max_iterations=max_iter)
    clf = DL85Booster(base_estimator=DecisionTreeClassifier(max_depth=depth), regulator=reguls[i], model=mod, verbose=False, quiet=False, max_iterations=max_iter)
    clf.fit(X_train, y_train)
    print("Number of trees =", clf.n_estimators_)
    y_pred = clf.predict(X_test)
    # print("Confusion Matrix below")
    # print(confusion_matrix(y_test, y_pred))
    # print(confusion_matrix(y_train, clf.predict(X_train)))
    print("Accuracy DL85Booster +", model_names[i], "on training set =", round(accuracy_score(y_train, clf.predict(X_train)), 4))
    print("Accuracy DL85Booster +", model_names[i], "on test set =", round(accuracy_score(y_test, y_pred), 4))
    print("AUC DL85Booster +", model_names[i], "on test set =", round(roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]), 4))
    predss = []
    for estim in clf.estimators_:
        predss.append([-1 if p != y_train[i] else 1 for i, p in enumerate(estim.predict(X_train))])
    predss = np.array(predss).transpose()
    clf.estimator_weights_ = [float(i)/sum(clf.estimator_weights_) for i in clf.estimator_weights_]
    margins_ = (predss @ np.array(clf.estimator_weights_).reshape(-1, 1)).transpose().tolist()[0]
    n_neg = len([marg for marg in margins_ if marg < 0])
    n_pos = len([marg for marg in margins_ if marg >= 0])
    print("len tree w", sum(clf.estimator_weights_), "w:", clf.estimator_weights_)
    print("margins sorted", sorted(margins_))
    margins_ = np.array(margins_)
    print("min:", margins_.min(), "max:", margins_.max(), "avg:", margins_.mean(), "std:", margins_.std(), "median:", np.median(margins_), "sum:", margins_.sum(), "var:", margins_.var())
    print("n_neg margins", n_neg, "n_pos margins", n_pos)
    print(clf.predict_proba(X_test))
    metrics.plot_roc_curve(clf, X_train, y_train)
    plt.show()
    # pred = sum((np.array(estimator.predict(X_test)) == np.array(y_test)).T * w
    #     for estimator, w in zip(clf.estimators_,
    #                             clf.estimator_weights_))
    # print(pred)
    # pred /= sum(clf.estimator_weights_)
    # pred[:, 0] *= -1
    # print(pred)
    # print(pred.sum(axis=1))
    # print("cons", len([x for x in margins_.tolist() if x <= margins_.max() / 10]))
    # print("conso", ((1 + margins_.min()) - (1 - margins_.max())))
    # print("conso", ((1 + margins_.min()) - (1 - margins_.max())) / margins_.std())
    # print("conssss", margins_.mean() - margins_.std())
    # print("conssss", margins_.mean() / margins_.std())
    # print("conssss", margins_.mean() - np.median(margins_))
    # print("conssss", margins_.mean() / np.median(margins_))
    # print("conssss", margins_.mean() + margins_.min())
    # print("conssss", margins_.mean() - margins_.var())
    # print("conssss", (margins_.mean() - margins_.var()) * (1 + (1 + margins_.min())))
    # print("conssss", (margins_.mean() ** .5 - margins_.var()) * (1 + (1 + margins_.min())))
    # print("conssss", margins_.mean() * (1 + (1 + margins_.min())) - margins_.var())
    # print("conssss", margins_.mean() / margins_.var())
    # print("conssss", margins_.mean() / margins_.var() * (1 + (1 + margins_.min())))
    # print(stats.describe(margins_))
    print("acc", n_pos / (n_pos + n_neg), "\n")
    print(X_train.shape)

# for n_estim in[5, 10, 20, 50, 100, 200, 500, 1000, 2000]:
for n_estim in []:
    clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=depth), n_estimators=n_estim, algorithm='SAMME')
    clf.fit(X_train, y_train)
    print("Number of trees =", len(clf.estimators_))
    y_pred = clf.predict(X_test)
    # print("Confusion Matrix below")
    # print(confusion_matrix(y_test, y_pred))
    # print(confusion_matrix(y_train, clf.predict(X_train)))
    print("Accuracy Adaboost on training set =", round(accuracy_score(y_train, clf.predict(X_train)), 4))
    print("Accuracy Adaboost on test set =", round(accuracy_score(y_test, y_pred), 4))
    predss = []
    for estim in clf.estimators_:
        predss.append([-1 if p != y_train[i] else 1 for i, p in enumerate(estim.predict(X_train))])
    predss = np.array(predss).transpose()
    clf.estimator_weights_ = [float(i)/sum(clf.estimator_weights_) for i in clf.estimator_weights_]
    margins_ = (predss @ np.array(clf.estimator_weights_).reshape(-1, 1)).transpose().tolist()[0]
    n_neg = len([marg for marg in margins_ if marg < 0])
    n_pos = len([marg for marg in margins_ if marg >= 0])
    print("len tree w", sum(clf.estimator_weights_), "w:", clf.estimator_weights_)
    print("margins sorted", sorted(margins_))
    margins_ = np.array(margins_)
    print("min:", margins_.min(), "max:", margins_.max(), "avg:", margins_.mean(), "std:", margins_.std(), "median:", np.median(margins_), "sum:", margins_.sum(), "var:", margins_.var())
    print("n_neg margins", n_neg, "n_pos margins", n_pos)
    # print("cons", len([x for x in margins_.tolist() if x <= margins_.max() / 10]))
    # print("conso", ((1 + margins_.min()) - (1 - margins_.max())))
    # print("conso", ((1 + margins_.min()) - (1 - margins_.max())) / margins_.std())
    # print("conssss", margins_.mean() - margins_.std())
    # print("conssss", margins_.mean() / margins_.std())
    # print("conssss", margins_.mean() - np.median(margins_))
    # print("conssss", margins_.mean() / np.median(margins_))
    # print("conssss", margins_.mean() + margins_.min())
    # print("conssss", margins_.mean() - margins_.var())
    # print("conssss", (margins_.mean() - margins_.var()) * (1 + (1 + margins_.min())))
    # print("conssss", (margins_.mean() ** 2 - margins_.var()) * (1 + (1 + margins_.min())))
    # print("conssss", margins_.mean() * (1 + (1 + margins_.min())) - margins_.var())
    # print("conssss", margins_.mean() / margins_.var())
    # print("conssss", margins_.mean() / margins_.var() * (1 + (1 + margins_.min())))
    print(stats.describe(margins_))
    print("acc", n_pos / (n_pos + n_neg), "\n")


# def nonzero(l):
#     return [i for i, j in enumerate(l) if j != 0]
#
# w = [0.7188892351922538, 0.7188892351923246, 0.7188892351922309, 0.7188892351922027, 0.7188892351922518, 0.7188892351922243, 0.7188892351922314, 0.7188892351922309, 0.7188892351922311, 0.7188892351922198, 1.3825308382537487, 2.737437859045444, 0.718889235192231, 0.7188892351922411, 0.718889235192224, 0.7188892351922309, 0.7188892351922311, 0.7188892351922254, 0.718889235192224, 0.7188892351922309, 0.7188892351922309, 0.7188892351922312, 0.7188892351922245, 0.7188892351922314, 0.7188892351922316, 0.7188892351922254, 2.0737962559838756, 0.7188892351922311, 0.7188892351922312, 0.7188892351922309, 2.73743785904537, 0.7188892351922307, 1.3825308382537935, 0.7188892351922307, 2.073796255983818, 0.7188892351922258, 0.7188892351922309, 0.7188892351922309, 0.7188892351921926, 0.7188892351922417, 0.7188892351922413, 0.7188892351922307, 0.7188892351922653, 0.7188892351922416, 0.7188892351922409, 0.7188892351922705, 0.7188892351921874, 0.7188892351922411, 0.7188892351922409, 0.7188892351922647, 0.7188892351922749, 0.7188892351922775, 0.718889235192282, 0.718889235192259, 1.382530838253792, 0.7188892351921857, 0.7188892351922775, 0.7188892351922707, 0.7188892351921925, 0.7188892351921918, 0.718889235192282, 0.7188892351922751, 0.718889235192192, 2.0737962559838374, 0.7188892351921952, 0.7188892351922712, 0.7188892351921871, 2.0737962559838383, 0.7188892351922485, 0.7188892351922705, 0.7188892351922713, 0.7188892351921872, 0.7188892351922473, 0.7188892351921924, 0.7188892351922297, 0.7188892351922479, 2.7374378590454618, 0.7188892351922329, 1.3825308382538033, 0.7188892351921877, 0.7188892351922411, 2.0737962559838232, 0.7188892351922401, 2.737437859045496, 0.7188892351921865, 0.7188892351922297, 1.3825308382537922, 2.073796255983898, 0.7188892351922316, 0.7188892351922238, 1.3825308382537953, 1.3825308382538057, 1.382530838253789, 0.7188892351921924, 1.3825308382537935, 0.7188892351922394, 1.3825308382537496, 0.7188892351922057, 2.7374378590454507, 0.7188892351922833, 0.7188892351922739, 0.718889235192198, 1.382530838253731, 0.7188892351921916, 2.0737962559838197, 0.7188892351921735, 0.7188892351921612, 2.0737962559838987, 2.737437859045399, 1.3825308382537473]
#
# # clf = DecisionTreeClassifier(max_depth=1)
# clf = DL85Classifier(max_depth=1)
# start = time.perf_counter()
# print("Model building...")
# clf.fit(X_train, y_train, sample_weight=w)
# pred = np.array([-1 if p != y_train[i] else 1 for i, p in enumerate(clf.predict(X_train))])
# print("p@w", pred @ w)
# print("pred sum", pred.sum())
# print("sum res", sum(clf.predict(X_train)))
# duration = time.perf_counter() - start
# print("Model built. Duration of building =", round(duration, 4))
# y_pred = clf.predict(X_test)
# print("Confusion Matrix below")
# print(confusion_matrix(y_train, clf.predict(X_train)))
# print("Accuracy DL8.5 opti on real training set =", round(accuracy_score(y_train[nonzero(w)], clf.predict(X_train[nonzero(w)])), 4))
# print("Accuracy DL8.5 opti on training set =", round(accuracy_score(y_train, clf.predict(X_train)), 4))
# print("Accuracy DL8.5 opti on test set =", round(accuracy_score(y_test, y_pred), 4), "\n\n\n")
#
#
# clf = DecisionTreeClassifier(max_depth=1)
# # clf = DL85Classifier(max_depth=1)
# start = time.perf_counter()
# print("Model building...")
# clf.fit(X_train, y_train, sample_weight=w)
# pred = np.array([-1 if p != y_train[i] else 1 for i, p in enumerate(clf.predict(X_train))])
# print("p@w", pred @ w)
# print("pred sum", pred.sum())
# print("sum res", sum(clf.predict(X_train)))
# duration = time.perf_counter() - start
# print("Model built. Duration of building =", round(duration, 4))
# y_pred = clf.predict(X_test)
# print("Confusion Matrix below")
# print(confusion_matrix(y_train, clf.predict(X_train)))
# print("Accuracy cart opti on real training set =", round(accuracy_score(y_train[nonzero(w)], clf.predict(X_train[nonzero(w)])), 4))
# print("Accuracy cart opti on training set =", round(accuracy_score(y_train, clf.predict(X_train)), 4))
# print("Accuracy cart opti on test set =", round(accuracy_score(y_test, y_pred), 4), "\n\n\n")

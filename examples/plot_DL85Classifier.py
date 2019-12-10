"""
===============================
Usage example of DL85Classifier
===============================

"""
import numpy as np
import sys
sys.path.insert(0, "../")
from dl85 import DL85Classifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import time

dataset = np.genfromtxt("../datasets/anneal.txt", delimiter=' ')
X = dataset[:, 1:]
y = dataset[:, 0]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

classes, supports = np.unique(y, return_counts=True)
class_support = dict(zip(classes, supports))


def error_function(original_targets, original_class_support):
    def func(tid_iterator):
        tid_iterator.init_iterator()
        size = tid_iterator.get_size()

        tid_list = []
        for i in range(size):
            tid_list.append(tid_iterator.get_value())
            if i != size - 1:
                tid_iterator.inc_iterator()

        target_subset = original_targets.take(tid_list)
        classes, supports = np.unique(target_subset, return_counts=True)
        class_support = dict(zip(classes, supports))
        maxclass = -1
        maxclassval = minclassval = -1
        conflict = 0

        for classe, sup in class_support.items():
            if sup > maxclassval:
                maxclass = classe
                maxclassval = sup
            elif sup == maxclassval:
                conflict += 1
                if original_class_support[classe] > original_class_support[maxclass]:
                    maxclass = classe
            else:
                minclassval = sup

        error_score = sum(supports) - maxclassval
        return [error_score, maxclass, conflict, minclassval]
    return func


clf = DL85Classifier(max_depth=3, print_output=True, error_function=error_function(y, class_support))
# clf = DL85Classifier(max_depth=3, print_output=True)
# clf.fit(X_train, y_train)
clf.fit(X, y)


# print("DL8.5 iterative in python")
# start = time.perf_counter()
# error = 0
# clf = None
# for i in range(1, 4):
#     clf = DL85Classifier(max_depth=i, max_error=error)
#     clf.fit(X_train, y_train)
#     error = clf.error_
# duration = time.perf_counter() - start
# print("Duration of model building =", duration)
# y_pred1 = clf.predict(X_test)
# print(confusion_matrix(y_test, y_pred1))
# score1 = accuracy_score(y_test, y_pred1)
# print("Accuracy DL8.5 on test set =", score1, "\n")
#
#
# print("DL8.5 iterative in c++")
# clf1 = DL85Classifier(max_depth=3, iterative=True)
# start = time.perf_counter()
# print("Model building...")
# clf1.fit(X_train, y_train)
# duration = time.perf_counter() - start
# print("Duration of model building =", duration)
# y_pred1 = clf1.predict(X_test)
# print(confusion_matrix(y_test, y_pred1))
# score1 = accuracy_score(y_test, y_pred1)
# print("Accuracy DL8.5 on test set =", score1, "\n")
#
#
# print("Default DT")
# clf2 = DecisionTreeClassifier(max_depth=3)
# start = time.perf_counter()
# print("Model building...")
# clf2.fit(X_train, y_train)
# duration = time.perf_counter() - start
# print("Duration of model building =", duration)
# y_pred2 = clf2.predict(X_test)
# print(confusion_matrix(y_test, y_pred2))
# score2 = accuracy_score(y_test, y_pred2)
# print("Accuracy default DT on test set =", score2, "\n")
#
#
# clf3 = DL85Classifier(max_depth=3)
# scores = cross_val_score(clf3, X, y, cv=5)
# print(scores)
#
#
# clf4 = DecisionTreeClassifier(max_depth=3)
# scores = cross_val_score(clf4, X, y, cv=5)
# print(scores)
#
#
# kf = KFold(n_splits=5, random_state=42, shuffle=True)
# accuracies = []
# for train_index, test_index in kf.split(X):
#     data_train = X[train_index]
#     target_train = y[train_index]
#     data_test = X[test_index]
#     target_test = y[test_index]
#
#     clf = DL85Classifier()
#     clf.fit(data_train, target_train)
#
#     preds = clf.predict(data_test)
#
#     # accuracy for the current fold only
#     accuracy = accuracy_score(target_test,preds)
#     accuracies.append(accuracy)
#
# # this is the average accuracy over all folds
# average_accuracy = np.mean(accuracies)
# print("Average accuracy =", average_accuracy)

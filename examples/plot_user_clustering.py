"""
===============================
Usage example of DL85Cluster
===============================

"""
import numpy as np
import sys
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
sys.path.insert(0, "../")
from dl85 import DL85Cluster

dataset = np.genfromtxt("../datasets/anneal.txt", delimiter=' ')
X = dataset[:, 1:]
y = dataset[:, 0]
X = X.astype('int32')
y = y.astype('int32')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


print("############################################################################################\n"
      "#      DL8.5 clustering : user specific error function and leaves' values assignation      #\n"
      "############################################################################################")


# user error function
def node_error(y):

    def funct(tid_iterator):
        tid_iterator.init_iterator()
        size = tid_iterator.get_size()

        tid_list = []
        for i in range(size):
            tid_list.append(tid_iterator.get_value())
            if i != size - 1:
                tid_iterator.inc_iterator()

        target_subset = y.take(tid_list)
        classes, supports = np.unique(target_subset, return_counts=True)
        subset_class_support = dict(zip(classes, supports))
        maxclass = maxclassval = -1

        for classe, sup in subset_class_support.items():
            if sup > maxclassval:
                maxclass = classe
                maxclassval = sup

        error_score = sum(supports) - maxclassval
        return error_score, maxclass
    return funct


def leaf_value(y):

    def funct(tid_list):
        target_subset = y.take(tid_list)
        classes, supports = np.unique(target_subset, return_counts=True)
        class_support = dict(zip(classes, supports))
        class_support = {k: v for k, v in sorted(class_support.items(), key=lambda item: item[1], reverse=True)}
        return list(class_support.keys())[0]
    return funct


clf = DL85Cluster(max_depth=2, error_function=node_error(y_train), leaf_value_function=leaf_value(y_train))
start = time.perf_counter()
print("Model building...")
clf.fit(X_train)
duration = time.perf_counter() - start
print("Model built. Duration of building =", round(duration, 4))
y_pred1 = clf.predict(X_test)
print("Confusion Matrix below")
print(confusion_matrix(y_test, y_pred1))
print("Accuracy DL8.5 on training set =", round(clf.accuracy_, 4))
print("Accuracy DL8.5 on test set =", round(accuracy_score(y_test, y_pred1), 4), "\n\n\n")


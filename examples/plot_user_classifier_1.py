"""
===============================
Usage example of DL85Classifier
===============================

"""
import numpy as np
import sys
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
sys.path.insert(0, "../")
from dl85 import DL85Classifier

dataset = np.genfromtxt("../datasets/anneal.txt", delimiter=' ')
X = dataset[:, 1:]
y = dataset[:, 0]
X = X.astype('int32')
y = y.astype('int32')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


print("##############################################################\n"
      "#      DL8.5 classifier : user specific error function       #\n"
      "##############################################################")


# user error function
def node_error(supports_iterator):
    supports_iterator.init_iterator()
    nclasses = supports_iterator.get_size()
    current_class_support = supports_iterator.get_value()

    maxclass = som = 0
    maxclassval = current_class_support
    som += current_class_support

    for i in range(1, nclasses):
        supports_iterator.inc_iterator()
        current_class_support = supports_iterator.get_value()
        if current_class_support > maxclassval:
            maxclassval = current_class_support
            maxclass = i
        som += current_class_support

    error_score = som - maxclassval
    return error_score, maxclass


clf = DL85Classifier(max_depth=2, fast_error_function=node_error)
start = time.perf_counter()
print("Model building...")
clf.fit(X_train, y_train)
duration = time.perf_counter() - start
print("Model built. Duration of building =", round(duration, 4))
y_pred1 = clf.predict(X_test)
print("Confusion Matrix below")
print(confusion_matrix(y_test, y_pred1))
print("Accuracy DL8.5 on training set =", round(clf.accuracy_, 4))
print("Accuracy DL8.5 on test set =", round(accuracy_score(y_test, y_pred1), 4), "\n\n\n")

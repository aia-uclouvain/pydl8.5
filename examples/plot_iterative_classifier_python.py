"""
===============================================
DL8.5 classifier : python side iterative search
===============================================

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
      "#      DL8.5 classifier : python side iterative search       #\n"
      "##############################################################")
start = time.perf_counter()
error = 0  # default max error value expressing no bound
clf = None
for i in range(1, 4):  # max depth = 3
    clf = DL85Classifier(max_depth=i, max_error=error)
    clf.fit(X_train, y_train)
    error = clf.error_
duration = time.perf_counter() - start
print("Model built. Duration of building =", round(duration, 4))
y_pred = clf.predict(X_test)
print("Confusion Matrix below")
print(confusion_matrix(y_test, y_pred))
print("Accuracy DL8.5 on training set =", round(clf.accuracy_, 4))
print("Accuracy DL8.5 on test set =", round(accuracy_score(y_test, y_pred), 4), "\n\n\n")

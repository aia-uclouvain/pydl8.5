"""
===============================================
DL8.5 classifier : python side iterative search
===============================================
Iterative search is the idea that the algorithm starts with finding an optimal 
shallow tree, and then uses the quality of this tree to bound the quality of 
deeper trees. This class shows how to perform this type of search by repeatedly 
calling the DL8.5 algorithm. A second implementation is illustrated in 
plot_classifier_iterative_c_plus.py, and uses C++.
"""
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from pydl85 import DL85Classifier

dataset = np.genfromtxt("../datasets/anneal.txt", delimiter=' ')
X, y = dataset[:, 1:], dataset[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


print("###########################################################################\n"
      "#      DL8.5 default classifier using python-based iterative search       #\n"
      "###########################################################################")
start = time.perf_counter()
error = 0  # default max error value expressing no bound
clf = None
remaining_time = 600
for i in range(1, 3):  # max depth = 2
    clf = DL85Classifier(max_depth=i, max_error=error, time_limit=remaining_time)
    clf.fit(X_train, y_train)
    error = clf.error_
    remaining_time -= clf.runtime_
duration = time.perf_counter() - start
print("Model built. Duration of building =", round(duration, 4))
y_pred = clf.predict(X_test)
print("Confusion Matrix below")
print(confusion_matrix(y_test, y_pred))
print("Accuracy DL8.5 on training set =", round(clf.accuracy_, 4))
print("Accuracy DL8.5 on test set =", round(accuracy_score(y_test, y_pred), 4))

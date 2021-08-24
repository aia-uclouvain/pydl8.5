"""
==========================================
DL8.5 classifier : native iterative search
==========================================
Iterative search is the idea that the algorithm starts with finding an optimal 
shallow tree, and then uses the quality of this tree to bound the quality of 
deeper trees. This class shows how to perform this type of search by using 'iterative'
option of the DL85Classifier class. In this case, an implementation in C++ is used.
A second implementation is shown in the plot_classifier_iterative_python.py class.
"""
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from dl85 import DL85Classifier

dataset = np.genfromtxt("../../datasets/anneal.txt", delimiter=' ')
X = dataset[:, 1:]
y = dataset[:, 0]
X = X.astype('int32')
y = y.astype('int32')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


print("##########################################################################\n"
      "#            DL8.5 default classifier using iterative search             #\n"
      "##########################################################################")
print("!!! This code is not currently supported !!!")
clf = DL85Classifier(max_depth=2, iterative=True, time_limit=600)
start = time.perf_counter()
print("Model building...")
clf.fit(X_train, y_train)
duration = time.perf_counter() - start
print("Model built. Duration of building =", round(duration, 4))
y_pred = clf.predict(X_test)
print("Confusion Matrix below")
print(confusion_matrix(y_test, y_pred))
print("Accuracy DL8.5 on training set =", round(clf.accuracy_, 4))
print("Accuracy DL8.5 on test set =", round(accuracy_score(y_test, y_pred), 4))

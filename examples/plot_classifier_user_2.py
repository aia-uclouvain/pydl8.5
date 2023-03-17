"""
========================================================================
DL8.5 classifier : user specific error function based on transactions ID
========================================================================
PyDL8.5 allows users to write their own error function. This example shows how 
to write an error function based on transaction identifiers. PyDL8.5 will determine
these transaction identifiers based on the occurrences of an itemset in the
training data. 

The error function is called very often, and calculating an error score based
on tids can be time consuming. For classification tasks, it is highly recommended
not to write an error function in Python that traverses the tids. 
check the plot_classifier_user_1.py example for a more efficient user-written
error function in classification settings.
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


print("########################################################################################\n"
      "#      DL8.5 classifier : user specific error function based on transactions ids       #\n"
      "########################################################################################")


# return the error and the majority class
def error(tids, y):
    classes, supports = np.unique(y.take(list(tids)), return_counts=True)
    maxindex = np.argmax(supports)
    return sum(supports) - supports[maxindex], classes[maxindex]


clf = DL85Classifier(max_depth=2, error_function=lambda tids: error(tids, y_train), time_limit=600)
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

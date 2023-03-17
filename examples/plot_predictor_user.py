"""
=============================================
DL8.5 Predictor used for task involving label
=============================================
PyDL8.5 allows users to write their own error function. This example shows how 
to write an error function based on transaction identifiers. PyDL8.5 will determine
these transaction identifiers (tids) based on the occurrences of an itemset in the
training data. 

The error function is called very often, and calculating an error score based
on tids can be time consuming. For classification tasks, it is highly recommended
not to write an error function in Python that operators on lists of tids. 
check the plot_classifier_user_1.py example for a more efficient user-written
error function in classification settings.

Moreover, this example shows how to use the DL85Predictor class. Using this
class, the labels do not need to be passed to DL8.5. For classification tasks,
the error function can also be specififed as a parameter of the DL85Classifier
class. In this case, a standard implementation is used for filling in the
class labels for leafs in the tree. 

Another example of a user-specified error function is given in plot_cluster_user.py.
"""
import numpy as np
from sklearn.model_selection import train_test_split
import time
from pydl85 import DL85Predictor

dataset = np.genfromtxt("../datasets/anneal.txt", delimiter=' ')
X, y = dataset[:, 1:], dataset[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


print("#####################################################################################\n"
      "#      DL8.5 Predictor used for classification : user specific error function       #\n"
      "#####################################################################################")


# return the error and the majority class
def error(tids):
    classes, supports = np.unique(y_train.take(list(tids)), return_counts=True)
    maxindex = np.argmax(supports)
    return sum(supports) - supports[maxindex], classes[maxindex]


clf = DL85Predictor(max_depth=2, error_function=error, time_limit=600)
start = time.perf_counter()
print("Model building...")
clf.fit(X_train)
duration = time.perf_counter() - start
print("Model built. Duration of building =", round(duration, 4))
print("Accuracy DL8.5 on training set =", round(clf.accuracy_, 4))
"""
===========================================
DL8.5 used to perform predictive clustering
===========================================
This example illustrates how to use a user-specified error function to perform predictive
clustering. The PyDL8.5 library also provides an implementation of predictive clustering
that does not require the use of user-specified error function. 
Check the DL85Cluster class for this implementation.

The main purpose of this example is to show how users of the library can implement their
own decision tree learning task using PyDL8.5's interface for writing error functions.
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import DistanceMetric
import time
from pydl85 import DL85Predictor

dataset = np.genfromtxt("../datasets/anneal.txt", delimiter=' ')
X = dataset[:, 1:]
X_train, X_test = train_test_split(X, test_size=0.2, random_state=0)


print("############################################################################################\n"
      "#      DL8.5 clustering : user specific error function and leaves' values assignment       #\n"
      "############################################################################################")

# The quality of every cluster is determined using the Euclidean distance.
eucl_dist = DistanceMetric.get_metric('euclidean')


# user error function
def error(tids):
    # collect the complete examples identified using the tids. 
    X_subset = X_train[list(tids), :]
    # determine the centroid of the cluster
    centroid = np.mean(X_subset, axis=0)
    # calculate the distances towards centroid
    distances = eucl_dist.pairwise(X_subset, [centroid])
    # return the sum of distances as the error
    return float(sum(distances))


# user leaf assignment
def leaf_value(tids):
    # The prediction for every leaf is the centroid of the cluster
    return np.mean(X.take(list(tids)))


# Change the parameters of the algorithm as desired.
clf = DL85Predictor(max_depth=2, min_sup=5, error_function=error, leaf_value_function=leaf_value, time_limit=600)

start = time.perf_counter()
print("Model building...")
clf.fit(X_train)
duration = time.perf_counter() - start
print("Model built. Duration of the search =", round(duration, 4))
predicted = clf.predict(X_test)



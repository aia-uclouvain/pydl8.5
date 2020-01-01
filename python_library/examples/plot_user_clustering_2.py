"""
==============================================================================
DL8.5 clustering : user specific error function and leaves' values assignation
==============================================================================

"""
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.neighbors import DistanceMetric
import time
sys.path.insert(0, "../")
from dl85 import DL85Cluster

dataset = np.genfromtxt("../datasets/anneal.txt", delimiter=' ')
X = dataset[:, 1:]
X = X.astype('int32')

X_train, X_test = train_test_split(X, test_size=0.2, random_state=0)


print("############################################################################################\n"
      "#      DL8.5 clustering : user specific error function and leaves' values assignation      #\n"
      "############################################################################################")


# user error function
def error(tids, X):
    dist = DistanceMetric.get_metric('euclidean')
    X_subset = np.asarray([X[index, :] for index in list(tids)], dtype='int32')
    centroid = np.mean(X_subset, axis=0).reshape(1, X_subset.shape[1])
    distances = [dist.pairwise(instance.reshape(1, X_subset.shape[1]), centroid)[0, 0] for instance in X_subset]
    return round(sum(distances), 4)


# user leaf assignation
def leaf_value(tids, X):
    return np.mean(X.take(list(tids)))


clf = DL85Cluster(max_depth=1, error_function=lambda tids: error(tids, X_train), leaf_value_function=lambda tids: leaf_value(tids, X_train), time_limit=600)
start = time.perf_counter()
print("Model building...")
clf.fit(X_train)
duration = time.perf_counter() - start
print("Model built. Duration of building =", round(duration, 4))
predicted = clf.predict(X_test)


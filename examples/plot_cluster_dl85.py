"""
===================================
DL8.5 default predictive clustering
===================================
This example illustrates how to use the DL85Cluster class for predictive clustering.
A second implementation of predictive clustering is provided in the plot_cluster_user.py
example. 
"""
import numpy as np
from sklearn.model_selection import train_test_split
import time
from pydl85 import DL85Cluster

dataset = np.genfromtxt("../datasets/anneal.txt", delimiter=' ')
X = dataset[:, 1:]
X_train, X_test = train_test_split(X, test_size=0.2, random_state=0)


print("####################################################################\n"
      "#                      DL8.5 default clustering                    #\n"
      "####################################################################")
clf = DL85Cluster(max_depth=2, time_limit=600)
start = time.perf_counter()
print("Model building...")
clf.fit(X_train)
duration = time.perf_counter() - start
print("Model built. Duration of building =", round(duration, 4), "\n\n\n")
predicted = clf.predict(X_test)


print("####################################################################\n"
      "#                DL8.5 default predictive clustering               #\n"
      "####################################################################")
X_desc = X_train[:X_test.shape[0], :]
X_err = X_test
clf = DL85Cluster(max_depth=2, time_limit=600)
start = time.perf_counter()
print("Model building...")
clf.fit(X_desc, X_err)
duration = time.perf_counter() - start
print("Model built. Duration of building =", round(duration, 4))
print("X_desc is used to describe data while X_err is used to compute errors")
predicted = clf.predict(X_err)


"""
===================================
DL8.5 default predictive clustering
===================================

"""
import numpy as np
from sklearn.model_selection import train_test_split
import time
from dl85 import DL85Cluster

dataset = np.genfromtxt("../datasets/anneal.txt", delimiter=' ')
X = dataset[:, 1:]
X = X.astype('int32')

X_train, X_test = train_test_split(X, test_size=0.2, random_state=0)


print("####################################################################\n"
      "#                      DL8.5 default clustering                    #\n"
      "####################################################################")
clf = DL85Cluster(max_depth=1, time_limit=600)
start = time.perf_counter()
print("Model building...")
clf.fit(X_train)
duration = time.perf_counter() - start
print("Model built. Duration of building =", round(duration, 4), "\n\n\n")
predicted = clf.predict(X_test)


print("####################################################################\n"
      "#                DL8.5 default predictive clustering               #\n"
      "####################################################################")
X_train1 = X_train[:X_test.shape[0], :]
clf = DL85Cluster(max_depth=1, time_limit=600)
start = time.perf_counter()
print("Model building...")
clf.fit(X_train1, X_test)
duration = time.perf_counter() - start
print("Model built. Duration of building =", round(duration, 4))
print("Xtrain1 is used to describe data while X_test is used to compute errors")
predicted = clf.predict(X_test)


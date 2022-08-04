"""
=============================
Memory bounded DL85Classifier
=============================

"""
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pydl85 import DL85Classifier, Cache_Type, Wipe_Type
import time

dataset = np.genfromtxt("../datasets/anneal.txt", delimiter=' ')
X = dataset[:, 1:]
y = dataset[:, 0]
X = X.astype('int32')
y = y.astype('int32')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


print("######################################################################\n"
      "#                   DL8.5 memory bounded classifier                  #\n"
      "######################################################################")
clf = DL85Classifier(max_depth=4, cache_type=Cache_Type.Cache_TrieItemset, maxcachesize=5000, wipe_factor=0.4, wipe_type=Wipe_Type.Reuses)
start = time.perf_counter()
print("Model building...")
clf.fit(X, y)
duration = time.perf_counter() - start
print("Model built. Duration of building =", round(duration, 4))
y_pred = clf.predict(X_test)
print("Confusion Matrix below")
print(confusion_matrix(y_test, y_pred))
print("Accuracy DL8.5 on training set =", round(clf.accuracy_, 4))
print("Accuracy DL8.5 on test set =", round(accuracy_score(y_test, y_pred), 4), "\n\n\n")


"""
========================================
Default DL85Classifier on COMPAS dataset
========================================

"""
import time
import graphviz
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from pydl85 import DL85Classifier, Cache_Type

dataset = np.genfromtxt("../datasets/compas.csv", delimiter=',', skip_header=1)
X, y = dataset[:, :-1], dataset[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# read the column names
with open("../datasets/compas.csv", 'r') as f:
    col_names = f.readline().strip().split(',')
    col_names = col_names[:-1]


print("######################################################################\n"
      "#                      DL8.5 default classifier                      #\n"
      "######################################################################")
clf = DL85Classifier(max_depth=4, cache_type=Cache_Type.Cache_HashCover)
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


# print the tree
print("Serialized json tree:", clf.tree_)
dot = clf.export_graphviz(feature_names=col_names, class_names=["No Recidivism", "Recidivism"])
graph = graphviz.Source(dot, format="png")
graph.render("plots/compas_odt")

"""
==================================
Default DL85Classifier export tree
==================================

"""
import graphviz
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from pydl85 import DL85Classifier


print("######################################################################\n"
      "#                      DL8.5 default classifier                      #\n"
      "######################################################################")

# read the dataset and split into features and targets
dataset = np.genfromtxt("../datasets/anneal.txt", delimiter=' ')
X, y = dataset[:, 1:], dataset[:, 0]
# split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

clf = DL85Classifier(max_depth=2, min_sup=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# show results
print("Model built in", round(clf.runtime_, 4), "seconds")
print("Confusion Matrix below\n", confusion_matrix(y_test, y_pred))
print("Accuracy on training set =", round(clf.accuracy_, 4))
print("Accuracy on test set =", round(accuracy_score(y_test, y_pred), 4))

# print the tree
print("Serialized json tree:", clf.tree_)
dot = clf.export_graphviz()
# uncomment the following lines to save the tree as a png file
# graph = graphviz.Source(dot, format="png")
# graph.render("plots/anneal_odt")


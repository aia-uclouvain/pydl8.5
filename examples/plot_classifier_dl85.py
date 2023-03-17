"""
======================
Default DL85Classifier
======================

"""
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import time
from pydl85 import DL85Classifier

dataset = np.genfromtxt("../datasets/anneal.txt", delimiter=' ')
X, y = dataset[:, 1:], dataset[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


print("######################################################################\n"
      "#                      DL8.5 default classifier                      #\n"
      "######################################################################")
clf = DL85Classifier(max_depth=2, time_limit=600, desc=True)
start = time.perf_counter()
print("Model building...")
clf.fit(X_train, y_train)
duration = time.perf_counter() - start
print("Model built. Duration of building =", round(duration, 4))
y_pred = clf.predict(X_test)
print("Confusion Matrix below")
print(confusion_matrix(y_test, y_pred))
print("Accuracy DL8.5 on training set =", round(clf.accuracy_, 4))
print("Accuracy DL8.5 on test set =", round(accuracy_score(y_test, y_pred), 4), "\n\n\n")


print("##############################################################\n"
      "#     DL8.5 classifier : Manual cross-validation (5-fold)    #\n"
      "##############################################################")
kf = KFold(n_splits=5, random_state=42, shuffle=True)
training_accuracies = []
test_accuracies = []
start = time.perf_counter()
print("Model building...")
for train_index, test_index in kf.split(X):
    data_train = X[train_index]
    target_train = y[train_index]
    data_test = X[test_index]
    target_test = y[test_index]
    clf = DL85Classifier(max_depth=2, time_limit=600)
    clf.fit(data_train, target_train)
    preds = clf.predict(data_test)
    training_accuracies.append(clf.accuracy_)
    test_accuracies.append(accuracy_score(target_test, preds))
duration = time.perf_counter() - start
print("Model built. Duration of building =", round(duration, 4))
print("Average accuracy on training set =", round(np.mean(training_accuracies), 4))
print("Average accuracy on test set =", round(np.mean(test_accuracies), 4), "\n\n\n")


print("##############################################################\n"
      "#   DL8.5 classifier : Automatic cross-validation (5-fold)   #\n"
      "##############################################################")
clf = DL85Classifier(max_depth=2, time_limit=600)
start = time.perf_counter()
print("Model building...")
scores = cross_val_score(clf, X, y, cv=5)
duration = time.perf_counter() - start
print("Model built. Duration of building =", round(duration, 4))
print("Average accuracy on test set =", round(np.mean(scores), 4))

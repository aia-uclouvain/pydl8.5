from sklearn.svm import SVC
from sklearn.datasets import load_iris, load_digits, fetch_openml
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelBinarizer, KBinsDiscretizer, Binarizer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.utils import check_random_state
from dl85 import DL85Boostera
import time
import numpy as np
from sklearn.metrics import confusion_matrix
import subprocess

depth, time_limit, N_FOLDS = 2, 0, 5
pr = subprocess.Popen("pwd", stderr=subprocess.PIPE, stdout=subprocess.PIPE)
print(pr.stderr.read().decode("utf-8"))
print(pr.stdout.read().decode("utf-8"))

dataset = np.genfromtxt("../datasets/boosting/mm/clean_mm_1.csv", delimiter=",", skip_header=1)
X = dataset[:, :-1]
y = dataset[:, -1]

# dataset = np.genfromtxt("../datasets/paper_test.txt", delimiter=" ")
# # dataset = np.genfromtxt("../datasets/paper.txt", delimiter=" ")
# X = dataset[:, 1:]
# y = dataset[:, 0]

enc = KBinsDiscretizer(n_bins=8, encode='ordinal', strategy='uniform')
X = enc.fit_transform(X)
enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
X = enc.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=0)
clf = DL85Boostera(max_depth=depth, min_sup=1, max_iterations=100000, time_limit=time_limit, quiet=False)
start = time.perf_counter()
print("Model building...")
# clf.fit(X, y)
clf.fit(X_train, y_train)
duration = time.perf_counter() - start
print("Model built. Duration of building =", round(duration, 4))
y_pred = clf.predict(X_test)
print("Confusion Matrix below")
print(confusion_matrix(y_test, y_pred))
print("Accuracy DL8.5 on training set =", round(clf.accuracy_, 4))
print("Accuracy DL8.5 on test set =", round(accuracy_score(y_test, y_pred), 4))
print(clf.problem)
print(clf.regulator, clf.n_estimators_)
for c in clf.estimators_:
    print(c.tree_)

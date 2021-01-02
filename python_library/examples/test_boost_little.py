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

depth, time_limit, N_FOLDS = 1, 0, 5

dataset = np.genfromtxt("../datasets/paper_test.txt", delimiter=" ")
X, y = dataset[:, 1:], dataset[:, 0]

clf = DL85Boostera(max_depth=depth, regulator=.5, quiet=False, gamma='auto')
start = time.perf_counter()
print("Model building...")
clf.fit(X, y)
print("Accuracy DL8.5 on training set =", round(clf.accuracy_, 4))

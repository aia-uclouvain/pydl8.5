from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from dl85 import DL85Booster, DL85Classifier, MODEL_LP_RATSCH, MODEL_LP_DEMIRIZ, MODEL_QP_MDBOOST
import time
import numpy as np
from sklearn.metrics import confusion_matrix

dataset = np.genfromtxt("../datasets/german-credit.txt", delimiter=' ')
X = dataset[:, 1:]
y = dataset[:, 0]
X = X.astype('int32')
y = y.astype('int32')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

models = [MODEL_LP_DEMIRIZ, MODEL_LP_RATSCH, MODEL_QP_MDBOOST]
reguls = [15] * len(models)
model_names = ['MODEL_LP_DEMIRIZ', 'MODEL_LP_RATSCH', 'MODEL_QP_MDBOOST']
depth = 2
n_trees = 100
percent_data = .4
percent_features = .6
reuse_data = True
reuse_features = True
use_test_error = False
keep_trees = False
seed = 0

print("######################################################################\n"
      "#                     DL8.5 boosting classifier                      #\n"
      "######################################################################")
clf = DL85Classifier(max_depth=depth)
start = time.perf_counter()
print("Model building...")
clf.fit(X_train, y_train)
duration = time.perf_counter() - start
print("Model built. Duration of building =", round(duration, 4))
y_pred = clf.predict(X_test)
print("Confusion Matrix below")
print(confusion_matrix(y_test, y_pred))
print("Accuracy DL8.5 on training set =", round(clf.accuracy_, 4))
print("Accuracy DL8.5 on test set =", round(accuracy_score(y_test, y_pred), 4), "\n\n")

for base_clf, name in zip([DL85Classifier(max_depth=depth), DecisionTreeClassifier(max_depth=depth)],
                          ["DL8.5", "CART"]):
    clf = BaggingClassifier(base_estimator=base_clf,
                            n_estimators=n_trees,
                            max_samples=percent_data,
                            max_features=percent_features,
                            bootstrap=reuse_data,
                            bootstrap_features=reuse_features,
                            oob_score=use_test_error,
                            warm_start=keep_trees,
                            n_jobs=-1,
                            random_state=seed)
    start = time.perf_counter()
    print("Model building...")
    clf.fit(X, y)
    duration = time.perf_counter() - start
    print("Model built. Duration of building =", round(duration, 4))
    y_pred = clf.predict(X_test)
    print("Confusion Matrix below")
    print(confusion_matrix(y_test, y_pred))
    print("Accuracy bagging {} on training set = {}".format(name, round(accuracy_score(y_train, clf.predict(X_train)), 4)))
    print("Accuracy bagging {} on test set = {}".format(name, round(accuracy_score(y_test, y_pred), 4)))
    print("\n\n")


clf = RandomForestClassifier(max_depth=depth, n_estimators=n_trees, random_state=seed,
                             # max_features=percent_features, bootstrap=reuse_data, oob_score=use_test_error, warm_start=keep_trees, max_samples=percent_data
                             )
start = time.perf_counter()
print("Model building...")
clf.fit(X_train, y_train)
duration = time.perf_counter() - start
print("Model built. Duration of building =", round(duration, 4))
y_pred = clf.predict(X_test)
print("Confusion Matrix below")
print(confusion_matrix(y_test, y_pred))
print("Accuracy RF on training set =", round(accuracy_score(y_train, clf.predict(X_train)), 4))
print("Accuracy RF on test set =", round(accuracy_score(y_test, y_pred), 4), "\n\n\n")

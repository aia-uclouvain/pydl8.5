from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import numpy as np

print(__doc__)

# Loading the Digits dataset
digits = datasets.load_digits()

# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))
y = digits.target

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0)

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

scores = ['precision', 'recall']

print("# Tuning hyper-parameters for auc")
print()

clf = GridSearchCV(
    SVC(), tuned_parameters, scoring='precision_macro', refit=False
)
clf.fit(X_train, y_train)

print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
rank = clf.cv_results_['rank_test_score']
print(type(rank))
# print(rank.shape)
print(rank)


def sortt(l):
    return [l[i] for i in np.argsort(rank)]


for mean, std, params in zip(sortt(means), sortt(stds), sortt(clf.cv_results_['params'])):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()
print()


for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()

"""
=================================
DL85Booster with cross-validation
=================================
"""
import time
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold
from dl85 import DL85Booster, MODEL_LP_RATSCH, MODEL_LP_DEMIRIZ, MODEL_QP_MDBOOST

dataset = np.genfromtxt("../../datasets/tic-tac-toe.txt", delimiter=' ')
X, y = dataset[:, 1:], dataset[:, 0]
n_folds, n_folds_tuning, verbose_level = 5, 4, 0

depth = 1
# params = {'model': MODEL_QP_MDBOOST, 'regulator': [.5, 6, 25, 100], 'name': 'MDBoost'}
params = {'model': MODEL_LP_DEMIRIZ, 'regulator': [0.5, 1, 25, 50], 'name': 'LPBoost'}
# params = {'model': MODEL_LP_RATSCH, 'regulator': [0.25, 0.5, 0.75, 1], 'name': 'LPBoost'}


# for maximization problems
def get_objective_max(estimator, X, y):
    return estimator.objective_


# for minimization problems
def get_objective_min(estimator, X, y):
    return -estimator.objective_


# use the minimum margin criterion to choose the best regulator
def get_min_margin(estimator, X, y):
    return min(estimator.margins_)


# use the sum of normalized margin to choose the best regulator
def get_sum_margin(estimator, X, y):
    return sum(estimator.margins_norm)


eval_criteria = {
    'obj': get_objective_min,  # use get_objective_min in case of minimization problem
    'auc': 'roc_auc',
    'acc': 'accuracy'
}
best_reg_criterion = 'auc'

print("######################################################################\n"
      "#       DL8.5 boosting classifier: CV + hyper-parameter tuning       #\n"
      "######################################################################")
print("<<=== Optiboost ===>>")

kf = KFold(n_splits=n_folds, random_state=42, shuffle=True)
training_accuracies, test_accuracies, test_aucs = [], [], []
start = time.perf_counter()
print("Model building...")
for k, (train_index, test_index) in enumerate(kf.split(X)):
    print("\n===== Fold", k+1, "=====")
    data_train, target_train = X[train_index], y[train_index]
    data_test, target_test = X[test_index], y[test_index]
    clf = DL85Booster(max_depth=depth, model=params['model'])
    grid = {'regulator': params['regulator']}
    gd_sr = GridSearchCV(estimator=clf, error_score=np.nan, param_grid=grid, scoring=eval_criteria, refit=best_reg_criterion, cv=n_folds_tuning, n_jobs=-1, verbose=verbose_level)
    print("Hyper-parameter tuning: grid search cv...", end=" ")
    gd_sr.fit(data_train, target_train)
    print("==> DONE!!!")
    print("Best regulator chosen:", gd_sr.best_params_['regulator'])
    print("Prediction...", end=" ")
    pred = gd_sr.predict(data_test)
    print("==> DONE!!!")
    training_accuracies.append(round(gd_sr.best_estimator_.accuracy_, 4))
    test_accuracies.append(round(accuracy_score(target_test, pred), 4))
    test_aucs.append(round(roc_auc_score(target_test, gd_sr.predict_proba(data_test)[:, 1]), 4))
    print("train_acc:", str(training_accuracies[-1]) + "%", "test_acc:", str(test_accuracies[-1]) + "%", "test_auc:", str(test_aucs[-1]) + "%")
duration = time.perf_counter() - start
print("\nModel built. Duration of building = {}%".format(round(duration, 4)))
print("Average accuracy on training set = {}%".format(round(np.mean(training_accuracies), 4)))
print("Average accuracy on test set = {}%".format(round(np.mean(test_accuracies), 4)))
print("Average auc on test set = {}%\n\n\n".format(round(np.mean(test_aucs), 4)))

import json
import os
import random
import sys
import time
import copy

from gurobipy import Model, GRB, quicksum
from sklearn.base import ClassifierMixin

from .classifier import DL85Classifier
from ...predictors.predictor import DL85Predictor
from ...errors.errors import SearchFailedError, TreeNotFoundError
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator
from copy import deepcopy
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
import cvxpy as cp

MODEL_RATSCH = 1  # regulator of ratsch is between ]0; 1]
MODEL_DEMIRIZ = 2  # regulator of demiriz is between ]1/n_instances; +\infty]
MODEL_AGLIN = 3  # regulator of aglin is between [0; 1]


class DL85Booster(BaseEstimator, ClassifierMixin):
    """
    An optimal binary decision tree classifier.

    Parameters
    ----------
    max_depth : int, default=1
        Maximum depth of the tree to be found
    min_sup : int, default=1
        Minimum number of examples per leaf
    iterative : bool, default=False
        Whether the search will be Iterative Deepening Search or not. By default, it is Depth First Search
    max_error : int, default=0
        Maximum allowed error. Default value stands for no bound. If no tree can be found that is strictly better, the model remains empty.
    stop_after_better : bool, default=False
        A parameter used to indicate if the search will stop after finding a tree better than max_error
    time_limit : int, default=0
        Allocated time in second(s) for the search. Default value stands for no limit. The best tree found within the time limit is stored, if this tree is better than max_error.
    verbose : bool, default=False
        A parameter used to switch on/off the print of what happens during the search
    desc : bool, default=False
        A parameter used to indicate if the sorting of the items is done in descending order of information gain
    asc : bool, default=False
        A parameter used to indicate if the sorting of the items is done in ascending order of information gain
    repeat_sort : bool, default=False
        A parameter used to indicate whether the sorting of items is done at each level of the lattice or only before the search
    nps : bool, default=False
        A parameter used to indicate if only optimal solutions should be stored in the cache.
    print_output : bool, default=False
        A parameter used to indicate if the search output will be printed or not

    Attributes
    ----------
    tree_ : str
        Outputted tree in serialized form; remains empty as long as no model is learned.
    size_ : int
        The size of the outputted tree
    depth_ : int
        Depth of the found tree
    error_ : float
        Error of the found tree
    accuracy_ : float
        Accuracy of the found tree on training set
    lattice_size_ : int
        The number of nodes explored before found the optimal tree
    runtime_ : float
        Time of the optimal decision tree search
    timeout_ : bool
        Whether the search reached timeout or not
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """

    def __init__(
            self,
            base_estimator=None,
            max_depth=1,
            min_sup=1,
            max_iterations=0,
            model=MODEL_DEMIRIZ,
            error_function=None,
            fast_error_function=None,
            iterative=False,
            min_trans_cost=0,
            opti_gap=0.01,
            max_error=0,
            regulator=-1,
            stop_after_better=False,
            time_limit=0,
            verbose=False,
            desc=False,
            asc=False,
            repeat_sort=False,
            print_output=False,
            quiet=True):
        self.clf_params = dict(locals())
        del self.clf_params["self"]
        del self.clf_params["regulator"]
        del self.clf_params["base_estimator"]
        del self.clf_params["max_iterations"]
        del self.clf_params["model"]
        del self.clf_params["min_trans_cost"]
        del self.clf_params["opti_gap"]

        self.base_estimator = base_estimator
        self.max_depth = max_depth
        self.min_sup = min_sup
        self.max_iterations = max_iterations
        self.error_function = error_function
        self.fast_error_function = fast_error_function
        self.iterative = iterative
        self.max_error = max_error
        self.stop_after_better = stop_after_better
        self.time_limit = time_limit
        self.verbose = verbose
        self.desc = desc
        self.asc = asc
        self.repeat_sort = repeat_sort
        self.print_output = print_output
        self.regulator = regulator
        self.quiet = quiet
        self.model = model
        self.min_trans_cost = min_trans_cost
        self.opti_gap = opti_gap

        self.estimators_ = []
        self.estimator_weights_ = []
        self.accuracy_ = 0
        self.n_estimators_ = 0
        self.optimal_ = True
        self.n_iterations_ = 0
        self.duration_ = 0

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("The \"y\" value is compulsory for boosting.")

        start_time = time.perf_counter()

        # initialize variables
        n_instances, n_features = X.shape
        sample_weights = np.array([1/n_instances] * n_instances)
        predictions, r, self.n_iterations_ = None, None, 1

        while (self.max_iterations > 0 and self.n_iterations_ <= self.max_iterations) or self.max_iterations <= 0:
            if not self.quiet:
                print("n_iter", self.n_iterations_)

            # initialize the classifier
            # self.clf_params["time_limit"] = self.clf_params["time_limit"] - (time.perf_counter() - start_time)
            clf = DL85Classifier(**self.clf_params) if self.base_estimator is None else self.base_estimator

            # fit the model
            if self.quiet:
                old_stdout = sys.stdout
                sys.stdout = open(os.devnull, "w")
                clf.fit(X, y, sample_weight=sample_weights.tolist())
                sys.stdout = old_stdout
            else:
                clf.fit(X, y, sample_weight=sample_weights.tolist())

            # print the tree expression of the estimator if it has
            if hasattr(clf, "tree_") and isinstance(clf.tree_, dict) and not self.quiet:
                print(clf.tree_)

            # compute the prediction of the new estimator : 1 if correct else -1
            try:
                pred = np.array([-1 if p != y[i] else 1 for i, p in enumerate(clf.predict(X))])
            except (NotFittedError, SearchFailedError, TreeNotFoundError) as error:
                if not self.quiet:
                    print("Problem during the search so we stop")
                break

            if not self.quiet:
                print("sum pred", pred.sum(), "\n", "sample weights", sample_weights, "\n", "p@w more plus or moins", pred @ sample_weights)

            # check if optimal condition is met
            if self.n_iterations_ > 1:
                if pred @ sample_weights < r + self.opti_gap:
                    if not self.quiet:
                        print("p@w < r ==> finished")
                    self.optimal_ = True
                    break

            # add new prediction to all prediction matrix. Each column represents predictions of a tree for all examples
            predictions = pred.reshape((-1, 1)) if predictions is None else np.concatenate((predictions, pred.reshape(-1, 1)), axis=1)

            if not self.quiet:
                print("pred shape", predictions.shape)
                print(predictions)

            # add the new estimator and compute the dual to find new sample weights for another estimator to add
            self.estimators_.append(clf)
            if self.model == MODEL_RATSCH:
                r, sample_weights, opti, self.estimator_weights_ = self.compute_dual_ratsch(r, sample_weights, predictions)
            elif self.model == MODEL_DEMIRIZ:
                r, sample_weights, opti, self.estimator_weights_ = self.compute_dual_demiriz(r, sample_weights, predictions)
            elif self.model == MODEL_AGLIN:
                r, sample_weights, opti, self.estimator_weights_ = self.compute_dual_aglin(r, sample_weights, predictions)

            if not self.quiet:
                print("sample w", sample_weights, "\n", "r", r, "\n", "opti", opti)
                print("len tree w", len(self.estimator_weights_), "w:", self.estimator_weights_, "\n")

            self.n_iterations_ += 1
        self.duration_ = time.perf_counter() - start_time
        self.n_iterations_ -= 1

        # remove the useless estimators
        zero_ind = [i for i, val in enumerate(self.estimator_weights_) if val == 0]
        if not self.quiet:
            print("all tree w", self.estimator_weights_, "\n", "zero ind", zero_ind)
        self.estimator_weights_ = np.delete(self.estimator_weights_, np.s_[zero_ind], axis=0)
        self.estimators_ = [clf for clf_id, clf in enumerate(self.estimators_) if clf_id not in zero_ind]
        predictions = np.delete(predictions, np.s_[zero_ind], axis=1)
        if not self.quiet:
            print("final pred shape", predictions.shape, "\n", predictions)

        # compute training accuracy of the found ensemble and store it in the variable `accuracy_`
        forest_pred_val = np.dot(predictions, np.array(self.estimator_weights_))
        train_pred_correct_or_not = np.where(forest_pred_val < 0, 0, 1)  # 1 if prediction is correct, 0 otherwise
        self.accuracy_ = sum(train_pred_correct_or_not)/len(y)

        # save the number of found estimators
        self.n_estimators_ = len(self.estimators_)

        # Show each non-zero estimator weight and its tree expression if it has
        if not self.quiet:
            for i, estimator in enumerate(sorted(zip(self.estimator_weights_, self.estimators_), key=lambda x: x[0], reverse=True)):
                print("clf n_", i+1, " ==>\tweight: ", estimator[0], sep="", end="")
                if hasattr(estimator[1], "tree_") and isinstance(estimator[1].tree_, dict):
                    print(" \tjson_string: ", estimator[1].tree_, sep="")
                else:
                    print()

        if self.n_estimators_ == 0:
            raise NotFittedError("No tree selected")

        return self

    def compute_dual_ratsch(self, r, u, predictions):
        r_ = cp.Variable()
        u_ = cp.Variable(u.shape[0])

        obj = cp.Minimize(r_)
        constr = [predictions[:, i] @ u_ <= r_ for i in range(predictions.shape[1])]
        constr.append(-u_ <= 0)
        constr.append(u_ <= self.regulator)
        constr.append(cp.sum(u_) == 1)

        problem = cp.Problem(obj, constr)

        if self.quiet:
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")
        opti = problem.solve(solver=cp.GUROBI)
        if self.quiet:
            sys.stdout = old_stdout

        return r_.value, u_.value, opti, [x.dual_value for x in problem.constraints[:predictions.shape[1]]]

    def compute_dual_demiriz(self, r, u, predictions):
        u_ = cp.Variable(u.shape[0])

        obj = cp.Maximize(cp.sum(u_))
        constr = [predictions[:, i] @ u_ <= 1 for i in range(predictions.shape[1])]
        constr.append(-u_ <= 0)
        constr.append(u_ <= self.regulator)

        problem = cp.Problem(obj, constr)

        if self.quiet:
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")
        opti = problem.solve(solver=cp.GUROBI)
        if self.quiet:
            sys.stdout = old_stdout

        return 1, u_.value, opti, [x.dual_value for x in problem.constraints[:predictions.shape[1]]]

    def compute_dual_aglin(self, r, u, predictions):
        r_ = cp.Variable()
        u_ = cp.Variable(u.shape[0])

        obj = cp.Minimize(r_)
        constr = [predictions[:, i] @ u_ <= r_ for i in range(predictions.shape[1])]
        constr.append(-u_ <= 0)
        # constr.append(u_ <= self.regulator)
        constr.append(cp.sum(u_) == self.regulator)

        problem = cp.Problem(obj, constr)

        if self.quiet:
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")
        opti = problem.solve(solver=cp.GUROBI)
        if self.quiet:
            sys.stdout = old_stdout

        # print(r_.value, u_.value, opti)
        return r_.value, u_.value, opti, [x.dual_value for x in problem.constraints[:predictions.shape[1]]]

    def get_class(self, forest_decision):
        """
        compute the class of each transaction in list, based on decision of multiples trees
        :param forest_decision: list representing the prediction of each tree
        :return: the class with highest weight
        """
        sums = {}
        for key, value in zip(forest_decision, self.estimator_weights_):
            try:
                sums[key] += value
            except KeyError:
                sums[key] = value
        return list({k: v for k, v in sorted(sums.items(), key=lambda item: item[1], reverse=True)}.keys())[0]

    def get_predictions(self, predict_per_clf):
        # transpose prediction list to have per row a list of decision for each tree for each transaction
        predict_per_trans = list(map(list, zip(*predict_per_clf)))
        return list(map(lambda x: self.get_class(x), predict_per_trans))

    def predict(self, X, y=None):
        if self.n_estimators_ == 0:  # fit method has not been called
            print(self.estimators_)
            print(self.estimator_weights_)
            raise NotFittedError("Call fit method first" % {'name': type(self).__name__})
        # Run a prediction on each estimator
        predict_per_clf = [clf.predict(X) for clf_id, clf in enumerate(self.estimators_)]
        return self.get_predictions(predict_per_clf)

    def get_nodes_count(self):
        if self.n_estimators_ == 0:  # fit method has not been called
            raise NotFittedError("Call fit method first" % {'name': type(self).__name__})
        return sum([clf.get_nodes_count() for clf in self.estimators_])
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
from sklearn.utils.multiclass import unique_labels
from sklearn.base import BaseEstimator
from copy import deepcopy
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from .utils.matrix import *
import numpy as np
import cvxpy as cp
import math

MODEL_LP_RATSCH = 1  # regulator of ratsch is between ]0; 1]
MODEL_LP_DEMIRIZ = 2  # regulator of demiriz is between ]1/n_instances; +\infty]
MODEL_QP_MDBOOST = 3


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
            model=MODEL_LP_DEMIRIZ,
            gamma=None,
            error_function=None,
            fast_error_function=None,
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
        del self.clf_params["gamma"]
        del self.clf_params["min_trans_cost"]
        del self.clf_params["opti_gap"]

        self.base_estimator = base_estimator
        self.max_depth = max_depth
        self.min_sup = min_sup
        self.max_iterations = max_iterations
        self.error_function = error_function
        self.fast_error_function = fast_error_function
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
        self.gamma = gamma
        self.min_trans_cost = min_trans_cost
        self.opti_gap = opti_gap
        self.n_instances = None
        self.A_inv = None

        self.optimal_ = False  # whether the ensemble is optimal
        self.estimators_, self.estimator_weights_ = [], []
        self.accuracy_ = self.duration_ = self.n_estimators_ = self.n_iterations_ = 0
        self.margins_ = []  # the ensemble margin on training instances
        self.margins_norm_ = []  # the normalized margins
        self.classes_ = []
        self.objective_ = None  # the optimal value reached by the ensemble

    def fit(self, X, y=None, X_test=None, y_test=None, iter_file=None):
        if y is None:
            raise ValueError("The \"y\" value is compulsory for boosting.")

        start_time = time.perf_counter()

        # initialize variables
        self.n_instances, _ = X.shape
        sample_weights = np.array([1/self.n_instances] * self.n_instances)
        predictions, r, self.n_iterations_, constant = None, None, 1, 0.0001

        if self.model == MODEL_QP_MDBOOST:
            # Define the inverse of the A matrix
            if self.gamma is None:
                A = np.full((self.n_instances, self.n_instances), -1/(self.n_instances - 1), dtype=np.float64)
                np.fill_diagonal(A, 1)
                A = np.add(A, np.dot(np.eye(self.n_instances), constant))  # regularize A to make A^-1 sure it is really PSD
            else:
                self.gamma = 1 / self.n_instances if self.gamma == 'auto' else 1 / (self.n_features * X.var()) if self.gamma == 'scale' else 1 / X.var() if self.gamma == 'nscale' else 1 / self.n_instances
                A = np.identity(self.n_instances, dtype=np.float64)
                for i in range(self.n_instances):
                    for j in range(self.n_instances):
                        if i != j:
                            A[i, j] = np.exp(-self.gamma * np.linalg.norm(np.subtract(X[i, :], X[j, :]))**2)

            if not self.quiet:
                print("Matrix A", A)
                print("is pos def: ", is_pd(A))
                print("is psd: ", is_psd(A))

            # invert the matrix A
            self.A_inv = np.linalg.pinv(A)

            # find the nearest pos def matrix if A_inv is not
            if pos_def(self.A_inv) is False:
                self.A_inv = nearest_pd(self.A_inv)

            if not self.quiet:
                print("Matrix A_inv", self.A_inv)
                print("is pos def", is_pd(self.A_inv))
                print("is psd", is_psd(self.A_inv))

        if not self.quiet:
            print()

        self.classes_ = unique_labels(y)

        # if a test set is provided, some metrics are store after each boosting iteration
        if X_test is not None and y_test is not None:
            iter_file_name = (iter_file if iter_file is not None else "iter_file") + ".csv"
            acc_stream = open(iter_file_name, "w")
            acc_stream.write("objective,train_acc,test_acc,train_auc,test_auc,n_iter,n_trees,min_margin,avg_margin,var_margin\n")
            acc_stream.flush()

        # run the boosting loop for a fixed number of iterations if max_iterations is specified
        while (self.max_iterations > 0 and self.n_iterations_ <= self.max_iterations) or self.max_iterations <= 0:
            if not self.quiet:
                print("n_iter", self.n_iterations_)

            # initialize the classifier
            # initialize the base classifier. DL8.5 can be changed by another optimal classifier
            clf = DL85Classifier(**self.clf_params) if self.base_estimator is None else self.base_estimator

            # fit the model with the correct weights. The print of the learning are disabled.
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")
            clf.fit(X, y, sample_weight=sample_weights.tolist())
            sys.stdout = old_stdout

            # print the tree expression of the estimator if it has
            if not self.quiet:
                print("A new tree has been learnt based on previous found sample weights")
                if hasattr(clf, "tree_") and isinstance(clf.tree_, dict):
                    print(clf.tree_)

            # compute the prediction of the new estimator : 1 if correct else -1
            try:
                pred = np.array([-1 if p != y[i] else 1 for i, p in enumerate(clf.predict(X))])
            except (NotFittedError, SearchFailedError, TreeNotFoundError) as error:
                if not self.quiet:
                    print("Problem during the search so we stop")
                break

            if not self.quiet:
                print("correct predictions - incorrect predictions =", pred.sum())
                print("np.dot(predictions, sample_weigths) =", pred @ sample_weights)

            # check if optimal condition is met
            if self.n_iterations_ > 1:

                # if a test set is provided, write metrics after each boosting iteration
                if X_test is not None and y_test is not None:
                    n_treess = len([i for i in self.estimator_weights_ if i != 0])
                    if n_treess > 0:
                        n_treess = str(n_treess)
                        self.n_estimators_ = len(self.estimators_)
                        train_acc = str(accuracy_score(y, self.predict(X)))
                        test_acc = str(accuracy_score(y_test, self.predict(X_test)))
                        train_auc = str(roc_auc_score(y, self.predict_proba(X)[:, 1]))
                        test_auc = str(roc_auc_score(y_test, self.predict_proba(X_test)[:, 1]))
                        acc_stream.write(str(opti) + "," + train_acc + "," + test_acc + "," + train_auc + "," + test_auc + "," + str(self.n_iterations_ - 1) + "," + n_treess + "," + str(min(self.margins_norm_)) + "," + str(np.mean(self.margins_norm_)) + "," + str(np.var(self.margins_norm_)) + "\n")
                        acc_stream.flush()

                # optimality condition. an epsilon is added to fix column generation convergence problems
                if pred @ sample_weights < r + self.opti_gap:
                    if not self.quiet:
                        print("np.dot(predictions, sample_weigths):{} < r:{} + espsilon:{} ==> we cannot add the new tree. End of iterations".format(pred @ sample_weights, r, self.opti_gap))
                        print("Objective value at end is", opti)
                    self.optimal_ = True
                    self.objective_ = opti
                    break
                # not yet optimal
                self.objective_ = opti
                if not self.quiet:
                    print("np.dot(predictions, sample_weigths):{} >= r:{} + epsilon:{}. We can add the new tree.".format(pred @ sample_weights, r, self.opti_gap))

            # add new prediction to all predictions matrix.Each column represents predictions of a tree for all examples
            predictions = pred.reshape((-1, 1)) if predictions is None else np.concatenate((predictions, pred.reshape(-1, 1)), axis=1)

            if not self.quiet:
                print("whole predictions shape", predictions.shape)
                print("run dual...")

            # add the new estimator to list of estimators
            self.estimators_.append(deepcopy(clf))

            # compute the dual to find new sample weights (the best to reach optimality) for the next estimator to add
            if self.model == MODEL_LP_RATSCH:
                r, sample_weights, opti, self.estimator_weights_ = self.compute_dual_ratsch(predictions)
            elif self.model == MODEL_LP_DEMIRIZ:
                r, sample_weights, opti, self.estimator_weights_ = self.compute_dual_demiriz(predictions)
            elif self.model == MODEL_QP_MDBOOST:
                r, sample_weights, opti, self.estimator_weights_ = self.compute_dual_mdboost(predictions)

            # compute some metrics (margins concept) based on adaboost intuition
            self.margins_ = (predictions @ np.array(self.estimator_weights_).reshape(-1, 1)).transpose().tolist()[0]
            self.margins_norm_ = (predictions @ np.array([float(i)/sum(self.estimator_weights_) for i in self.estimator_weights_]).reshape(-1, 1)).transpose().tolist()[0] if sum(self.estimator_weights_) > 0 else None

            if not self.quiet:
                print("after dual")
                print("We got", len(self.estimator_weights_), "trees with weights w:", self.estimator_weights_)
                print("Objective value at this stage is", opti)
                print("Value of r is", r)
                print("The sorted margin at this stage is", sorted(self.margins_))
                print("min margin:", min(self.margins_norm_), "\tmax margin:", max(self.margins_norm_), "\tavg margin:", np.mean(self.margins_norm_), "\tstd margin:", np.std(self.margins_norm_), "\tsum:", sum(self.margins_norm_))
                print("number of neg margins:", len([marg for marg in self.margins_norm_ if marg < 0]), "\tnumber of pos margins:", len([marg for marg in self.margins_norm_ if marg >= 0]))
                print("The new sample weight for the next iteration is", sample_weights.tolist(), "\n")

            self.n_iterations_ += 1

        # close the metrics file
        if X_test is not None and y_test is not None:  # handle each iteration value
            acc_stream.close()

        # compute the learning duration and fix the iterations number counter
        self.duration_ = time.perf_counter() - start_time
        self.n_iterations_ -= 1

        # at the end, remove the useless estimators (the one with weight = 0)
        zero_ind = [i for i, val in enumerate(self.estimator_weights_) if val == 0]
        if not self.quiet:
            print("\nall tree w", self.estimator_weights_, "\n", "zero ind", zero_ind)
        self.estimator_weights_ = np.delete(self.estimator_weights_, np.s_[zero_ind], axis=0)
        self.estimators_ = [clf for clf_id, clf in enumerate(self.estimators_) if clf_id not in zero_ind]
        predictions = np.delete(predictions, np.s_[zero_ind], axis=1)
        if not self.quiet:
            print("final pred shape", predictions.shape)

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

    def compute_dual_ratsch(self, predictions):  # primal is maximization
        r_ = cp.Variable()
        u_ = cp.Variable(self.n_instances)
        obj = cp.Minimize(r_)
        constr = [predictions[:, i] @ u_ <= r_ for i in range(predictions.shape[1])]
        constr.append(-u_ <= 0)
        if self.regulator > 0:
            constr.append(u_ <= self.regulator)
        constr.append(cp.sum(u_) == 1)
        problem = cp.Problem(obj, constr)
        if self.quiet:
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")
        opti = problem.solve(solver=cp.GUROBI)
        if self.quiet:
            sys.stdout = old_stdout
        return r_.value, u_.value, opti, [x.dual_value.tolist() for x in problem.constraints[:predictions.shape[1]]]

    def compute_dual_demiriz(self, predictions):  # primal is minimization
        u_ = cp.Variable(self.n_instances)
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
        return 1, u_.value, opti, [x.dual_value.tolist() for x in problem.constraints[:predictions.shape[1]]]

    def compute_dual_mdboost(self, predictions):  # primal is maximization
        r_ = cp.Variable()
        u_ = cp.Variable(self.n_instances)
        obj = cp.Minimize(r_ + 1/(2*self.regulator) * cp.quad_form((u_ - 1), self.A_inv))
        constr = [predictions[:, i] @ u_ <= r_ for i in range(predictions.shape[1])]
        problem = cp.Problem(obj, constr)
        if self.quiet:
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")
        opti = problem.solve(solver=cp.GUROBI)
        if self.quiet:
            sys.stdout = old_stdout
        return r_.value, u_.value, opti, [x.dual_value.tolist() for x in problem.constraints]

    def softmax(self, X, copy=True):
        """
        Calculate the softmax function.
        The softmax function is calculated by
        np.exp(X) / np.sum(np.exp(X), axis=1)
        This will cause overflow when large values are exponentiated.
        Hence the largest value in each row is subtracted from each data
        point to prevent this.
        Parameters
        ----------
        X : array-like of float of shape (M, N)
            Argument to the logistic function.
        copy : bool, default=True
            Copy X or not.
        Returns
        -------
        out : ndarray of shape (M, N)
            Softmax function evaluated at every point in x.
        """
        if copy:
            X = np.copy(X)
        max_prob = np.max(X, axis=1).reshape((-1, 1))
        X -= max_prob
        np.exp(X, X)
        sum_prob = np.sum(X, axis=1).reshape((-1, 1))
        X /= sum_prob
        return X

    def predict(self, X, y=None):
        # in case no tree has been found
        if self.n_estimators_ == 0:  # fit method has not been called or the regulator is not good
            print(self.estimators_)
            print(self.estimator_weights_)
            raise NotFittedError("Call fit method first or change the regulator" % {'name': type(self).__name__})
        # Run a prediction on each estimator
        predict_per_clf = np.asarray([clf.predict(X) for clf in self.estimators_]).transpose()
        # return the prediction based on all estimators. The mex weighted class is returned
        return np.apply_along_axis(lambda x: np.argmax(np.bincount(x, weights=self.estimator_weights_)), axis=1, arr=predict_per_clf.astype('int'))

    def predict_proba(self, X):
        classes = self.classes_[:, np.newaxis]
        pred = sum((np.array(estimator.predict(X)) == classes).T * w for estimator, w in zip(self.estimators_, self.estimator_weights_))
        pred /= sum(self.estimator_weights_)
        pred[:, 0] *= -1
        decision = pred.sum(axis=1)
        decision = np.vstack([-decision, decision]).T / 2
        return self.softmax(decision, False)

    def get_nodes_count(self):
        if self.n_estimators_ == 0:  # fit method has not been called
            raise NotFittedError("Call fit method first" % {'name': type(self).__name__})
        return sum([clf.get_nodes_count() for clf in self.estimators_])

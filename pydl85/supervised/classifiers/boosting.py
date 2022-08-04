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
from ..classifiers.classifier import Cache_Type, Wipe_Type, DL85Classifier
from ...errors.errors import SearchFailedError, TreeNotFoundError
from sklearn.exceptions import NotFittedError
from sklearn.utils.multiclass import unique_labels
from sklearn.base import BaseEstimator
from copy import deepcopy
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
import cvxpy as cp
import math

MODEL_LP_RATSCH = 1  # regulator of ratsch is between ]0; 1]
MODEL_LP_DEMIRIZ = 2  # regulator of demiriz is between ]1/n_instances; +\infty]
MODEL_LP_AGLIN = 3  # regulator of aglin is between [0; 1]
MODEL_QP_MDBOOST = 4


def is_pos_def(x):
    # print(np.linalg.eigvals(x))
    return np.all(np.linalg.eigvals(x) > 0)


def is_semipos_def(x):
    return np.all(np.linalg.eigvals(x) >= 0)


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False


def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def get_near_psd(A):
    C = (A + A.T)/2
    eigval, eigvec = np.linalg.eig(C)
    eigval[eigval < 0] = 0

    return eigvec.dot(np.diag(eigval)).dot(eigvec.T)


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
            quiet=True,
            cache_type=Cache_Type.Cache_TrieItemset,
            maxcachesize=0,
            wipe_type=Wipe_Type.Subnodes,
            wipe_factor=0.5,
            use_cache=True,
            use_ub=True,
            dynamic_branch=True):
        self.clf_params = dict(locals())
        self.clf_params["depth_two_special_algo"] = False
        self.clf_params["similar_lb"] = True
        self.clf_params["similar_for_branching"] = False
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

        self.optimal_ = True
        self.estimators_, self.estimator_weights_ = [], []
        self.accuracy_ = self.duration_ = self.n_estimators_ = self.n_iterations_ = 0
        self.margins_ = []
        self.margins_norm_ = []
        self.classes_ = []

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("The \"y\" value is compulsory for boosting.")

        start_time = time.perf_counter()

        # initialize variables
        self.n_instances, _ = X.shape
        sample_weights = np.array([1/self.n_instances] * self.n_instances)
        predictions, r, self.n_iterations_, constant = None, None, 1, 0.0001

        if self.model == MODEL_QP_MDBOOST:
            # A_inv = None
            if self.gamma is None:
                # Build positive semidefinite A matrix
                self.A_inv = np.full((self.n_instances, self.n_instances), -1/(self.n_instances - 1), dtype=np.float64)
                np.fill_diagonal(self.A_inv, 1)
                # regularize A to make sure it is really PSD
                self.A_inv = np.add(self.A_inv, np.dot(np.eye(self.n_instances), constant))
            else:
                if self.gamma == 'auto':
                    self.gamma = 1 / self.n_instances
                elif self.gamma == 'scale':
                    self.gamma = 1 / (self.n_features * X.var())
                elif self.gamma == 'nscale':
                    # scaler = MinMaxScaler(feature_range=(-10, 10))
                    # self.gamma = 1 / scaler.fit_transform(X).var()
                    self.gamma = 1 / X.var()
                self.A_inv = np.identity(self.n_instances, dtype=np.float64)
                for i in range(self.n_instances):
                    for j in range(self.n_instances):
                        if i != j:
                            self.A_inv[i, j] = np.exp(-self.gamma * np.linalg.norm(np.subtract(X[i, :], X[j, :]))**2)
                        # else:
                        #     self.A_inv[i, j] = 1
                        # for k in range(n_instances):
                        #     if k != i:
                        #         self.A_inv[i, j] += np.exp(-self.gamma * np.linalg.norm(np.subtract(X[i, :], X[k, :]))**2)
            # if not is_pos_def(self.A_inv) and not is_semipos_def(self.A_inv):
            # A_inv = np.add(self.A_inv, np.dot(np.eye(n_instances), constant))

            if not self.quiet:
                print(self.A_inv)
                print("is psd", is_pos_def(self.A_inv))
                print("is semi psd", is_semipos_def(self.A_inv))

            # if isPD(A_inv) is False:
            #     A_inv = nearestPD(A_inv)

            # invert A matrix
            # try:
            #     A_inv = np.linalg.inv(A_inv)
            # except:
            self.A_inv = np.linalg.pinv(self.A_inv)

            if isPD(self.A_inv) is False:
                self.A_inv = nearestPD(self.A_inv)

            if not self.quiet:
                print("A_inv")
                # print(A_inv)
                print("is psd", isPD(self.A_inv))
                print("is psd", is_pos_def(self.A_inv))
                print("is semi psd", is_semipos_def(self.A_inv))
                print("is semi psd", is_pos_def(nearestPD(self.A_inv)))

        if not self.quiet:
            print()
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
            if not self.quiet:
                print("A new tree has been learnt based on previous found sample weights")
                if hasattr(clf, "tree_") and isinstance(clf.tree_, dict):
                    pass
                    # print(clf.tree_)

            # compute the prediction of the new estimator : 1 if correct else -1
            try:
                pred = np.array([-1 if p != y[i] else 1 for i, p in enumerate(clf.predict(X))])
                # pred = np.array([2*p[y[i]]-1 for i, p in enumerate(clf.predict_proba(X))])
                # pred = np.array([.5 * y[i] * math.log(p[1]/(p[0]+.00000001)) for i, p in enumerate(clf.predict_proba(X))])
                # if not self.quiet:
                #     print("entering tree value", pred @ sample_weights)
            except (NotFittedError, SearchFailedError, TreeNotFoundError) as error:
                if not self.quiet:
                    print("Problem during the search so we stop")
                break

            if not self.quiet:
                print("correct predictions - incorrect predictions =", pred.sum())
                print("np.dot(predictions, sample_weigths) =", pred @ sample_weights)
                # print("sum pred", pred.sum(), "\n", "sample weights inline", sample_weights.tolist(), "\nsample weights", sample_weights, "\n", "p@w more plus or moins", pred @ sample_weights)

            # check if optimal condition is met
            if self.n_iterations_ > 1:
                if pred @ sample_weights < r + self.opti_gap:
                    if not self.quiet:
                        print("np.dot(predictions, sample_weigths) < r + espsilon ==> we cannot add the new tree. End of iterations")
                        print("Objective value at end is", opti)
                    self.optimal_ = True
                    break
                if not self.quiet:
                    print("np.dot(predictions, sample_weigths) >= r + epsilon. We can add the new tree.")

            # add new prediction to all prediction matrix. Each column represents predictions of a tree for all examples
            predictions = pred.reshape((-1, 1)) if predictions is None else np.concatenate((predictions, pred.reshape(-1, 1)), axis=1)

            if not self.quiet:
                print("whole predictions shape", predictions.shape)
                print("run dual...")
                # print(predictions)

            # add the new estimator and compute the dual to find new sample weights for another estimator to add
            self.estimators_.append(deepcopy(clf))
            if self.model == MODEL_LP_RATSCH:
                r, sample_weights, opti, self.estimator_weights_ = self.compute_dual_ratsch(predictions)
            elif self.model == MODEL_LP_DEMIRIZ:
                r, sample_weights, opti, self.estimator_weights_ = self.compute_dual_demiriz(predictions)
            elif self.model == MODEL_LP_AGLIN:
                r, sample_weights, opti, self.estimator_weights_ = self.compute_dual_aglin(predictions)
            elif self.model == MODEL_QP_MDBOOST:
                r, sample_weights, opti, self.estimator_weights_ = self.compute_dual_mdboost(predictions)

            self.margins_ = (predictions @ np.array(self.estimator_weights_).reshape(-1, 1)).transpose().tolist()[0]
            self.margins_norm_ = (predictions @ np.array([float(i)/sum(self.estimator_weights_) for i in self.estimator_weights_]).reshape(-1, 1)).transpose().tolist()[0] if sum(self.estimator_weights_) > 0 else None

            if not self.quiet:
                print("after dual")
                print("We got", len(self.estimator_weights_), "trees with weights w:", self.estimator_weights_)
                print("Objective value at this stage is", opti)
                print("Value of r is", r)
                print("The sorted margin at this stage is", sorted(self.margins_))
                mean = sum(self.margins_) / len(self.margins_)
                variance = sum([((x - mean) ** 2) for x in self.margins_]) / len(self.margins_)
                std = variance ** 0.5
                print("min margin:", min(self.margins_), "\tmax margin:", max(self.margins_), "\tavg margin:", mean, "\tstd margin:", std, "\tsum:", sum(self.margins_))
                print("number of neg margins:", len([marg for marg in self.margins_ if marg < 0]), "\tnumber of pos margins:", len([marg for marg in self.margins_ if marg >= 0]))
                print("The new sample weight for the next iteration is", sample_weights.tolist(), "\n")

            self.n_iterations_ += 1
        self.duration_ = time.perf_counter() - start_time
        self.n_iterations_ -= 1

        # remove the useless estimators
        zero_ind = [i for i, val in enumerate(self.estimator_weights_) if val == 0]
        if not self.quiet:
            print("\nall tree w", self.estimator_weights_, "\n", "zero ind", zero_ind)
        self.estimator_weights_ = np.delete(self.estimator_weights_, np.s_[zero_ind], axis=0)
        self.estimators_ = [clf for clf_id, clf in enumerate(self.estimators_) if clf_id not in zero_ind]
        predictions = np.delete(predictions, np.s_[zero_ind], axis=1)
        if not self.quiet:
            print("final pred shape", predictions.shape)
            # print(predictions)

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

        self.classes_ = unique_labels(y)

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

    def compute_dual_aglin(self, predictions):  # primal is maximization
        r_ = cp.Variable()
        u_ = cp.Variable(self.n_instances)
        v_ = cp.Variable(self.n_instances)
        obj = cp.Minimize(r_)
        constr = [-(predictions[:, t] @ u_) <= r_ for t in range(predictions.shape[1])]
        constr.append(u_ + v_ == -1)
        constr.append(cp.sum(v_) == self.regulator)
        constr.append(-v_ <= 0)
        problem = cp.Problem(obj, constr)
        if self.quiet:
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")
        opti = problem.solve(solver=cp.GUROBI)
        if self.quiet:
            sys.stdout = old_stdout
        # print(predictions.shape[1])
        # print(problem.constraints)
        # print("size cons", len([x.dual_value for x in problem.constraints[:predictions.shape[1]]]))
        # print("size cons", len([x.dual_value for x in problem.constraints]))
        # print("type cons", type([x.dual_value for x in problem.constraints][1]))
        # print("size cons", [x.dual_value for x in problem.constraints][0].shape)
        # print("size cons", [x.dual_value for x in problem.constraints][0])
        # print("size cons", [x.dual_value for x in problem.constraints][1].shape)
        # print("size cons", [x.dual_value for x in problem.constraints][2].shape)
        # print("size cons", [x.dual_value for x in problem.constraints][3].shape)
        # print("size cons", len([x.dual_value for x in problem.constraints[0]]))
        # print("size cons", len(problem.constraints[1]))
        # print("size cons", len(problem.constraints[2]))
        # print("size cons", len(problem.constraints[3]))
        # print(type([x.dual_value for x in problem.constraints[predictions.shape[1]:predictions.shape[1] + 1]]))
        # print(type([x.dual_value for x in problem.constraints[predictions.shape[1]:predictions.shape[1] + 1]][0]))
        # print(type(np.array([x.dual_value for x in problem.constraints[predictions.shape[1]:predictions.shape[1] + 1]])))
        # print([x.dual_value for x in problem.constraints[predictions.shape[1]:predictions.shape[1] + 1]])
        # return r_.value, [x.dual_value for x in problem.constraints[predictions.shape[1]:predictions.shape[1] + 1]][0], opti, [x.dual_value.tolist() for x in problem.constraints[:predictions.shape[1]]]
        return r_.value, v_.value, opti, [x.dual_value.tolist() for x in problem.constraints[:predictions.shape[1]]]

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

    # def compute_dual_aglin(self, predictions):  # primal is minimization
    #     r_ = cp.Variable()
    #     u_ = cp.Variable(self.n_instances)
    #     obj = cp.Minimize(r_)
    #     constr = [predictions[:, i] @ u_ <= r_ for i in range(predictions.shape[1])]
    #     constr.append(-u_ <= 0)
    #     # constr.append(u_ <= self.regulator)
    #     constr.append(cp.sum(u_) == self.regulator)
    #     problem = cp.Problem(obj, constr)
    #     if self.quiet:
    #         old_stdout = sys.stdout
    #         sys.stdout = open(os.devnull, "w")
    #     opti = problem.solve(solver=cp.GUROBI)
    #     if self.quiet:
    #         sys.stdout = old_stdout
    #     return r_.value, u_.value, opti, [x.dual_value.tolist() for x in problem.constraints[:predictions.shape[1]]]

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
        if self.n_estimators_ == 0:  # fit method has not been called
            print(self.estimators_)
            print(self.estimator_weights_)
            raise NotFittedError("Call fit method first" % {'name': type(self).__name__})
        # return np.argmax(self.predict_proba(X), axis=1)
        # Run a prediction on each estimator
        predict_per_clf = np.asarray([clf.predict(X) for clf in self.estimators_]).transpose()
        return np.apply_along_axis(lambda x: np.argmax(np.bincount(x, weights=self.estimator_weights_)), axis=1, arr=predict_per_clf.astype('int'))

    # def predict_proba(self, X):
    #     prob_per_tree = np.asarray([clf.predict_proba(X) for clf in self.estimators_])
    #     return np.average(prob_per_tree, axis=0, weights=self.estimator_weights_)

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

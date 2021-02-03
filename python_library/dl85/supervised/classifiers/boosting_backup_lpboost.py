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


BOOST_SVM1 = 1
BOOST_SVM2 = 2
MAX_TREES_PROBLEM = 1
SUITABLE_TREES_PROBLEM = 2


def get_validation_data(X_all, y_all, X_train, y_train):
    comp_x = X_train == X_all
    comp_y = y_train == y_all
    if comp_x.all() and comp_y.all():
        return None, None
    kf = StratifiedKFold(n_splits=5)
    for train_index, test_index in kf.split(X_all, y_all):
        comp_x = X_train == X_all[train_index]
        comp_y = y_train == y_all[train_index]
        if comp_x.all() and comp_y.all():
            return X_all[test_index], y_all[test_index]


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
            max_estimators=0,
            max_iterations=0,
            model=BOOST_SVM2,
            error_function=None,
            fast_error_function=None,
            iterative=False,
            min_trans_cost=0,
            opti_gap=0.01,
            step=1,
            tolerance=0.00001,
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
        del self.clf_params["max_estimators"]
        del self.clf_params["regulator"]
        del self.clf_params["base_estimator"]
        del self.clf_params["max_iterations"]
        del self.clf_params["model"]
        del self.clf_params["min_trans_cost"]
        del self.clf_params["opti_gap"]
        del self.clf_params["step"]
        del self.clf_params["tolerance"]

        self.base_estimator = base_estimator
        self.max_depth = max_depth
        self.min_sup = min_sup
        self.max_estimators = max_estimators
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
        self.problem = None
        self.step = step
        self.tolerance = tolerance

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
        preds, sample_weights = [], []

        if self.regulator <= 0:
            self.regulator = 1 / (random.uniform(0, 1) * X.shape[0])

        # initialize the first classifier
        if not self.quiet:
            print("search for first estimator")
        if self.base_estimator is None:
            self.clf_params["time_limit"] = self.clf_params["time_limit"] - (time.perf_counter() - start_time)
            first_clf = DL85Classifier(**self.clf_params)
        else:
            first_clf = self.base_estimator
        # fit the first classifier
        if self.quiet:
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")
            first_clf.fit(X, y, sample_weight=[1/X.shape[0]] * X.shape[0])
            sys.stdout = old_stdout
        else:
            first_clf.fit(X, y, sample_weight=[1/X.shape[0]] * X.shape[0])

        # print the tree expression of the estimator if it has
        if not self.quiet and hasattr(first_clf, "tree_") and isinstance(first_clf.tree_, dict):
            print(json.dumps(first_clf.tree_))

        # add the current estimator to the ensemble
        self.estimators_.append(first_clf)

        # save the prediction of the estimator : 1 if correct else -1
        first_pred = [-1 if p != y[i] else 1 for i, p in enumerate(first_clf.predict(X))]
        preds.append(first_pred)

        # search expresses the fact that we will look for at leat a second tree
        search = True
        # minimum value regulator should be set to, to ensure we will get a least one tree
        ppred = first_clf.predict(X)
        n_error = sum([1 if u != y[i] else 0 for i, u in enumerate(ppred)])
        denom = X.shape[0] - (2 * n_error)
        min_regulator = 1 / denom
        if denom <= 0:
            raise ValueError("base estimator error =", n_error/X.shape[0], "It should be lower than 50%")
            # print("regulator should be lower than", denom, file=sys.stderr)
            # print("the ideal case would be to have base estimator error lower than 50%", file=sys.stderr)
            # print("base estimator error = 50% will also crash the code", file=sys.stderr)
        min_epsilon, max_epsilon = 1 / X.shape[0], X.shape[0]

        # this case is similar a simple call to dl85classifier
        if self.max_estimators == 1:
            self.regulator = min_regulator + max_epsilon  # to make the weight greater than 0
            search = False  # to avoid searching for another tree

        # we will search for the suitable regulator reaching less than n_max_trees at optimality
        elif self.max_estimators > 1:
            self.problem = MAX_TREES_PROBLEM
            self.regulator = min_regulator + min_epsilon
            print("denom", denom, X.shape[0], n_error)
            print("reg", self.regulator)

        # n_max_trees is not provided
        else:
            # as regulator is also not provided, we return just one tree as in dl85classifier
            if self.regulator <= 0:  # return one tree
                self.regulator = min_regulator + max_epsilon  # to make the weight greater than 0
                search = False  # to avoid searching for another tree

            # we will look for the suitable n_trees according to the provided regulator
            else:
                self.problem = SUITABLE_TREES_PROBLEM

        # compute the weights of the estimator and the optimal value for the objective
        first_weight, opti_obj = [], None
        if self.model == BOOST_SVM1:
            first_weight, rho, opti_obj = self.calculate_estimator_weights(y, preds)
        elif self.model == BOOST_SVM2:
            first_weight, rho, opti_obj = self.calculate_estimator_weights_(y, preds)
        self.estimator_weights_ = first_weight[:]

        self.n_iterations_ += 1

        best_estimators_, best_preds, best_estimator_weights_, best_regulator, best_iterations, best_n_trees = [], [], [], None, None, 0

        # in case, we want to test another regulator, we have to store the current values in case there is no better
        def save(obj, n_trees):
            best_estimators_[:] = copy.deepcopy(obj.estimators_)
            best_preds[:] = copy.deepcopy(preds[:])
            best_estimator_weights_[:] = copy.deepcopy(obj.estimator_weights_)
            nonlocal best_regulator
            best_regulator = obj.regulator
            nonlocal best_iterations
            best_iterations = obj.n_iterations_
            nonlocal best_n_trees
            best_n_trees = n_trees

        # in case, we want to test another regulator, we have to clean the current values
        def clear(obj):
            obj.estimators_ = [copy.deepcopy(first_clf)]
            preds[:] = [copy.deepcopy(first_pred)]
            obj.estimator_weights_ = copy.deepcopy(first_weight)
            obj.n_iterations_ = 1
            # obj.regulator = min_regulator + min_epsilon

        # in case, we tested different regulators, we have to restore the better one
        def restore(obj):
            obj.estimators_ = best_estimators_
            preds[:] = best_preds
            obj.estimator_weights_ = best_estimator_weights_
            obj.n_iterations_ = best_iterations
            obj.regulator = best_regulator

        if self.problem == MAX_TREES_PROBLEM:
            # lower and upper bounds of the regulator.
            # The lower is set to the minimum possible value to get one 1 tree. The upper is set to "no bound"
            r_min, r_max = self.regulator, -1
            # variable to indicate the end of iterations for a regulator
            end = False
            # value of objective after working on a regulator
            local_opti = opti_obj
            opti_obj += 1

        while True and search:
            n_trees = sum(w > 0 for w in self.estimator_weights_)
            # print("n_trees", n_trees)
            # n_max_iter or time_limit violated
            if (self.problem == SUITABLE_TREES_PROBLEM and
                    ((self.n_iterations_ >= self.max_iterations > 0) or
                     (time.perf_counter() - start_time >= self.time_limit > 0))):
                if not self.quiet:
                    print("stop condition reached!!!")
                self.optimal_ = False
                break

            # n_max_trees violated
            if self.problem == MAX_TREES_PROBLEM:

                # Greater n_trees. It always means we finished with a regulator value
                if n_trees > self.max_estimators and end:
                    print("great", self.regulator, local_opti, n_trees)

                    clear(self)
                    # if local_opti < opti_obj:
                    #     print("bingo")
                    #     save(self)
                    #     clear(self)

                    if not self.quiet:
                        print("stop condition reached for this regulator !!!")

                    if r_max == -1:  # never encounter upper bound
                        r_max = self.regulator
                        r_min = self.regulator - self.step
                        self.regulator = (r_min + r_max) / 2
                        if r_max - self.regulator <= self.tolerance:
                            restore(self)
                            break
                    else:
                        r_max = self.regulator
                        self.regulator = (r_min + r_max) / 2
                        if r_max - self.regulator <= self.tolerance:
                            restore(self)
                            break

                    # print("new reg", self.regulator)

                    # self.optimal_ = False
                    # break

                # lower and end iteration for a regulator value
                elif n_trees <= self.max_estimators and end:

                    print("low", self.regulator, local_opti, n_trees)
                    # print("low", self.regulator, local_opti, opti_obj, n_trees)
                    end = False

                    # if local_opti < opti_obj:
                    if n_trees > best_n_trees or (n_trees == best_n_trees and local_opti < opti_obj):
                        save(self, n_trees)
                        opti_obj = local_opti
                        # print("cool", local_opti, opti_obj)

                    clear(self)

                    if r_max == -1:  # never encounter upper bound
                        self.regulator = r_min = self.regulator + self.step
                    else:
                        r_min = self.regulator
                        self.regulator = (r_min + r_max) / 2
                        if self.regulator - r_min <= self.tolerance:
                            restore(self)
                            break

                    # print("new reg", self.regulator, opti_obj)

            # We do not reach the number of max_estimators
            if self.model == BOOST_SVM1:
                sample_weights, gamma = self.calculate_sample_weights(sample_weights, y, preds)
            elif self.model == BOOST_SVM2:
                sample_weights, gamma = self.calculate_sample_weights_(sample_weights, y, preds)

            if not self.quiet:
                print("search for new estimator")
            if self.base_estimator is None:
                self.clf_params["time_limit"] = self.clf_params["time_limit"] - (time.perf_counter() - start_time)
                clf = DL85Classifier(**self.clf_params)
            else:
                clf = self.base_estimator

            if self.quiet:
                old_stdout = sys.stdout
                sys.stdout = open(os.devnull, "w")
                clf.fit(X, y, sample_weight=sample_weights)
                sys.stdout = old_stdout
            else:
                clf.fit(X, y, sample_weight=sample_weights)

            # print the tree expression of the estimator if it has
            if hasattr(clf, "tree_") and isinstance(clf.tree_, dict) and not self.quiet:
                print(clf.tree_)

            # compute the prediction of the new estimator : 1 if correct else -1
            try:
                clf_pred = [-1 if p != y[i] else 1 for i, p in enumerate(clf.predict(X))]
            except (NotFittedError, SearchFailedError, TreeNotFoundError) as error:
                if not self.quiet:
                    print("Problem during the search so we stop")
                self.optimal_ = False
                break

            # compute the accuracy of the new estimator based on the weights of samples
            accuracy = sum([sample_weights[tid] * clf_pred[tid] for tid in range(X.shape[0])])
            if not self.quiet:
                print("estimator_accuracy =", accuracy)

            if (self.model == BOOST_SVM1 and accuracy <= gamma + self.opti_gap) or (self.model == BOOST_SVM2 and accuracy <= 1 + self.opti_gap):
                if not self.quiet:
                    print("\n\naccuracy <= gamma", "***END***")

                # in case we are looking for the most suitable n_trees, we are done
                if self.problem == SUITABLE_TREES_PROBLEM:
                    break
                # in case, we are looking for the most suitable regulator, we will check if there is a better one
                else:
                    end = True
                    continue

            # if the new estimator is good to enter into the basis
            self.estimators_.append(clf)
            preds.append(clf_pred)
            if self.model == BOOST_SVM1:
                self.estimator_weights_, rho, local_opti = self.calculate_estimator_weights(y, preds)
            elif self.model == BOOST_SVM2:
                self.estimator_weights_, rho, local_opti = self.calculate_estimator_weights_(y, preds)
            self.n_iterations_ += 1

        self.duration_ = time.perf_counter() - start_time

        # remove the useless estimators
        zero_ind = [i for i, val in enumerate(self.estimator_weights_) if val == 0]
        self.estimator_weights_ = [w for w in self.estimator_weights_ if w != 0]
        self.estimators_ = [clf for clf_id, clf in enumerate(self.estimators_) if clf_id not in zero_ind]
        preds = [clf_pred_vals for clf_pred_id, clf_pred_vals in enumerate(preds) if clf_pred_id not in zero_ind]

        # compute training accuracy of the found ensemble and store it in the variable `accuracy_`
        weighted_train_pred_correct_or_no = [[self.estimator_weights_[clf_id] * preds[clf_id][tid] for tid in range(len(y))] for clf_id in range(len(self.estimators_))]
        train_pred_correct_or_not = [0 if sum(tid_pred) < 0 else 1 for tid_pred in zip(*weighted_train_pred_correct_or_no)]
        self.accuracy_ = sum(train_pred_correct_or_not)/len(y)

        # save the number of found estimators
        self.n_estimators_ = len(self.estimators_)
        # print("n_estim =", self.n_estimators_)

        # Show each non-zero estimator weight and its tree expression if it has
        if not self.quiet:
            for i, estimator in enumerate(sorted(zip(self.estimator_weights_, self.estimators_), key=lambda x: x[0], reverse=True)):
                print("clf n_", i+1, " ==>\tweight: ", estimator[0], sep="", end="")
                if hasattr(estimator[1], "tree_") and isinstance(estimator[1].tree_, dict):
                    print(" \tjson_string: ", estimator[1].tree_, sep="")
                else:
                    print()

        return self

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

    # Primal problem
    def calculate_estimator_weights(self, c, preds):
        if not self.quiet:
            print("\nrun primal_" + str(len(self.estimators_)))
        # the new estimator is already added in get_predict_error before the call to this function
        # initialize the model
        if self.quiet:
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")
        model = Model("estimator_weight_optimiser")
        model.setParam("LogToConsole", 0)
        if self.quiet:
            sys.stdout = old_stdout

        # add variables
        rho = model.addVar(vtype=GRB.CONTINUOUS, name="rho", lb=-GRB.INFINITY)
        error_margin = [model.addVar(vtype=GRB.CONTINUOUS, name="error_margin " + str(i)) for i in range(len(c))]
        new_clf_weights = [model.addVar(vtype=GRB.CONTINUOUS, name="clf_weights " + str(i)) for i in range(len(self.estimators_))]
        # Use last values of estimators weights as warm start
        if not self.estimator_weights_:  # not none, not empty
            for clf_id in range(len(self.estimator_weights_)):
                new_clf_weights[clf_id].setAttr("Start", self.estimator_weights_[clf_id])

        # add constraints
        model.addConstr(quicksum(new_clf_weights) == 1, name="weights = 1")
        for tid in range(len(c)):
            model.addConstr(quicksum([new_clf_weights[clf_id] * preds[clf_id][tid] for clf_id in range(len(self.estimators_))]) + error_margin[tid] >= rho, name="Constraint on sample " + str(tid))

        # add objective function
        model.setObjective(rho - self.regulator * quicksum(error_margin), GRB.MAXIMIZE)
        model.optimize()

        clf_weights = [w.X for w in new_clf_weights]
        rho_ = rho.X
        opti = rho.X - self.regulator * sum(e.X for e in error_margin)

        # print("primal opti =", opti)

        if not self.quiet:
            print("primal opti =", opti, "rho :", rho_, "clfs_w :", clf_weights)

        return clf_weights, rho_, opti

    def calculate_estimator_weights_(self, c, preds):
        if not self.quiet:
            print("\nrun primal_" + str(len(self.estimators_)))
        # the new estimator is already added in get_predict_error before the call to this function
        # initialize the model
        if self.quiet:
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")
        model = Model("estimator_weight_optimiser")
        model.setParam("LogToConsole", 0)
        if self.quiet:
            sys.stdout = old_stdout

        # add variables
        # rho = model.addVar(vtype=GRB.CONTINUOUS, name="rho", lb=float("-inf"))
        error_margin = [model.addVar(vtype=GRB.CONTINUOUS, name="error_margin " + str(i)) for i in range(len(c))]
        new_clf_weights = [model.addVar(vtype=GRB.CONTINUOUS, name="clf_weights " + str(i)) for i in range(len(self.estimators_))]
        # Use last values of estimators weights as warm start
        if not self.estimator_weights_:  # not none, not empty
            for clf_id in range(len(self.estimator_weights_)):
                new_clf_weights[clf_id].setAttr("Start", self.estimator_weights_[clf_id])

        # add constraints
        # model.addConstr(quicksum(new_clf_weights) == 1, name="weights = 1")
        for tid in range(len(c)):
            model.addConstr(quicksum([new_clf_weights[clf_id] * preds[clf_id][tid] for clf_id in range(len(self.estimators_))]) + error_margin[tid] >= 1, name="Constraint on sample " + str(tid))

        # add objective function
        model.setObjective(quicksum(new_clf_weights) + self.regulator * quicksum(error_margin), GRB.MINIMIZE)
        model.optimize()

        clf_weights = [w.X for w in new_clf_weights]
        opti = sum(e.X for e in new_clf_weights) + self.regulator * sum(e.X for e in error_margin)

        # print("primal opti =", opti)

        if not self.quiet:
            print("primal opti =", opti, "clfs_w :", clf_weights, "slacks :", [w.X for w in error_margin])

        return clf_weights, None, opti

    # Dual problem
    def calculate_sample_weights(self, sample_weights, c, preds):
        if not self.quiet:
            print("\nrun dual_" + str(len(self.estimators_)))
        # initialize the model
        if self.quiet:
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")
        model = Model("sample_weight_optimiser")
        model.setParam("LogToConsole", 0)
        if self.quiet:
            sys.stdout = old_stdout

        # add variables
        gamma = model.addVar(vtype=GRB.CONTINUOUS, name="gamma", lb=-GRB.INFINITY)
        new_sample_weights = [model.addVar(vtype=GRB.CONTINUOUS, name="sample_weights " + str(tid), ub=self.regulator if self.regulator > 0 else 1) for tid in range(len(c))]
        # Use last values of examples weights as warm start
        # if not sample_weights:  # not none, not empty
        #     for tid in range(len(sample_weights)):
        #         new_sample_weights[tid].setAttr("Start", sample_weights[tid])

        # add constraints
        model.addConstr(quicksum(new_sample_weights) == 1, name="weights = 1")
        for clf_id in range(len(self.estimators_)):
            model.addConstr(quicksum([new_sample_weights[tid] * preds[clf_id][tid] for tid in range(len(new_sample_weights))]) <= gamma, name="Constraint on estimator " + str(clf_id))

        # add objective function
        model.setObjective(gamma, GRB.MINIMIZE)
        model.optimize()

        ex_weights = [w.X for w in new_sample_weights]
        gamma_ = gamma.X

        # print("dual opti =", gamma_)

        if not self.quiet:
            print("gamma :", gamma_, "new_ex :", ex_weights)

        return ex_weights, gamma_

    def calculate_sample_weights_(self, sample_weights, c, preds):
        if not self.quiet:
            print("\nrun dual_" + str(len(self.estimators_)))
        # initialize the model
        if self.quiet:
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")
        model = Model("sample_weight_optimiser")
        model.setParam("LogToConsole", 0)
        if self.quiet:
            sys.stdout = old_stdout

        # add variables
        # gamma = model.addVar(vtype=GRB.CONTINUOUS, name="gamma", lb=float("-inf"))
        new_sample_weights = [model.addVar(vtype=GRB.CONTINUOUS, name="sample_weights " + str(tid), lb=self.min_trans_cost, ub=self.regulator if self.regulator > 0 else 1) for tid in range(len(c))]
        # Use last values of examples weights as warm start
        if not sample_weights:  # not none, not empty
            for tid in range(len(sample_weights)):
                new_sample_weights[tid].setAttr("Start", sample_weights[tid])

        # add constraints
        # model.addConstr(quicksum(new_sample_weights) == 1, name="weights = 1")
        for clf_id in range(len(self.estimators_)):
            model.addConstr(quicksum([new_sample_weights[tid] * preds[clf_id][tid] for tid in range(len(new_sample_weights))]) <= 1, name="Constraint on estimator " + str(clf_id))

        # add objective function
        model.setObjective(quicksum(new_sample_weights), GRB.MAXIMIZE)
        model.optimize()

        ex_weights = [w.X for w in new_sample_weights]
        opti = sum(e.X for e in new_sample_weights)

        # print("dual opti =", opti)

        if not self.quiet:
            print("gamma :", opti, "new_ex :", ex_weights)

        return ex_weights, opti
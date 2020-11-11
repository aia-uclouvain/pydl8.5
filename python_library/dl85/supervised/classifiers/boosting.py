import os
import random
import sys
import time

from gurobipy import Model, GRB, quicksum
from sklearn.base import ClassifierMixin

from .classifier import DL85Classifier
from ...predictors.predictor import DL85Predictor
from ...errors.errors import SearchFailedError, TreeNotFoundError
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator
from copy import deepcopy


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
            error_function=None,
            fast_error_function=None,
            iterative=False,
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

        self.estimators_ = []
        self.estimator_weights_ = []
        self.accuracy_ = 0
        self.n_estimators_ = 0
        self.optimal_ = True
        self.n_iterations_ = 0

    def fit(self, X, y=None):
        if y is None or len(set(y)) != 2:
            raise ValueError("The \"y\" value is compulsory for boosting and must have two values.")

        start_time = time.perf_counter()
        converted_classes = [-1 if p == 0 else 1 for p in y]
        preds = []
        sample_weights = []
        time_involve = True if self.time_limit > 0 else False

        if self.regulator <= 0:
            self.regulator = 1 / (random.uniform(0, 1) * X.shape[0])

        if not self.quiet:
            print("search for first estimator")
        # clf = None
        if self.base_estimator is None:
            self.clf_params["time_limit"] = self.clf_params["time_limit"] - (time.perf_counter() - start_time)
            #print(self.clf_params["print_output"])
            clf = DL85Classifier(**self.clf_params)
        else:
            clf = self.base_estimator

        if self.quiet:
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")
        clf.fit(X, y)
        if self.quiet:
            sys.stdout = old_stdout

        # print the tree expression of the estimator if it has
        if hasattr(clf, "tree_") and isinstance(clf.tree_, dict) and not self.quiet:
            print(clf.tree_)

        # add the current estimator to the ensemble
        self.estimators_.append(clf)

        # save the prediction of the estimator
        preds.append([-1 if p == 0 else 1 for p in clf.predict(X)])

        # compute the weights of the estimator
        self.estimator_weights_, rho = self.calculate_estimator_weights(converted_classes, preds)

        # for i in range(self.max_estimators - 1):
        self.n_iterations_ += 1
        while True:
            # print("n_iter :", len(self.estimator_weights_), "nvalid :", sum(w > 0 for w in self.estimator_weights_), "max :", self.max_estimators)
            if (sum(w > 0 for w in self.estimator_weights_) >= self.max_estimators > 0) or (self.n_iterations_ >= self.max_iterations > 0) or (time.perf_counter() - start_time >= self.time_limit > 0):
                if not self.quiet:
                    print("stop condition reached!!!")
                self.optimal_ = False
                break

            # We do not reach the number of max_estimators
            sample_weights, gamma = self.calculate_sample_weights(sample_weights, converted_classes, preds)

            if not self.quiet:
                print("search for new estimator")
            if self.base_estimator is None:
                # clf = DL85Classifier(max_depth=self.max_depth, min_sup=self.min_sup, time_limit=self.time_limit)  # , print_output=True)
                self.clf_params["time_limit"] = self.clf_params["time_limit"] - (time.perf_counter() - start_time)
                clf = DL85Classifier(**self.clf_params)
            else:
                clf = self.base_estimator

            if self.quiet:
                old_stdout = sys.stdout
                sys.stdout = open(os.devnull, "w")
            clf.fit(X, y, sample_weight=sample_weights)
            if self.quiet:
                sys.stdout = old_stdout

            # Error function for DL8
            # def weighted_error(tids):
            #     all_classes = [0, 1]
            #     supports = [0, 0]
            #     for tid in tids:
            #         supports[y[tid]] += self.example_weights[tid]
            #     maxindex = supports.index(max(supports))
            #     return sum(supports) - supports[maxindex], all_classes[maxindex]
            # tree_clf = DL85Classifier(max_depth=self.max_depth, min_sup=self.min_sup, print_output=True, error_function=lambda tree: weighted_error(tree))
            # tree_clf.fit(X, y)

            # print the tree expression of the estimator if it has
            if hasattr(clf, "tree_") and isinstance(clf.tree_, dict) and not self.quiet:
                print(clf.tree_)

            # compute prediction of the new estimator
            try:
                clf_pred = [-1 if p == 0 else 1 for p in clf.predict(X)]
            except (NotFittedError, SearchFailedError, TreeNotFoundError) as error:
                if not self.quiet:
                    print("Problem during the search so we stop")
                self.optimal_ = False
                break
            # compute its accuracy based on the weights of samples
            accuracy = sum([converted_classes[tid] * sample_weights[tid] * clf_pred[tid] for tid in range(X.shape[0])])
            if not self.quiet:
                print("estimator_accuracy =", accuracy)

            if accuracy <= gamma:
                if not self.quiet:
                    print("\n\naccuracy <= gamma", "***END***")
                break

            # if the new estimator is good to enter into the basis
            self.estimators_.append(clf)
            preds.append(clf_pred)
            # print("n_w bef :", len(self.estimator_weights_))
            self.estimator_weights_, rho = self.calculate_estimator_weights(converted_classes, preds)
            # print("n_w aft :", len(self.estimator_weights_))
            self.n_iterations_ += 1

        # remove the useless estimators
        zero_ind = [i for i, val in enumerate(self.estimator_weights_) if val == 0]
        self.estimator_weights_ = [w for w in self.estimator_weights_ if w != 0]
        self.estimators_ = [clf for clf_id, clf in enumerate(self.estimators_) if clf_id not in zero_ind]
        preds = [clf_pred_vals for clf_pred_id, clf_pred_vals in enumerate(preds) if clf_pred_id not in zero_ind]

        # compute training accuracy of the found ensemble and store it in the variable `accuracy_`
        weighted_train_pred = [[self.estimator_weights_[clf_id] * preds[clf_id][tid] for tid in range(len(y))] for clf_id in range(len(self.estimators_))]
        train_pred = [0 if sum(tid_pred) < 0 else 1 for tid_pred in zip(*weighted_train_pred)]
        self.accuracy_ = sum(p == y[i] for i, p in enumerate(train_pred))/len(y)

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

    def predict(self, X, y=None):
        if self.n_estimators_ == 0:  # fit method has not been called
            raise NotFittedError("Call fit method first" % {'name': type(self).__name__})
        # Run a prediction on each estimator in term of 0/1
        predict_per_clf = [clf.predict(X) for clf_id, clf in enumerate(self.estimators_)]
        # Convert 0/1 prediction into -1/1
        predict_per_clf = [[-1 if p == 0 else 1 for p in row] for row in predict_per_clf]
        # Apply the estimator weight on each prediction
        weighted_predict_per_clf = [[self.estimator_weights_[clf_id] * predict_per_clf[clf_id][tid] for tid in range(X.shape[0])] for clf_id in range(len(self.estimators_))]
        # Compute the prediction based on all estimators in term of 0/1
        pred = [0 if sum(tid_pred) < 0 else 1 for tid_pred in zip(*weighted_predict_per_clf)]
        return pred

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
        rho = model.addVar(vtype=GRB.CONTINUOUS, name="rho", lb=float("-inf"))
        error_margin = [model.addVar(vtype=GRB.CONTINUOUS, name="error_margin " + str(i)) for i in range(len(c))]
        new_clf_weights = [model.addVar(vtype=GRB.CONTINUOUS, name="clf_weights " + str(i)) for i in range(len(self.estimators_))]
        # Use last values of estimators weights as warm start
        if not self.estimator_weights_:  # not none, not empty
            for clf_id in range(len(self.estimator_weights_)):
                new_clf_weights[clf_id].setAttr("Start", self.estimator_weights_[clf_id])

        # add constraints
        model.addConstr(quicksum(new_clf_weights) == 1, name="weights = 1")
        for tid in range(len(c)):
            model.addConstr(quicksum([c[tid] * new_clf_weights[clf_id] * preds[clf_id][tid] for clf_id in range(len(self.estimators_))]) + error_margin[tid] >= rho, name="Constraint on sample " + str(tid))

        # add objective function
        model.setObjective(rho - self.regulator * quicksum(error_margin), GRB.MAXIMIZE)
        model.optimize()

        clf_weights = [w.X for w in new_clf_weights]
        rho_ = rho.X
        opti = rho.X - self.regulator * sum(e.X for e in error_margin)

        if not self.quiet:
            print("primal opti =", opti, "rho :", rho_, "clfs_w :", clf_weights)

        return clf_weights, rho_

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
        gamma = model.addVar(vtype=GRB.CONTINUOUS, name="gamma", lb=float("-inf"))
        new_sample_weights = [model.addVar(vtype=GRB.CONTINUOUS, name="sample_weights " + str(tid), ub=self.regulator if self.regulator > 0 else 1) for tid in range(len(c))]
        # Use last values of examples weights as warm start
        if not sample_weights:  # not none, not empty
            for tid in range(len(sample_weights)):
                new_sample_weights[tid].setAttr("Start", sample_weights[tid])

        # add constraints
        model.addConstr(quicksum(new_sample_weights) == 1, name="weights = 1")
        for clf_id in range(len(self.estimators_)):
            model.addConstr(quicksum([c[tid] * new_sample_weights[tid] * preds[clf_id][tid] for tid in range(len(new_sample_weights))]) <= gamma, name="Constraint on estimator " + str(clf_id))

        # add objective function
        model.setObjective(gamma, GRB.MINIMIZE)
        model.optimize()

        ex_weights = [w.X for w in new_sample_weights]
        gamma_ = gamma.X

        if not self.quiet:
            print("gamma :", gamma_, "new_ex :", ex_weights)

        return ex_weights, gamma_

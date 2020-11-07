from sklearn.base import ClassifierMixin
from ...predictors.predictor import DL85Predictor
from .classifier import DL85Classifier
from gurobipy import Model, GRB, quicksum
import json
import random


class DL85Booster(DL85Predictor, ClassifierMixin):
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
            print_output=False):

        self.base_estimator = base_estimator
        self.max_depth = max_depth
        self.min_sup = min_sup
        self.max_estimators = max_estimators
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

        self.estimators = []
        self.sample_weights = []
        self.estimator_weights = []
        self.c = []
        self.preds = []
        self.D = regulator
        self.accuracy_ = 0
        self.n_estimators_ = 0

        DL85Predictor.__init__(self,
                               max_depth=self.max_depth,
                               min_sup=self.min_sup,
                               max_estimators=self.max_estimators,
                               error_function=self.error_function,
                               fast_error_function=self.fast_error_function,
                               iterative=self.iterative,
                               max_error=self.max_error,
                               stop_after_better=self.stop_after_better,
                               time_limit=self.time_limit,
                               verbose=self.verbose,
                               desc=self.desc,
                               asc=self.asc,
                               repeat_sort=self.repeat_sort,
                               print_output=self.print_output)

    def fit(self, X, y=None):
        if y is None or len(set(y)) > 2:
            raise ValueError("The \"y\" value is compulsory for boosting and must have two values.")

        self.c = [-1 if p == 0 else 1 for p in y]
        if self.D <= 0:
            self.D = 1/(random.uniform(0, 1) * X.shape[0])
            # self.D = 0.2

        print("search for first estimator")
        clf = None
        if self.base_estimator is None:
            clf = DL85Classifier(max_depth=self.max_depth, min_sup=self.min_sup, time_limit=self.time_limit)  # , print_output=True)
        else:
            clf = self.base_estimator
        clf.fit(X, y)

        # print the tree expression of the estimator if it has
        if hasattr(clf, "tree_"):
            print(clf.tree_)

        # add the current estimator to the ensemble
        self.estimators.append(clf)

        # save the prediction of the estimator
        self.preds.append([-1 if p == 0 else 1 for p in clf.predict(X)])

        # compute the weights of the estimator
        self.estimator_weights, rho = self.calculate_estimator_weights()

        # for i in range(self.max_estimators - 1):
        while True:
            if sum(w > 0 for w in self.estimator_weights) >= self.max_estimators > 0:  # n_estimators > max_estimators
                print("max_estimators reached!!!")
                break

            # We do not reach the number of max_estimators
            self.sample_weights, gamma = self.calculate_sample_weights()

            print("search for new estimator")
            if self.base_estimator is None:
                tree_clf = DL85Classifier(max_depth=self.max_depth, min_sup=self.min_sup, time_limit=self.time_limit)  # , print_output=True)
            else:
                tree_clf = self.base_estimator
            tree_clf.fit(X, y, sample_weight=self.sample_weights)

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
            if hasattr(clf, "tree_"):
                print(clf.tree_)

            # compute prediction of the new estimator
            clf_pred = [-1 if p == 0 else 1 for p in clf.predict(X)]
            # compute its accuracy based on the weights of samples
            accuracy = sum([self.c[tid] * self.sample_weights[tid] * clf_pred[tid] for tid in range(X.shape[0])])
            print("estimator_accuracy =", accuracy)

            if accuracy <= gamma:
                print("\n\naccuracy <= gamma", "***END***")
                break

            # if the new estimator is good to enter into the basis
            self.estimators.append(clf)
            self.preds.append(clf_pred)
            self.estimator_weights, rho = self.calculate_estimator_weights()

        # compute training accuracy of the found ensemble and store it in the variable `accuracy_`
        weighted_train_pred = [[self.estimator_weights[clf_id] * self.preds[clf_id][tid] for tid in range(len(y))] for clf_id in range(len(self.estimators))]
        train_pred = [0 if sum(tid_pred) < 0 else 1 for tid_pred in zip(*weighted_train_pred)]
        self.accuracy_ = sum(p == y[i] for i, p in enumerate(train_pred))/len(y)

        # save the number of found estimators
        self.n_estimators_ = sum(w > 0 for w in self.estimator_weights)

        # Show each non-zero trees with its weight
        for i, estimator in enumerate(sorted([elt for elt in list(zip(self.estimators, self.estimator_weights)) if elt[1] > 0], key=lambda x: x[1], reverse=True)):
            print("tree n_", i+1, " ==>\tweight: ", estimator[1], sep="", end="")
            if hasattr(estimator[0], "tree_"):
                print(" \tjson_string: ", estimator[0], sep="")
            else:
                print()
        # for i, tree in enumerate(self.trees):
        #     if self.tree_weights[i] > 0:
        #         print("tree:",  tree, "weight:", self.tree_weights[i])
        return self

    def predict(self, X, y=None):
        # Run a prediction on each tree in term of 0/1
        predict_per_clf = [clf.predict(X) for clf_id, clf in enumerate(self.estimators) if self.estimator_weights[clf_id] > 0]
        # Convert 0/1 prediction into -1/1
        predict_per_clf = [[-1 if p == 0 else 1 for p in row] for row in predict_per_clf]
        # Apply the tree weight on each prediction
        estimator_weights = [w for w in self.estimator_weights if w > 0]
        weighted_predict_per_clf = [[estimator_weights[clf_id] * predict_per_clf[clf_id][tid] for tid in range(len(self.c))] for clf_id in range(len(estimator_weights))]
        # Compute the prediction based on all trees in term of 0/1
        pred = [0 if sum(tid_pred) < 0 else 1 for tid_pred in zip(*weighted_predict_per_clf)]
        return pred

    # Primal problem
    def calculate_estimator_weights(self):
        print("\nrun primal_" + str(len(self.estimators)))
        # the new tree is already added in get_predict_error before the call to this function
        # initialize the model
        model = Model("tree_weight_optimiser")
        model.setParam("LogToConsole", 0)

        # add variables
        rho = model.addVar(vtype=GRB.CONTINUOUS, name="rho", lb=float("-inf"))
        error_margin = [model.addVar(vtype=GRB.CONTINUOUS, name="error_margin " + str(i)) for i in range(len(self.c))]
        new_clf_weights = [model.addVar(vtype=GRB.CONTINUOUS, name="clf_weights " + str(i)) for i in range(len(self.estimators))]
        # Use last values of trees weights as warm start
        if not self.estimator_weights:  # not none, not empty
            for clf_id in range(len(self.estimator_weights)):
                new_clf_weights[clf_id].setAttr("Start", self.estimator_weights[clf_id])

        # add constraints
        model.addConstr(quicksum(new_clf_weights) == 1, name="weights = 1")
        for tid in range(len(self.c)):
            model.addConstr(quicksum([self.c[tid] * new_clf_weights[clf_id] * self.preds[clf_id][tid] for clf_id in range(len(self.estimators))]) + error_margin[tid] >= rho, name="Constraint on sample " + str(tid))

        # add objective function
        model.setObjective(rho - self.D * quicksum(error_margin), GRB.MAXIMIZE)
        model.optimize()

        clf_weights = [w.X for w in new_clf_weights]
        rho_ = rho.X
        opti = rho.X - self.D * sum(e.X for e in error_margin)

        print("primal opti =", opti, "rho :", rho_, "clfs_w :", clf_weights)

        return clf_weights, rho_

    # Dual problem
    def calculate_sample_weights(self):
        print("\nrun dual_" + str(len(self.estimators)))
        # initialize the model
        model = Model("example_weight_optimiser")
        model.setParam("LogToConsole", 0)

        # add variables
        gamma = model.addVar(vtype=GRB.CONTINUOUS, name="gamma", lb=float("-inf"))
        new_example_weights = [model.addVar(vtype=GRB.CONTINUOUS, name="sample_weights " + str(tid), ub=self.D if self.D > 0 else 1) for tid in range(len(self.c))]
        # Use last values of examples weights as warm start
        if not self.sample_weights:  # not none, not empty
            for tid in range(len(self.sample_weights)):
                new_example_weights[tid].setAttr("Start", self.sample_weights[tid])

        # add constraints
        model.addConstr(quicksum(new_example_weights) == 1, name="weights = 1")
        for clf_id in range(len(self.estimators)):
            model.addConstr(quicksum([self.c[tid] * new_example_weights[tid] * self.preds[clf_id][tid] for tid in range(len(new_example_weights))]) <= gamma, name="Constraint on tree " + str(clf_id))

        # add objective function
        model.setObjective(gamma, GRB.MINIMIZE)
        model.optimize()

        ex_weights = [w.X for w in new_example_weights]
        gamma_ = gamma.X

        print("gamma :", gamma_, "new_ex :", ex_weights)

        return ex_weights, gamma_

    def predict_one_tree(self, X, tree):
        p = []
        for i in range(X.shape[0]):
            if tree is None:
                p.append(self.pred_value_on_dict(X[i, :], tree))
            else:
                p.append(self.pred_value_on_dict(X[i, :], tree))
        return p

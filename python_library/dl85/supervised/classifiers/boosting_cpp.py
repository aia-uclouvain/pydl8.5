from sklearn.base import ClassifierMixin
from ...predictors.predictor import DL85Predictor
from gurobipy import Model, GRB, quicksum
import json


class DL85BoosterC(DL85Predictor, ClassifierMixin):
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

    # @staticmethod
    def intermediary_test_error_function(self):
        return self.intermediary_test_error
        # return 1.25

    def __init__(
            self,
            max_depth=1,
            min_sup=1,
            max_estimators=2,
            error_function=None,
            fast_error_function=None,
            iterative=False,
            max_error=0,
            stop_after_better=False,
            time_limit=0,
            verbose=False,
            desc=False,
            asc=False,
            repeat_sort=False,
            print_output=False):

        self.trees = []
        self.example_weights = []
        self.tree_weights = []
        self.c = []
        self.y = []
        self.preds = []
        self.D = 0.2
        DL85Predictor.__init__(self,
                               max_depth=max_depth,
                               min_sup=min_sup,
                               max_estimators=max_estimators,
                               error_function=error_function,
                               fast_error_function=fast_error_function,
                               example_weight_function=None,
                               predict_error_function=None,
                               iterative=iterative,
                               max_error=max_error,
                               stop_after_better=stop_after_better,
                               time_limit=time_limit,
                               verbose=verbose,
                               desc=desc,
                               asc=asc,
                               repeat_sort=repeat_sort,
                               leaf_value_function=None,
                               print_output=print_output)

    def fit(self, X, y=None):
        if y is None or len(set(y)) > 2:
            raise ValueError("The \"y\" value is compulsory for boosting and must have two values.")
        if self.max_estimators <= 1:
            raise ValueError("You need at least 2 estimators to perform a boosting")

        self.predict_error_function = lambda tree: self.get_predict_error(tree, X)
        self.example_weight_function = self.get_example_weights
        self.c = [-1 if p == 0 else 1 for p in y]
        self.y = y
        print('eeee =', len(self.c))

        DL85Predictor.fit(self, X, y)

    def predict(self, X, y=None):
        # Run a (weighted) prediction on all trees
        predict_per_tree = [self.get_predict_error(tree, X) for tree in self.trees]
        predict_per_tree = [[-1 if p == 0 else 1 for p in row] for row in predict_per_tree]
        weighted_predict_per_tree = [[self.tree_weights[tree] * predict_per_tree[tree][tid] for tid in range(len(predict_per_tree[tree]))] for tree in range(len(self.trees))]
        pred = [0 if sum(tid_predictions) < 0 else 1 for tid_predictions in zip(*weighted_predict_per_tree)]
        return pred

    def get_predict_error(self, tree, X):
        print("a")
        print(type(tree))
        print(tree)
        tree = json.loads(tree.decode("utf-8"))
        self.trees.append(tree)
        tree_pred = [-1 if p == 0 else 1 for p in self.predict_one_tree(X, tree)]
        self.preds.append(tree_pred)
        self.tree_weights, rho = self.calculate_tree_weights()
        all_tree_pred = self.predict(X)
        return sum(1 for i, j in zip(self.y, all_tree_pred) if i != j), rho

    def calculate_tree_weights(self):
        # Primal problem
        # the new tree is already added in get_predict_error before the call to this function

        # initialize the model
        model = Model("tree_weight_optimiser")
        model.setParam("LogToConsole", 0)

        # add variables
        rho = model.addVar(vtype=GRB.CONTINUOUS, name="rho", lb=float("-inf"))
        error_margin = [model.addVar(vtype=GRB.CONTINUOUS, name="error_margin " + str(i)) for i in range(len(self.c))]
        new_tree_weights = [model.addVar(vtype=GRB.CONTINUOUS, name="tree_weights " + str(i)) for i in range(len(self.trees))]
        # Use last values of trees weights as warm start
        if not self.tree_weights:  # not none, not empty
            for i in range(len(self.tree_weights)):
                new_tree_weights[i].setAttr("Start", self.tree_weights[i])

        # add constraints
        model.addConstr(quicksum(new_tree_weights) == 1, name="weights = 1")
        for tid in range(len(self.c)):
            model.addConstr(quicksum([self.c[tid] * new_tree_weights[tree] * self.preds[tree][tid] for tree in range(len(self.trees))]) + error_margin[tid] >= rho, name="Constraint on sample " + str(tid))

        # add objective function
        model.setObjective(rho - self.D * quicksum(error_margin), GRB.MAXIMIZE)
        model.optimize()

        return [w.X for w in new_tree_weights], rho.X

    def get_example_weights(self):
        # pred = [[-1 if p == 0 else 1 for p in DL85Predictor.predict(X, tree)] for tree in self.trees]
        self.example_weights, gamma = self.calculate_example_weights()
        return self.example_weights + [gamma]

    def calculate_example_weights(self):
        # Dual problem

        # initialize the model
        model = Model("example_weight_optimiser")
        model.setParam("LogToConsole", 0)

        # add variables
        gamma = model.addVar(vtype=GRB.CONTINUOUS, name="gamma", lb=float("-inf"))
        new_example_weights = [model.addVar(vtype=GRB.CONTINUOUS, name="sample_weights " + str(i), ub=self.D if self.D > 0 else 1) for i in range(len(self.c))]
        # Use last values of examples weights as warm start
        if not self.example_weights:  # not none, not empty
            for i in range(len(self.example_weights)):
                new_example_weights[i].setAttr("Start", self.example_weights[i])

        print("siiiize =", len(new_example_weights))
        print("siiiize =", len(self.c))

        # add constraints
        model.addConstr(quicksum(new_example_weights) == 1, name="weights = 1")
        for tree in range(len(self.trees)):
            model.addConstr(quicksum([self.c[tid] * new_example_weights[tid] * self.preds[tree][tid] for tid in range(len(new_example_weights))]) <= gamma, name="Constraint on tree " + str(tree))

        # add objective function
        model.setObjective(gamma, GRB.MINIMIZE)
        model.write("model.lp")
        model.optimize()

        return [w.X for w in new_example_weights], gamma.X

    def predict_one_tree(self, X, tree):
        p = []
        for i in range(X.shape[0]):
            if tree is None:
                p.append(self.pred_value_on_dict(X[i, :], tree))
            else:
                p.append(self.pred_value_on_dict(X[i, :], tree))
        return p

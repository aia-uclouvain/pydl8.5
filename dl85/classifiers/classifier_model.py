import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import dl85Optimizer
from ..errors.errors import SearchFailedError, TreeNotFoundError
import ast
import json
import random


class OptimalDecisionTreesClassifier(BaseEstimator, ClassifierMixin):
    """ An optimal decision tree classifier.

    Parameters
    ----------
    max_depth : int, default=1
        Maximum depth of the tree to be found
    min_sup : int, default=1
        Minimum number of examples per leaf
    max_error : int, default=0
        Maximum error that the searched tree cannot reach. Default value stand for no bound
    stop_after_better : bool, default=False
        A parameter used to indicate if the search will stop after finding tree better than max_error
    time_limit : int, default=0
        Allocated time in second(s) for the search. Default value stands for no limit
    verbose : bool, default=False
        A parameter used to switch on/off the print of what happens during the search
    desc : bool, default=False
        A parameter used to indicate if the sorting of the items is done in descendent order over the information gain
    asc : bool, default=False
        A parameter used to indicate if the sorting of the items is done in ascendant order over the information gain
    repeat_sort : bool, default=False
        A parameter used to indicate the sorting of items is done at each level of the lattice or only before the search
    bin_save : bool, default=False
        A parameter used to indicate the continuous dataset will just be discretized and export without search
    nps : bool, default=False
        A parameter used to indicate if only optimal solutions which will be reuse or not

    Attributes
    ----------
    tree_ : str
        Outputted tree in serialized form
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
            max_depth=1,
            min_sup=1,
            iterative=False,
            max_error=0,
            stop_after_better=False,
            time_limit=0,
            verbose=False,
            desc=False,
            asc=False,
            repeat_sort=False,
            bin_save=False,
            nps=False):
        self.max_depth = max_depth
        self.min_sup = min_sup
        self.iterative = iterative
        self.max_error = max_error
        self.stop_after_better = stop_after_better
        self.time_limit = time_limit
        self.verbose = verbose
        self.desc = desc
        self.asc = asc
        self.repeat_sort = repeat_sort
        self.continuous = False
        self.bin_save = bin_save
        self.nps = nps

    # def _more_tags(self):
    #     return {'X_types': 'categorical',
    #             'allow_nan': False}

    def fit(self, X, y):
        """A reference implementation of a fitting function for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """

        # Check that X and y have correct shape and raise ValueError if not
        X, y = check_X_y(X, y)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        # np.savetxt("foo" + str(random.randint(0,100)) + ".csv", X, delimiter=",")
        solution = dl85Optimizer.solve(X, y, self.max_depth, self.min_sup, self.max_error, self.stop_after_better,
                                       self.iterative, self.time_limit, self.verbose, self.desc, self.asc,
                                       self.repeat_sort, self.continuous, self.bin_save, self.nps)

        # print(type(solution))
        # print(solution)
        # assert (isinstance(solution, str))
        solution = solution.splitlines()
        sol_size = len(solution)

        if sol_size == 1:
            raise ValueError(solution[0])

        if sol_size == 8 or sol_size == 9:  # solution found
            self.tree_ = ast.literal_eval(solution[1].split('Tree: ')[1])
            self.size_ = int(solution[2].split(" ")[1])
            self.depth_ = int(solution[3].split(" ")[1])
            self.error_ = float(solution[4].split(" ")[1])
            self.accuracy_ = float(solution[5].split(" ")[1])
            # print("error =", self.error_)
            if sol_size == 8:  # without timeout
                print("DL8.5 fitting: Solution found")
                self.lattice_size_ = int(solution[6].split(" ")[1])
                self.runtime_ = float(solution[7].split(" ")[1])
                self.timeout_ = False
            else:  # timeout reached
                print("DL8.5 fitting: Timeout reached but solution found")
                self.lattice_size_ = int(solution[7].split(" ")[1])
                self.runtime_ = float(solution[8].split(" ")[1])
                self.timeout_ = True

        elif sol_size == 4 or sol_size == 5:  # solution not found
            self.tree_ = False
            self.size_ = -1
            self.depth_ = -1
            self.error_ = -1
            self.accuracy_ = -1
            if sol_size == 4:  # without timeout
                print("DL8.5 fitting: Solution not found")
                self.lattice_size_ = int(solution[2].split(" ")[1])
                self.runtime_ = float(solution[3].split(" ")[1])
                self.timeout_ = False
            else:  # timeout reached
                print("DL8.5 fitting: Timeout reached and solution not found")
                self.lattice_size_ = int(solution[3].split(" ")[1])
                self.runtime_ = float(solution[4].split(" ")[1])
                self.timeout_ = True

        # Return the classifier
        return self

    def predict(self, X):
        """ A reference implementation of a prediction for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """

        if hasattr(self, 'tree_') is False:  # actually this case is not possible.
            raise SearchFailedError("PredictionError: ", "DL8.5 training has failed")
            # return None

        # Check is fit had been called
        # check_is_fitted(self, ['X_', 'y_'])
        check_is_fitted(self, 'tree_')

        if self.tree_ is False:
            raise TreeNotFoundError("predict(): ", "Tree not found during training by DL8.5")
            # return None

        # Input validation
        X = check_array(X)

        self.y_ = []

        for i in range(X.shape[0]):
            self.y_.append(self.pred_on_dict(X[i, :]))

        return self.y_

        # closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        # return self.y_[closest]

    # def get_params(self, deep=True):
    #     params = {}
    #     params['max_depth'] = self.max_depth
    #     params['min_sup'] = self.min_sup
    #     params['max_error'] = self.max_error
    #     params['time'] = self.time
    #     params['verbose'] = self.verbose
    #     params['desc'] = self.desc
    #     params['asc'] = self.asc
    #     params['repeatSort'] = self.repeatSort
    #     params['continuous'] = self.continuous
    #     params['binSave'] = self.binSave
    #     params['nps'] = self.nps
    #     return params
    #
    # def set_params(self, **params):
    #     for parameter, value in params.items():
    #         setattr(self, parameter, value)
    #     return self

    def pred_on_dict(self, instance):
        node = self.tree_
        while self.is_leaf_node(node) is not True:
            if instance[node['feat']] == 1:
                node = node['left']
            else:
                node = node['right']
        return node['class']

    @staticmethod
    def is_leaf_node(node):
        return list(node.items())[0][0] == 'class'

    # @staticmethod
    # def tree2graph(data, verbose=True):
    #     """
    #     Convert a JSON to a graph.
    #
    #     Run `dot -Tpng -otree.png`
    #
    #     Parameters
    #     ----------
    #     json_filepath : str
    #         Path to a JSON file
    #     out_dot_path : str
    #         Path where the output dot file will be stored
    #
    #     Examples
    #     --------
    #     >>> s = {"Harry": [ "Bill", \
    #                        {"Jane": [{"Diane": ["Mary", "Mark"]}]}]}
    #     >>> tree2graph(s)
    #     [('Harry', 'Bill'), ('Harry', 'Jane'), ('Jane', 'Diane'), ('Diane', 'Mary'), ('Diane', 'Mark')]
    #     """
    #     # Extract tree edges from the dict
    #     edges = []
    #
    #     def get_edges(treedict, parent=None):
    #         name = next(iter(treedict.keys()))
    #         if parent is not None:
    #             edges.append((parent, name))
    #         for item in treedict[name]:
    #             if isinstance(item, dict):
    #                 get_edges(item, parent=name)
    #             elif isinstance(item, list):
    #                 for el in item:
    #                     if isinstance(item, dict):
    #                         edges.append((parent, item.keys()[0]))
    #                         get_edges(item[item.keys()[0]])
    #                     else:
    #                         edges.append((parent, el))
    #             else:
    #                 edges.append((name, item))
    #
    #     get_edges(data)
    #     return edges
    #
    # def export_graphviz(self, lr=False, verbose=False):
    #     check_is_fitted(self, 'tree_')
    #     if verbose:
    #         # Convert back to JSON & print to stderr so we can verfiy that the tree
    #         # is correct.
    #         print(json.dumps(self.tree_, indent=4))
    #
    #     # Get edges
    #     edges = self.tree2graph(self.tree_, verbose)
    #
    #     # Dump edge list in Graphviz DOT format
    #     dot_string = ""
    #     dot_string += 'strict digraph tree {\n'
    #     if lr:
    #         dot_string += 'rankdir="LR";\n'
    #     for row in edges:
    #         dot_string += '    "{0}" -> "{1}";\n'.format(*row)
    #     dot_string += '}\n'
    #
    #     return dot_string
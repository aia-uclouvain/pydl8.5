#from sklearn.base import BaseEstimator, ClassifierMixin
#from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
#from sklearn.utils.multiclass import unique_labels
from ..errors.errors import SearchFailedError, TreeNotFoundError
import json
import numpy as np


class DL85Predictor:
    """ An optimal binary decision tree classifier.

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
    desc_sort_function : function, default=None
        A parameter used to indicate heuristic function used to sort the items in descending order
    asc_sort_function : function, default=None
        A parameter used to indicate heuristic function used to sort the items in ascending order
    repeat_sort : bool, default=False
        A parameter used to indicate whether the heuristic sort will be applied at each level of the lattice or only at the root
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
            max_depth=1,
            min_sup=1,
            error_function=None,
            iterative=False,
            max_error=0,
            stop_after_better=False,
            time_limit=0,
            verbose=False,
            desc_sort_function=None,
            asc_sort_function=None,
            repeat_sort=False,
            leaf_value_function=None,
            nps=False,
            print_output=False):
        self.max_depth = max_depth
        self.min_sup = min_sup
        self.error_function = error_function
        self.iterative = iterative
        self.max_error = max_error
        self.stop_after_better = stop_after_better
        self.time_limit = time_limit
        self.verbose = verbose
        self.desc_sort_function = desc_sort_function
        self.asc_sort_function = asc_sort_function
        self.repeat_sort = repeat_sort
        self.leaf_value_function = leaf_value_function
        self.nps = nps
        self.print_output = print_output

    def predictor_fit(self, X):
        """Implements the standard fitting function for a DL8.5 classifier.

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

        # sys.path.insert(0, "../../")
        import dl85Optimizer
        solution = dl85Optimizer.solve(data=X,
                                       target=None,
                                       func=None,
                                       fast_func=None,
                                       predictor_func=self.error_function,
                                       max_depth=self.max_depth,
                                       min_sup=self.min_sup,
                                       max_error=self.max_error,
                                       stop_after_better=self.stop_after_better,
                                       iterative=self.iterative,
                                       time_limit=self.time_limit,
                                       verb=self.verbose,
                                       desc=self.desc_sort_function,
                                       asc=self.asc_sort_function,
                                       repeat_sort=self.repeat_sort,
                                       bin_save=False,
                                       nps=self.nps,
                                       predictor=True)

        if self.print_output:
            print(solution)

        solution = solution.splitlines()
        self.sol_size = len(solution)

        # if self.sol_size_ == 1:
        #     raise ValueError(solution[0])

        if self.sol_size == 8 or self.sol_size == 9:  # solution found
            self.tree_ = json.loads(solution[1].split('Tree: ')[1])
            self.size_ = int(solution[2].split(" ")[1])
            self.depth_ = int(solution[3].split(" ")[1])
            self.error_ = float(solution[4].split(" ")[1])
            self.accuracy_ = float(solution[5].split(" ")[1])

            if self.sol_size == 8:  # without timeout
                print("DL8.5 fitting: Solution found")
                self.lattice_size_ = int(solution[6].split(" ")[1])
                self.runtime_ = float(solution[7].split(" ")[1])
                self.timeout_ = False
            else:  # timeout reached
                print("DL8.5 fitting: Timeout reached but solution found")
                self.lattice_size_ = int(solution[7].split(" ")[1])
                self.runtime_ = float(solution[8].split(" ")[1])
                self.timeout_ = True

        elif self.sol_size == 4 or self.sol_size == 5:  # solution not found
            self.tree_ = False
            self.size_ = -1
            self.depth_ = -1
            self.error_ = -1
            self.accuracy_ = -1
            if self.sol_size == 4:  # without timeout
                print("DL8.5 fitting: Solution not found")
                self.lattice_size_ = int(solution[2].split(" ")[1])
                self.runtime_ = float(solution[3].split(" ")[1])
                self.timeout_ = False
            else:  # timeout reached
                print("DL8.5 fitting: Timeout reached and solution not found")
                self.lattice_size_ = int(solution[3].split(" ")[1])
                self.runtime_ = float(solution[4].split(" ")[1])
                self.timeout_ = True

        if hasattr(self, 'tree_'):
            # add transactions to nodes of the tree
            self.tree_dfs(X)

            if self.leaf_value_function is not None:
                def search(node):
                    if self.is_leaf_node(node) is not True:
                        search(node['left'])
                        search(node['right'])
                    else:
                        node['value'] = self.leaf_value_function(node['transactions'])
                node = self.tree_
                search(node)

        # Return the classifier
        # return self

    def predictor_predict(self, X):
        """ Implements the standard predict function for a DL8.5 classifier.

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

        y = []

        for i in range(X.shape[0]):
            y.append(self.pred_on_dict(X[i, :]))

        return y

    def pred_on_dict(self, instance):
        node = self.tree_
        while self.is_leaf_node(node) is not True:
            if instance[node['feat']] == 1:
                node = node['left']
            else:
                node = node['right']
        return node['value']

    @staticmethod
    def is_leaf_node(node):
        names = [x[0] for x in node.items()]
        return 'error' in names

    def tree_dfs(self, X):

        def recurse(transactions, node, feature, positive):
            if transactions is None:
                current_transactions = list(range(0, X.shape[0]))
                node['transactions'] = current_transactions
                if 'feat' in node.keys():
                    recurse(current_transactions, node['left'], node['feat'], True)
                    recurse(current_transactions, node['right'], node['feat'], False)
            else:
                feature_vector = X[:, feature]
                feature_vector = feature_vector.astype('int32')
                if positive:
                    positive_vector = np.where(feature_vector == 1)[0]
                    positive_vector = positive_vector.tolist()
                    current_transactions = set(transactions).intersection(positive_vector)
                    node['transactions'] = list(current_transactions)
                    if 'feat' in node.keys():
                        recurse(current_transactions, node['left'], node['feat'], True)
                        recurse(current_transactions, node['right'], node['feat'], False)
                else:
                    negative_vector = np.where(feature_vector == 0)[0]
                    negative_vector = negative_vector.tolist()
                    current_transactions = set(transactions).intersection(negative_vector)
                    node['transactions'] = list(current_transactions)
                    if 'feat' in node.keys():
                        recurse(current_transactions, node['left'], node['feat'], True)
                        recurse(current_transactions, node['right'], node['feat'], False)

        node = self.tree_
        recurse(None, node, None, None)


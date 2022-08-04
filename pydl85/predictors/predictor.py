from sklearn.base import BaseEstimator
from sklearn.utils.validation import assert_all_finite, check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.exceptions import NotFittedError
from ..errors.errors import SearchFailedError, TreeNotFoundError
from distutils.util import strtobool
import json
import numpy as np
import uuid
from enum import Enum
from subprocess import check_call


def get_dot_body(treedict, parent=None, left=True):
    gstring = ""
    id = str(uuid.uuid4())
    id = id.replace('-', '_')

    if "feat" in treedict.keys():
        feat = treedict["feat"]
        if parent is None:
            gstring += "node_" + id + " [label=\"{{feat|" + str(feat) + "}}\"];\n"
            gstring += get_dot_body(treedict["left"], id)
            gstring += get_dot_body(treedict["right"], id, False)
        else:
            gstring += "node_" + id + " [label=\"{{feat|" + str(feat) + "}}\"];\n"
            gstring += "node_" + parent + " -> node_" + id + " [label=" + str(int(left)) + "];\n"
            gstring += get_dot_body(treedict["left"], id)
            gstring += get_dot_body(treedict["right"], id, False)
    else:
        val = str(int(treedict["value"])) if treedict["value"] - int(treedict["value"]) == 0 else str(
            round(treedict["value"], 3))
        err = str(int(treedict["error"])) if treedict["error"] - int(treedict["error"]) == 0 else str(
            round(treedict["error"], 2))
        # maxi = max(len(val), len(err))
        # val = val if len(val) == maxi else val + (" " * (maxi - len(val)))
        # err = err if len(err) == maxi else err + (" " * (maxi - len(err)))
        gstring += "leaf_" + id + " [label=\"{{class|" + val + "}|{error|" + err + "}}\"];\n"
        gstring += "node_" + parent + " -> leaf_" + id + " [label=" + str(int(left)) + "];\n"
    return gstring


class Cache_Type(Enum):
    Cache_TrieItemset = 1
    Cache_HashItemset = 2
    Cache_HashCover = 3


class Wipe_Type(Enum):
    All = 1
    Subnodes = 2
    Reuses = 3


class DL85Predictor(BaseEstimator):
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
    desc : function, default=None
        A parameter used to indicate heuristic function used to sort the items in descending order
    asc : function, default=None
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
            fast_error_function=None,
            predict_error_function=None,
            iterative=False,
            max_error=0,
            stop_after_better=False,
            time_limit=0,
            verbose=False,
            desc=False,
            asc=False,
            repeat_sort=False,
            leaf_value_function=None,
            nps=False,
            quiet=True,
            print_output=False,
            cache_type=Cache_Type.Cache_TrieItemset,
            maxcachesize=0,
            wipe_type=Wipe_Type.Subnodes,
            wipe_factor=0.5,
            use_cache=True,
            depth_two_special_algo=False,
            use_ub=True,
            similar_lb=False,
            dynamic_branch=True,
            similar_for_branching=False):

        self.max_depth = max_depth
        self.min_sup = min_sup
        # self.max_estimators = max_estimators
        self.sample_weight = []
        self.error_function = error_function
        self.fast_error_function = fast_error_function
        # self.example_weight_function = example_weight_function
        # self.predict_error_function = predict_error_function
        self.iterative = iterative
        self.max_error = max_error
        self.stop_after_better = stop_after_better
        self.time_limit = time_limit
        self.verbose = verbose
        self.desc = desc
        self.asc = asc
        self.repeat_sort = repeat_sort
        self.leaf_value_function = leaf_value_function
        self.nps = nps
        self.quiet = quiet
        self.print_output = print_output
        self.cache_type = cache_type
        self.maxcachesize = maxcachesize
        self.wipe_type = wipe_type
        self.wipe_factor = wipe_factor
        self.use_cache = use_cache
        self.depth_two_special_algo = depth_two_special_algo
        self.use_ub = use_ub
        self.similar_lb = similar_lb
        self.dynamic_branch = dynamic_branch
        self.similar_for_branching = similar_for_branching

        self.tree_ = None
        self.base_tree_ = None
        self.size_ = -1
        self.depth_ = -1
        self.error_ = -1
        self.accuracy_ = -1
        self.lattice_size_ = -1
        self.runtime_ = -1
        self.timeout_ = False
        self.classes_ = []
        self.is_fitted_ = False

    def _more_tags(self):
        return {'X_types': 'categorical',
                'allow_nan': False}

    def fit(self, X, y=None):
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

        target_is_need = True if y is not None else False
        opt_func = self.error_function
        opt_fast_func = self.fast_error_function
        opt_pred_func = self.error_function
        predict = True

        if target_is_need:  # target-needed tasks (eg: classification, regression, etc.)
            # Check that X and y have correct shape and raise ValueError if not
            X, y = check_X_y(X, y, dtype='int32')
            if self.leaf_value_function is None:
                opt_pred_func = None
                predict = False
            else:
                opt_func = None
                opt_fast_func = None
            # if opt_func is None and opt_pred_func is None:
            #     print("No optimization criterion defined. Misclassification error is used by default.")
        else:  # target-less tasks (clustering, etc.)
            # Check that X has correct shape and raise ValueError if not
            assert_all_finite(X)
            X = check_array(X, dtype='int32')
            if self.leaf_value_function is None:
                opt_pred_func = None
                predict = False
            else:
                opt_func = None
                opt_fast_func = None

        # sys.path.insert(0, "../../")
        import dl85Optimizer
        # print(opt_func)
        solution_str = dl85Optimizer.solve(data=X,
                                           target=y,
                                           tec_func_=opt_func,
                                           sec_func_=opt_fast_func,
                                           te_func_=opt_pred_func,
                                           # exw_func_=self.example_weight_function,
                                           # pred_func_=self.predict_error_function,
                                           max_depth=self.max_depth,
                                           min_sup=self.min_sup,
                                           # max_estimators=self.max_estimators,
                                           example_weights=self.sample_weight,
                                           max_error=self.max_error,
                                           stop_after_better=self.stop_after_better,
                                           # iterative=self.iterative,
                                           time_limit=self.time_limit,
                                           verb=self.verbose,
                                           desc=self.desc,
                                           asc=self.asc,
                                           repeat_sort=self.repeat_sort,
                                           # bin_save=False,
                                           # nps=self.nps,
                                           predictor=predict,
                                           cachetype=dl85Optimizer.CacheType.CacheTrieItemset if self.cache_type == Cache_Type.Cache_TrieItemset else dl85Optimizer.CacheType.CacheHashItemset if self.cache_type == Cache_Type.Cache_HashItemset else dl85Optimizer.CacheType.CacheHashCover,
                                           cachesize=self.maxcachesize,
                                           wipetype=dl85Optimizer.WipeType.All if self.wipe_type == Wipe_Type.All else dl85Optimizer.WipeType.Subnodes if self.wipe_type == Wipe_Type.Subnodes else dl85Optimizer.WipeType.Reuses,
                                           wipefactor=self.wipe_factor,
                                           withcache=self.use_cache,
                                           usespecial=self.depth_two_special_algo,
                                           useub=self.use_ub,
                                           similar_lb=self.similar_lb,
                                           dyn_branch=self.dynamic_branch,
                                           similar_for_branching=self.similar_for_branching)

        solution = solution_str.rstrip("\n").splitlines()

        self.tree_ = json.loads(solution[-8].split('Tree: ')[1]) if "No such tree" not in solution[-8] else None
        self.size_ = int(solution[-7].split(" ")[1])
        self.depth_ = int(solution[-6].split(" ")[1])
        self.error_ = float(solution[-5].split(" ")[1])
        self.accuracy_ = float(solution[-4].split(" ")[1])
        self.lattice_size_ = int(solution[-3].split(" ")[1])
        self.runtime_ = float(solution[-2].split(" ")[1])
        self.timeout_ = bool(strtobool(solution[-1].split(" ")[1]))

        if target_is_need:  # problem with target
            self.classes_ = unique_labels(y)  # Store the classes seen during fit

        if self.tree_ is None:  # No solution
            if not self.timeout_:
                print("DL8.5 fitting: Solution not found. However, a solution exists with error greater than or equal to the max error (exlusive upper bound) you specify. Please increase your bound if you want to find it.")
            else:
                print("DL8.5 fitting: Timeout reached without solution. Please increase the time limit and/or the max error (exlusive upper bound)")
        else:
            if not self.quiet:
                if not self.timeout_:
                    print("DL8.5 fitting: Solution found")
                else:
                    print("DL8.5 fitting: Timeout reached. The solution found may not be optimal")

        if hasattr(self, 'tree_') and self.tree_ is not None:
            # add transactions to nodes of the tree
            self.add_transactions_and_proba(X, y)

            # label the leafs when a labelling function is provided
            if self.leaf_value_function is not None:
                def search(node):
                    if self.is_leaf_node(node) is not True:
                        search(node['left'])
                        search(node['right'])
                    else:
                        node['value'] = self.leaf_value_function(node['transactions'])

                node = self.tree_
                search(node)

            self.remove_transactions()

        self.base_tree_ = self.get_tree_without_transactions_and_probas() if self.tree_ is not None else None

        if self.print_output:
            print(solution_str)

        # Return the classifier
        self.is_fitted_ = True
        return self

    def predict(self, X):
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

        # Check is fit is called
        # check_is_fitted(self, attributes='tree_') # use of attributes is deprecated. alternative solution is below

        # if hasattr(self, 'sol_size') is False:  # fit method has not been called
        if self.is_fitted_ is False:  # fit method has not been called
            raise NotFittedError("Call fit method first" % {'name': type(self).__name__})

        if self.tree_ is None:
            raise TreeNotFoundError("predict(): ", "Tree not found during training by DL8.5 - "
                                                   "Check fitting message for more info.")

        if hasattr(self, 'tree_') is False:  # normally this case is not possible.
            raise SearchFailedError("PredictionError: ", "DL8.5 training has failed. Please contact the developers "
                                                         "if the problem is in the scope supported by the tool.")

        # Input validation
        X = check_array(X)

        pred = []

        for i in range(X.shape[0]):
            pred.append(self.pred_value_on_dict(X[i, :]))

        return pred

    def pred_value_on_dict(self, instance, tree=None):
        node = tree if tree is not None else self.tree_
        while self.is_leaf_node(node) is not True:
            if instance[node['feat']] == 1:
                node = node['left']
            else:
                node = node['right']
        return node['value']

    def predict_proba(self, X):
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

        # Check is fit is called
        # check_is_fitted(self, attributes='tree_') # use of attributes is deprecated. alternative solution is below

        # if hasattr(self, 'sol_size') is False:  # fit method has not been called
        if self.is_fitted_ is False:  # fit method has not been called
            raise NotFittedError("Call fit method first" % {'name': type(self).__name__})

        if self.tree_ is None:
            raise TreeNotFoundError("predict(): ", "Tree not found during training by DL8.5 - "
                                                   "Check fitting message for more info.")

        if hasattr(self, 'tree_') is False:  # normally this case is not possible.
            raise SearchFailedError("PredictionError: ", "DL8.5 training has failed. Please contact the developers "
                                                         "if the problem is in the scope supported by the tool.")

        # Input validation
        X = check_array(X)

        pred = []

        for i in range(X.shape[0]):
            pred.append(self.pred_proba_on_dict(X[i, :]))

        return np.array(pred)

    def pred_proba_on_dict(self, instance, tree=None):
        node = tree if tree is not None else self.tree_
        while self.is_leaf_node(node) is not True:
            if instance[node['feat']] == 1:
                node = node['left']
            else:
                node = node['right']
        return node['proba']

    def get_nodes_count(self):
        if self.is_fitted_ is False:  # fit method has not been called
            raise NotFittedError("Call fit method first" % {'name': type(self).__name__})

        if self.tree_ is None:
            raise TreeNotFoundError("predict(): ", "Tree not found during training by DL8.5 - "
                                                   "Check fitting message for more info.")

        if hasattr(self, 'tree_') is False:  # normally this case is not possible.
            raise SearchFailedError("PredictionError: ", "DL8.5 training has failed. Please contact the developers "
                                                         "if the problem is in the scope supported by the tool.")

        tree_str = json.dumps(self.tree_)
        return tree_str.count('feat') + tree_str.count('value')

    @staticmethod
    def is_leaf_node(node):
        names = [x[0] for x in node.items()]
        return 'error' in names

    # explore the decision tree found and add transactions and class probabilities to leaf nodes.
    def add_transactions_and_proba(self, X, y=None):
        def recurse(transactions, node, feature, positive):
            if transactions is None:
                current_transactions = list(range(0, X.shape[0]))
                node['transactions'] = current_transactions
                if y is not None:
                    unique, counts = np.unique(y[node['transactions']], return_counts=True)
                    count_dict = dict(zip(unique, counts))
                    node['proba'] = []
                    for c in self.classes_:
                        if c in count_dict:
                            node['proba'].append(count_dict[c] / sum(counts))
                        else:
                            node['proba'].append(0)
                else:
                    node['proba'] = None
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
                    if y is not None:
                        unique, counts = np.unique(y[node['transactions']], return_counts=True)
                        count_dict = dict(zip(unique, counts))
                        node['proba'] = []
                        for c in self.classes_:
                            if c in count_dict:
                                node['proba'].append(count_dict[c] / sum(counts))
                            else:
                                node['proba'].append(0)
                    else:
                        node['proba'] = None
                    if 'feat' in node.keys():
                        recurse(current_transactions, node['left'], node['feat'], True)
                        recurse(current_transactions, node['right'], node['feat'], False)
                else:
                    negative_vector = np.where(feature_vector == 0)[0]
                    negative_vector = negative_vector.tolist()
                    current_transactions = set(transactions).intersection(negative_vector)
                    node['transactions'] = list(current_transactions)
                    if y is not None:
                        unique, counts = np.unique(y[node['transactions']], return_counts=True)
                        count_dict = dict(zip(unique, counts))
                        node['proba'] = []
                        for c in self.classes_:
                            if c in count_dict:
                                node['proba'].append(count_dict[c] / sum(counts))
                            else:
                                node['proba'].append(0)
                    else:
                        node['proba'] = None
                    if 'feat' in node.keys():
                        recurse(current_transactions, node['left'], node['feat'], True)
                        recurse(current_transactions, node['right'], node['feat'], False)

        root_node = self.tree_
        recurse(None, root_node, None, None)

    def get_tree_without_transactions(self):

        def recurse(node):
            if 'transactions' in node and ('feat' in node.keys() or 'value' in node.keys()):
                del node['transactions']
                if 'left' in node.keys():
                    recurse(node['left'])
                    recurse(node['right'])

        tree = dict(self.tree_)
        recurse(tree)
        return tree

    def remove_transactions(self):
        def recurse(node):
            if 'transactions' in node and ('feat' in node.keys() or 'value' in node.keys()):
                del node['transactions']
                if 'left' in node.keys():
                    recurse(node['left'])
                    recurse(node['right'])

        recurse(self.tree_)

    def get_tree_without_probas(self):

        def recurse(node):
            if 'proba' in node and ('feat' in node.keys() or 'value' in node.keys()):
                del node['proba']
                if 'left' in node.keys():
                    recurse(node['left'])
                    recurse(node['right'])

        tree = dict(self.tree_)
        recurse(tree)
        return tree

    def remove_probas(self):
        def recurse(node):
            if 'proba' in node and ('feat' in node.keys() or 'value' in node.keys()):
                del node['proba']
                if 'left' in node.keys():
                    recurse(node['left'])
                    recurse(node['right'])

        recurse(self.tree_)

    def get_tree_without_transactions_and_probas(self):

        def recurse(node):
            if 'feat' in node.keys() or 'value' in node.keys():
                if 'proba' in node:
                    del node['proba']
                if 'transactions' in node:
                    del node['transactions']
                if 'left' in node.keys():
                    recurse(node['left'])
                    recurse(node['right'])

        tree = dict(self.tree_)
        recurse(tree)
        return tree

    def export_graphviz(self):
        if self.is_fitted_ is False:  # fit method has not been called
            raise NotFittedError("Call fit method first" % {'name': type(self).__name__})

        if self.tree_ is None:
            raise TreeNotFoundError("predict(): ", "Tree not found during training by DL8.5 - "
                                                   "Check fitting message for more info.")

        if hasattr(self, 'tree_') is False:  # normally this case is not possible.
            raise SearchFailedError("PredictionError: ", "DL8.5 training has failed. Please contact the developers "
                                                         "if the problem is in the scope supported by the tool.")

        # initialize the header
        graph_string = "digraph Tree { \n" \
                       "graph [ranksep=0]; \n" \
                       "node [shape=record]; \n"

        # build the body
        graph_string += get_dot_body(self.tree_)

        # end by the footer
        graph_string += "}"

        return graph_string

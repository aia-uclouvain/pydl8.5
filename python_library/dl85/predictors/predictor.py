from sklearn.base import BaseEstimator
from sklearn.utils.validation import assert_all_finite, check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.exceptions import NotFittedError
from ..errors.errors import SearchFailedError, TreeNotFoundError
from distutils.util import strtobool
import json
import numpy as np
import uuid
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
        val = str(int(treedict["value"])) if treedict["value"] - int(treedict["value"]) == 0 else str(round(treedict["value"], 3))
        err = str(int(treedict["error"])) if treedict["error"] - int(treedict["error"]) == 0 else str(round(treedict["error"], 2))
        # maxi = max(len(val), len(err))
        # val = val if len(val) == maxi else val + (" " * (maxi - len(val)))
        # err = err if len(err) == maxi else err + (" " * (maxi - len(err)))
        gstring += "leaf_" + id + " [label=\"{{class|" + val + "}|{error|" + err + "}}\"];\n"
        gstring += "node_" + parent + " -> leaf_" + id + " [label=" + str(int(left)) + "];\n"
    return gstring


class DL85Predictor(BaseEstimator):
    """ An optimal binary decision tree classifier.

    Parameters
    ----------
    max_depth : int, default=1
        Maximum depth of the tree to be found
    min_sup : int, default=1
        Minimum number of examples per leaf
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
            print_output=False):
        self.max_depth = max_depth
        self.min_sup = min_sup
        self.sample_weight = []
        self.error_function = error_function
        self.fast_error_function = fast_error_function
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

        self.tree_ = None
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
        solution = dl85Optimizer.solve(data=X,
                                       target=y,
                                       tec_func_=opt_func,
                                       sec_func_=opt_fast_func,
                                       te_func_=opt_pred_func,
                                       max_depth=self.max_depth,
                                       min_sup=self.min_sup,
                                       example_weights=self.sample_weight,
                                       max_error=self.max_error,
                                       stop_after_better=self.stop_after_better,
                                       time_limit=self.time_limit,
                                       verb=self.verbose,
                                       desc=self.desc,
                                       asc=self.asc,
                                       repeat_sort=self.repeat_sort)

        # if self.print_output:
        #     print(solution)

        solution = solution.rstrip("\n").splitlines()
        self.sol_size = len(solution)

        if self.sol_size == 9:  # solution found
            self.tree_ = json.loads(solution[1].split('Tree: ')[1])
            self.size_ = int(solution[2].split(" ")[1])
            self.depth_ = int(solution[3].split(" ")[1])
            self.error_ = float(solution[4].split(" ")[1])
            self.lattice_size_ = int(solution[6].split(" ")[1])
            self.runtime_ = float(solution[7].split(" ")[1])
            self.timeout_ = bool(strtobool(solution[8].split(" ")[1]))
            if self.size_ >= 3 or self.max_error <= 0:
                self.accuracy_ = float(solution[5].split(" ")[1])

            # if sol_size == 8:  # without timeout
            if self.size_ < 3 and self.max_error > 0:  # return just a leaf as fake solution
                if not self.timeout_:
                    print("DL8.5 fitting: Solution not found. However, a solution exists with error equal to the "
                      "max error you specify as unreachable. Please increase your bound if you want to reach it.")
                else:
                    print("DL8.5 fitting: Timeout reached without solution. However, a solution exists with "
                          "error equal to the max error you specify as unreachable. Please increase "
                          "your bound if you want to reach it.")
            else:
                if not self.quiet:
                    if not self.timeout_:
                        print("DL8.5 fitting: Solution found")
                    else:
                        print("DL8.5 fitting: Timeout reached but solution found")

            if target_is_need:  # problem with target
                # Store the classes seen during fit
                self.classes_ = unique_labels(y)

        elif self.sol_size == 5:  # solution not found
            self.lattice_size_ = int(solution[2].split(" ")[1])
            self.runtime_ = float(solution[3].split(" ")[1])
            self.timeout_ = bool(strtobool(solution[4].split(" ")[1]))
            if not self.timeout_:
                print("DL8.5 fitting: Solution not found")
            else:  # timeout reached
                print("DL8.5 fitting: Timeout reached and solution not found")

        if hasattr(self, 'tree_') and self.tree_ is not None:
            # add transactions to nodes of the tree
            self.add_transactions_and_proba(X, y)

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

        if self.print_output:
            print(solution[0])
            print("Tree:", self.tree_)
            print("Size:", str(self.size_))
            print("Depth:", str(self.depth_))
            print("Error:", str(self.error_))
            # print("Accuracy:", str(self.accuracy_))
            print("LatticeSize:", str(self.lattice_size_))
            print("Runtime:", str(self.runtime_))
            print("Timeout:", str(self.timeout_))

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

    def add_transactions_and_proba(self, X, y=None):  # explore the decision tree found and add transactions to leaf nodes.
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

    def tree_without_transactions(self):

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

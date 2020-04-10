from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.exceptions import NotFittedError
from ..errors.errors import SearchFailedError, TreeNotFoundError
import json
import numpy as np
from ..supervised.classifiers import DL85Classifier


class DL85FeatureSelector(BaseEstimator, TransformerMixin):
    """ An optimal binary decision tree classifier.

    Parameters
    ----------
    max_features : int, default=1
        Maximum number of the features to be output

    Attributes
    ----------
    features_ : array-like
        Outputted tree in serialized form; remains empty as long as no model is learned.
    """

    def __init__(
            self,
            max_features=1,
            n_try=500,
            algo='d2'):
        self.max_features = max_features
        self.n_try = n_try
        self.algo = algo

    def _more_tags(self):
        return {'X_types': 'categorical',
                'allow_nan': False}

    def fit(self, X, y):
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

        def get_features_d2(tree):
            def recurse(node, in_feats):
                if 'feat' in node.keys():
                    in_feats.append(node['feat'])
                    if 'left' in node.keys():
                        recurse(node['left'], in_feats)
                        recurse(node['right'], in_feats)

            tree = dict(tree)
            feats = list()
            recurse(tree, feats)
            return feats

        def get_leaf_transactions_d2(tree):
            def recurse(node, in_transactions):
                if 'error' in node.keys():
                    in_transactions.append(node['transactions'])
                elif 'left' in node.keys():
                    recurse(node['left'], in_transactions)
                    recurse(node['right'], in_transactions)

            tree = dict(tree)
            transactions = []
            recurse(tree, transactions)
            return transactions

        def add_level_transactions_d2(tree, X, tids):  # explore the decision tree found and add transactions to leaf nodes.
            def recurse(transactions, node, feature, positive):
                if transactions is None:
                    current_transactions = list(range(0, X.shape[0]))
                    if 'error' in node.keys():
                        node['transactions'] = list(set(current_transactions).intersection(set(tids)))  #intersect with transactions covered by the leaf become root
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
                        current_transactions = current_transactions.intersection(set(tids))  #intersect with transactions covered by the leaf become root
                        if 'error' in node.keys():
                            node['transactions'] = list(current_transactions)
                        if 'feat' in node.keys():
                            recurse(current_transactions, node['left'], node['feat'], True)
                            recurse(current_transactions, node['right'], node['feat'], False)
                    else:
                        negative_vector = np.where(feature_vector == 0)[0]
                        negative_vector = negative_vector.tolist()
                        current_transactions = set(transactions).intersection(negative_vector)
                        current_transactions = current_transactions.intersection(set(tids))  #intersect with transactions covered by the leaf become root
                        if 'error' in node.keys():
                            node['transactions'] = list(current_transactions)
                        if 'feat' in node.keys():
                            recurse(current_transactions, node['left'], node['feat'], True)
                            recurse(current_transactions, node['right'], node['feat'], False)
            root_node = tree
            recurse(None, root_node, None, None)

        def get_features_d1(tree):
            if 'feat' in tree.keys():
                return [tree['feat']]
            else:
                return []

        def get_leaf_transactions_d1(tree):
            if 'left' in tree.keys():
                return [tree['left']['transactions'], tree['right']['transactions']]
            else:
                return []

        def add_level_transactions_d1(tree, X, tids):
            if 'feat' in tree.keys():
                feature_vector = X[:, tree['feat']]
                feature_vector = feature_vector.astype('int32')
                positive_trans = np.where(feature_vector == 1)[0]
                positive_trans = list(set(positive_trans.tolist()).intersection(set(tids)))
                negative_trans = np.where(feature_vector == 0)[0]
                negative_trans = list(set(negative_trans.tolist()).intersection(set(tids)))
                tree['left']['transactions'] = positive_trans
                tree['right']['transactions'] = negative_trans
            # else:
            #     print(tree)

        # Check that X and y have correct shape and raise ValueError if not
        X, y = check_X_y(X, y, dtype='int32')

        relevant_features = []
        tids_fifo_list = [list(range(0, X.shape[0]))]

        # sys.path.insert(0, "../../")
        import dl85Optimizer
        tries = 0
        while len(tids_fifo_list) > 0 and len(set(relevant_features)) < self.max_features and tries < self.n_try:
            tries += 1
            tids = tids_fifo_list.pop(0)
            solution = dl85Optimizer.solve(data=X[tids, ],
                                           target=y.take(tids),
                                           func=None,
                                           fast_func=None,
                                           predictor_func=None,
                                           max_depth=2,
                                           min_sup=1,
                                           max_error=0,
                                           stop_after_better=False,
                                           alph=0,
                                           iterative=False,
                                           time_limit=600,
                                           verb=False,
                                           desc=True,
                                           asc=False,
                                           repeat_sort=False,
                                           bin_save=False,
                                           nps=False,
                                           predictor=False)
            solution = solution.splitlines()
            sol_size = len(solution)

            if sol_size == 8 or sol_size == 9:  # solution found
                tree = json.loads(solution[1].split('Tree: ')[1])
                if self.algo == 'd1':
                    add_level_transactions_d1(tree, X, tids)
                    relevant_features += get_features_d1(tree)
                    # print("best features here", list(set(relevant_features)), len(list(set(relevant_features))), "time = ", float(solution[7].split(" ")[1]))
                    tids_fifo_list += get_leaf_transactions_d1(tree)
                elif self.algo == 'd2':
                    add_level_transactions_d2(tree, X, tids)
                    relevant_features += get_features_d2(tree)
                    # print("best features here", list(set(relevant_features)), len(list(set(relevant_features))), "time = ", float(solution[7].split(" ")[1]))
                    tids_fifo_list += get_leaf_transactions_d2(tree)

            # print(y.take(tids).tolist())
            # print(np.unique(y.take(tids).tolist(), return_counts=True)[1])
            # clf = DL85Classifier(max_depth=2, time_limit=600, desc=True, print_output=False)
            # clf.fit(X[tids, ], y.take(tids))
            # print("feat = ", get_features(clf.tree_), "\n")
            # relevant_features += get_features(clf.tree_)
            # # print("best features here", list(set(relevant_features)), len(list(set(relevant_features))), "time = ", clf.runtime_)
            # tids_fifo_list += get_leaf_transactions(clf.tree_)

        # print("len tids list = ", len(tids_fifo_list), "len relevant feat =", len(set(relevant_features)), "ntries = ", tries)
        feats, supports = np.unique(relevant_features, return_counts=True)
        self.features_ = [x for _, x in sorted(zip(supports, feats), key=lambda pair: pair[0])]
        if self.max_features < len(feats):
            self.features_ = self.features_[:self.max_features]

        # Return the classifier
        return self

    def transform(self, X):
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

        if hasattr(self, 'features_') is False:  # actually this case is not possible.
            raise NotFittedError("Call fit method first" % {'name': type(self).__name__})

        # Input validation
        X = check_array(X)

        return X[:, self.features_]

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import assert_all_finite, check_array, check_is_fitted
from sklearn.neighbors import DistanceMetric
from ..errors.errors import SearchFailedError, TreeNotFoundError
from sklearn.exceptions import NotFittedError
from ..predictors.predictor import DL85Predictor
import numpy as np


class DL85Cluster(DL85Predictor, BaseEstimator, ClusterMixin):
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
            max_depth=1,
            min_sup=1,
            error_function=None,
            iterative=False,
            max_error=0,
            stop_after_better=False,
            time_limit=0,
            verbose=False,
            # desc_sort_function=None,
            # asc_sort_function=None,
            repeat_sort=False,
            leaf_value_function=None,
            nps=False,
            print_output=False):

        DL85Predictor.__init__(self,
                               max_depth,
                               min_sup,
                               error_function,
                               iterative,
                               max_error,
                               stop_after_better,
                               time_limit,
                               verbose,
                               None,
                               None,
                               repeat_sort,
                               leaf_value_function,
                               nps,
                               print_output)

    @staticmethod
    def default_error(tids, X):
        dist = DistanceMetric.get_metric('euclidean')
        X_subset = np.asarray([X[index, :] for index in list(tids)], dtype='int32')
        centroid = np.mean(X_subset, axis=0).reshape(1, X_subset.shape[1])
        distances = [dist.pairwise(instance.reshape(1, X_subset.shape[1]), centroid)[0, 0] for instance in X_subset]
        return sum(distances)

    @staticmethod
    def default_leaf_value(tids, X):
        return np.mean(X.take(list(tids)))

    def _more_tags(self):
        return {'X_types': 'categorical',
                'allow_nan': False}

    def fit(self, X, X_error=None):
        """Implements the standard fitting function for a DL8.5 classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples. If X_error is provided, it represents explanation input
        X_error : array-like, shape (n_samples, n_features_1)
            The training input used to calculate error. If it is not provided X is used to calculate error

        Returns
        -------
        self : object
            Returns self.
        """

        # Check that X has correct shape and raise ValueError if not
        assert_all_finite(X)
        X = check_array(X, dtype='int32')

        # Check that X_error has correct shape and raise ValueError if not
        if X_error is not None:
            assert_all_finite(X_error)
            X_error = check_array(X_error, dtype='int32')

        if self.error_function is None:
            if X_error is None:
                self.error_function = lambda tids: self.default_error(tids, X)
            else:
                if X_error.shape[0] == X.shape[0]:
                    self.error_function = lambda tids: self.default_error(tids, X_error)
                else:
                    raise ValueError("X_error does not have the same number of rows as X")

        if self.leaf_value_function is None:
            if X_error is None:
                self.leaf_value_function = lambda tids: self.default_leaf_value(tids, X)
            else:
                if X_error.shape[0] == X.shape[0]:
                    self.leaf_value_function = lambda tids: self.default_leaf_value(tids, X_error)
                else:
                    raise ValueError("X_error does not have the same number of rows as X")

        # call fit method of the predictor
        self.predictor_fit(X)
        #print(self.tree_)

        # Return the classifier
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

        # Check is fit had been called
        # check_is_fitted(self, attributes='tree_') # use of attributes is deprecated. alternative solution is below

        if hasattr(self, 'sol_size') is False:  # actually this case is not possible.
            raise NotFittedError("Call fit method first" % {'name': type(self).__name__})

        if self.tree_ is None:
            raise TreeNotFoundError("predict(): ", "Tree not found during training by DL8.5")

        if hasattr(self, 'tree_') is False:  # normally this case is not possible.
            raise SearchFailedError("PredictionError: ", "DL8.5 training has failed. Please contact the developers if the problem is in the scope claimed by the tool.")

        # Input validation
        X = check_array(X)

        self.y_ = self.predictor_predict(X)

        return self.y_



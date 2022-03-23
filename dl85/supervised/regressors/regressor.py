from sklearn.base import RegressorMixin
from ...predictors.predictor import DL85Predictor
import numpy as np
from math import floor, ceil


class DL85Regressor(DL85Predictor, RegressorMixin):
    """An optimal binary decision tree regressor.
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
    desc : bool, default=False
        A parameter used to indicate if the sorting of the items is done in descending order of information gain
    asc : bool, default=False
        A parameter used to indicate if the sorting of the items is done in ascending order of information gain
    repeat_sort : bool, default=False
        A parameter used to indicate whether the sorting of items is done at each level of the lattice or only before the search
    print_output : bool, default=False
        A parameter used to indicate if the search output will be printed or not
    backup_error : str, default = "mse"
        Error to optimize if no user error function is provided. Can be one of {"mse", "quantile"}
    quantile_value: float, default = 0.5 
        Quantile value. Only used when backup_error is "quantile"

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
        max_error=0,
        stop_after_better=False,
        time_limit=0,
        verbose=False,
        desc=False,
        asc=False,
        repeat_sort=False,
        leaf_value_function=None,
        print_output=False,
        backup_error = "mse",
        quantile_value = 0.5,
    ):

        if backup_error not in ["mse", "quantile"]:
            raise ValueError(f"{backup_error} is not a valid error function string.")

        DL85Predictor.__init__(
            self,
            max_depth=max_depth,
            min_sup=min_sup,
            error_function=error_function,
            fast_error_function=None,
            max_error=max_error,
            stop_after_better=stop_after_better,
            time_limit=time_limit,
            verbose=verbose,
            desc=desc,
            asc=asc,
            repeat_sort=repeat_sort,
            leaf_value_function=leaf_value_function,
            print_output=print_output,
            backup_error=backup_error,
            quantile_value=quantile_value,
        )

        self.to_redefine = self.leaf_value_function is None

    
    @staticmethod
    def mean_leaf_value(tids, y):
        return np.mean(y[list(tids)], axis=0)

    @staticmethod
    def quantile_leaf_value(tids, y, q):
        return np.quantile(y[list(tids)], q)

    def fit(self, X, y):
        """Implements the standard fitting function for a DL8.5 regressor.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples, n_predictions)
            The training output samples.

        Returns
        -------
        self : object
            Returns self.
        """

        if self.backup_error == "quantile":
            idx = np.argsort(y)
        else:
            idx = np.arange(len(y))

        X = X[idx]
        y = y[idx]

        if self.to_redefine:
            if self.backup_error == "mse":
                self.leaf_value_function = lambda tids: self.mean_leaf_value(tids, y)
            elif self.backup_error == "quantile":
                self.leaf_value_function = lambda tids: self.quantile_leaf_value(tids, y, self.quantile_value)

        

        # call fit method of the predictor
        DL85Predictor.fit(self, X, y)

        # Return the regressor
        return self

    def predict(self, X):
        """Implements the standard predict function for a DL8.5 regressor.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            The predicted value for each sample is the mean of the closest samples seen during fit.
        """

        return DL85Predictor.predict(self, X)

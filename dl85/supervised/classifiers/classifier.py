from sklearn.base import ClassifierMixin
from ...predictors.predictor import DL85Predictor
import json


class DL85Classifier(DL85Predictor, ClassifierMixin):
    """
    An optimal binary decision tree classifier.

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
            quiet=True,
            print_output=False):

        DL85Predictor.__init__(self,
                               max_depth=max_depth,
                               min_sup=min_sup,
                               error_function=error_function,
                               fast_error_function=fast_error_function,
                               max_error=max_error,
                               stop_after_better=stop_after_better,
                               time_limit=time_limit,
                               verbose=verbose,
                               desc=desc,
                               asc=asc,
                               repeat_sort=repeat_sort,
                               leaf_value_function=None,
                               quiet=quiet,
                               print_output=print_output)

    def fit(self, X, y=None, sample_weight=None):
        if sample_weight is None:
            return DL85Predictor.fit(self, X, y)
        else:
            self.sample_weight = sample_weight
            return DL85Predictor.fit(self, X, y)

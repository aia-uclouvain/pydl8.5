from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.exceptions import NotFittedError
from ...errors.errors import SearchFailedError, TreeNotFoundError
import json


class DL85Classifier(BaseEstimator, ClassifierMixin):
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
            fast_error_function=None,
            iterative=False,
            max_error=0,
            stop_after_better=False,
            time_limit=0,
            verbose=False,
            desc=False,
            asc=False,
            repeat_sort=False,
            nps=False,
            print_output=False):
        self.error_function = error_function
        self.fast_error_function = fast_error_function
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
        self.nps = nps
        self.print_output = print_output

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

        # Check that X and y have correct shape and raise ValueError if not
        X, y = check_X_y(X, y, dtype='int32')

        # sys.path.insert(0, "../../")
        import dl85Optimizer
        solution = dl85Optimizer.solve(data=X,
                                       target=y,
                                       func=self.error_function,
                                       fast_func=self.fast_error_function,
                                       predictor_func=None,
                                       max_depth=self.max_depth,
                                       min_sup=self.min_sup,
                                       max_error=self.max_error,
                                       stop_after_better=self.stop_after_better,
                                       iterative=self.iterative,
                                       time_limit=self.time_limit,
                                       verb=self.verbose,
                                       desc=self.desc,
                                       asc=self.asc,
                                       repeat_sort=self.repeat_sort,
                                       bin_save=False,
                                       nps=self.nps)

        # print(solution)
        solution = solution.splitlines()
        self.sol_size = len(solution)

        # if self.sol_size_ == 1:
        #     raise ValueError(solution[0])

        if self.sol_size == 8 or self.sol_size == 9:  # solution found
            self.tree_ = json.loads(solution[1].split('Tree: ')[1])
            self.size_ = int(solution[2].split(" ")[1])
            self.depth_ = int(solution[3].split(" ")[1])
            self.error_ = float(solution[4].split(" ")[1])
            if self.size_ < 3 and self.max_error > 0:
                self.accuracy_ = -1
            else:
                self.accuracy_ = float(solution[5].split(" ")[1])

            if self.sol_size == 8:  # without timeout
                if self.size_ < 3 and self.max_error > 0:
                    print("DL8.5 fitting: Solution found but not kept since it has the same error as the max error you specify as unreachable.")
                    self.tree_ = None
                    self.size_ = -1
                    self.depth_ = -1
                    self.error_ = -1
                    self.accuracy_ = -1
                else:
                    print("DL8.5 fitting: Solution found")

                self.lattice_size_ = int(solution[6].split(" ")[1])
                self.runtime_ = float(solution[7].split(" ")[1])
                self.timeout_ = False
            else:  # timeout reached
                if self.size_ < 3 and self.max_error > 0:
                    print("DL8.5 fitting: Timeout reached and solution found is not kept since it has the same error as the max error you specify as unreachable.")
                    self.tree_ = None
                    self.size_ = -1
                    self.depth_ = -1
                    self.error_ = -1
                    self.accuracy_ = -1
                else:
                    print("DL8.5 fitting: Timeout reached but solution found")

                self.lattice_size_ = int(solution[7].split(" ")[1])
                self.runtime_ = float(solution[8].split(" ")[1])
                self.timeout_ = True

            # Store the classes seen during fit
            self.classes_ = unique_labels(y)

        elif self.sol_size == 4 or self.sol_size == 5:  # solution not found
            self.tree_ = None
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

        if self.print_output:
            print(solution[0])
            print("Tree:", self.tree_)
            print("Size:", str(self.size_))
            print("Depth:", str(self.depth_))
            print("Error:", str(self.error_))
            print("Accuracy:", str(self.accuracy_))
            print("LatticeSize:", str(self.lattice_size_))
            print("Runtime:", str(self.runtime_))
            print("Timeout:", str(self.timeout_))

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
            raise TreeNotFoundError("predict(): ", "Tree not found during training by DL8.5 — Check fitting message for more info.")

        if hasattr(self, 'tree_') is False:  # normally this case is not possible.
            raise SearchFailedError("PredictionError: ", "DL8.5 training has failed. Please contact the developers if the problem is in the scope claimed by the tool.")

        # Input validation
        X = check_array(X)

        self.y_ = []

        for i in range(X.shape[0]):
            self.y_.append(self.pred_on_dict(X[i, :]))

        return self.y_

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
        names = [x[0] for x in node.items()]
        return 'error' in names

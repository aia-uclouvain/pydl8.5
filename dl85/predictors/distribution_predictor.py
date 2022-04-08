from re import T
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import assert_all_finite, check_X_y, check_array, check_is_fitted
from distutils.util import strtobool
import json
from dl85.errors.errors import SearchFailedError, TreeNotFoundError
from dl85.predictors.predictor import DL85Predictor
import numpy as np


class DL85DistributionPredictor(DL85Predictor): 
    
    def __init__(self,
            max_depth=1,
            min_sup=1,
            max_errors=None,
            stop_after_better=None,
            time_limit=0,
            verbose=False,
            desc=False,
            asc=False,
            repeat_sort=False,
            leaf_value_function=None,
            quiet=True,
            print_output=False, 
            quantiles=[0.5]):
        
        self.max_depth = max_depth
        self.min_sup = min_sup
        self.sample_weight = []
        self.max_errors = max_errors
        self.stop_after_better = stop_after_better
        self.time_limit = time_limit
        self.verbose = verbose
        self.desc = desc
        self.asc = asc
        self.repeat_sort = repeat_sort
        self.leaf_value_function = leaf_value_function
        self.quiet = quiet
        self.print_output = print_output

        self.classes_ = []
        
        self.quantiles = sorted(quantiles)

        # if self.max_errors is None:
        #     self.max_errors = [0] * len(self.quantiles)
        
        # if self.stop_after_better is None:
        #     self.stop_after_better = [False] * len(self.quantiles)

        if self.max_errors is not None and len(self.max_errors) != len(self.quantiles):
            print('max_errors must be of same length as quantiles')
            return 

        if self.stop_after_better is not None and len(self.stop_after_better) != len(self.stop_after_better):
            print('stop_after_better must be of same length as quantiles')
            return

        self.trees_ = [{
            'tree': None,
            'size': -1,
            'depth': -1,
            'error': -1,
            'accuracy': -1,
            'lattice_size': -1,
            'runtime': -1,
            'timeout': False,
            'classes': [],
        } for q in self.quantiles]

        self.is_fitted_ = False

    @classmethod
    def quantile_leaf_value(cls, y, q):
        y_p = np.quantile(y, q)
        return y_p

    @classmethod 
    def quantile_error(cls, y, q):
        y_p = np.quantile(y, q)
        delta = y_p - y 
        delta[delta > 0] *= q 
        delta[delta < 0] *= (q - 1)
        return np.sum(delta)

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y, dtype='int32')

        idx = np.argsort(y)
        X = X[idx]
        y = y[idx]

        if self.leaf_value_function is None:
            self.leaf_value_function = lambda tids, q: self.quantile_leaf_value(y[list(tids)], q)

        # sys.path.insert(0, "../../")
        import dl85Optimizer
        # print(opt_func)
        solution = dl85Optimizer.solve(data=X,
                                       target=y,
                                       tec_func_=None,
                                       sec_func_=None,
                                       te_func_=None,
                                       backup_error="quantile",
                                       max_depth=self.max_depth,
                                       min_sup=self.min_sup,
                                       example_weights=self.sample_weight,
                                       max_error=self.max_errors,
                                       stop_after_better=self.stop_after_better,
                                       time_limit=self.time_limit,
                                       verb=self.verbose,
                                       desc=self.desc,
                                       asc=self.asc,
                                       repeat_sort=self.repeat_sort,
                                       quantiles=self.quantiles)

        solution = solution.splitlines()
        self.sol_size = len(solution)

        n_trees = len(self.quantiles)

        if self.sol_size == 2 + n_trees*9:  # solution found
            for i in range(n_trees):
                self.trees_[i]['tree'] = json.loads(solution[2+i*9].lstrip('Tree: '))
                self.trees_[i]['size'] = int(solution[2+i*9+1].split(" ")[1])
                self.trees_[i]['depth'] = int(solution[2+i*9+2].split(" ")[1])
                self.trees_[i]['error'] = float(solution[2+i*9+3].split(" ")[1])
                # self.trees_[i]['accuracy'] = float(solution[i*9+5].split(" ")[1])
                self.trees_[i]['lattice_size'] = int(solution[2+i*9+5].split(" ")[1])
                self.trees_[i]['runtime'] = float(solution[2+i*9+6].split(" ")[1])
                self.trees_[i]['timeout'] = bool(strtobool(solution[2+i*9+7].split(" ")[1]))

                if self.trees_[i]['size'] >= 3 or self.max_errors[i] <= 0:
                    self.trees_[i]['accuracy'] = float(solution[2+i*9+4].split(" ")[1])

            
                if self.trees_[i]['size'] < 3 and self.max_errors[i] > 0:  # return just a leaf as fake solution
                    if not self.timeout_:
                        print("DL8.5 fitting: Solution not found. However, a solution exists with error equal to the "
                        "max error you specify as unreachable. Please increase your bound if you want to reach it.")
                    else:
                        print("DL8.5 fitting: Timeout reached without solution. However, a solution exists with "
                            "error equal to the max error you specify as unreachable. Please increase "
                            "your bound if you want to reach it.")
                else:
                    if not self.quiet:
                        if not self.trees_[i]['timeout']:
                            print("DL8.5 fitting: Solution found")
                        else:
                            print("DL8.5 fitting: Timeout reached but solution found")

        elif self.sol_size == 5:  # solution not found
            self.lattice_size_ = int(solution[2].split(" ")[1])
            self.runtime_ = float(solution[3].split(" ")[1])
            self.timeout_ = bool(strtobool(solution[4].split(" ")[1]))
            if not self.timeout_:
                print("DL8.5 fitting: Solution not found")
            else:  # timeout reached
                print("DL8.5 fitting: Timeout reached and solution not found")

        if hasattr(self, 'trees_'):
            for i in range(n_trees):
                if self.trees_[i]['tree'] is not None:
                    # add transactions to nodes of the tree

                    leaf_fun = lambda tids: self.leaf_value_function(tids, self.quantiles[i])

                    self.add_transactions_and_proba(X, y, tree=self.trees_[i]['tree'])
                    
                    def search(node):
                        if self.is_leaf_node(node) is not True:
                            search(node['left'])
                            search(node['right'])
                        else:
                            node['value'] = self.leaf_value_function(node['transactions'], self.quantiles[i])
                            node['error'] = self.quantile_error(y[list(node['transactions'])], self.quantiles[i])
                    node = self.trees_[i]['tree']
                    search(node)

                    self.remove_transactions(self.trees_[i]['tree'])

        if self.print_output:
            print(solution[0])
            for i in range(n_trees):
                print(f"Tree for quantile {self.quantiles[i]}:")
                print("Tree:", self.trees_[i]['tree'])
                print("Size:", str(self.trees_[i]['size']))
                print("Depth:", str(self.trees_[i]['depth']))
                print("Error:", str(self.trees_[i]['error']))
                # print("Accuracy:", str(self.accuracy_))
                print("LatticeSize:", str(self.trees_[i]['lattice_size']))
                print("Runtime:", str(self.trees_[i]['runtime']))
                print("Timeout:", str(self.trees_[i]['timeout']))

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


        for i in range(len(self.quantiles)):
            if self.trees_[i]['tree'] is None:
                raise TreeNotFoundError("predict(): ", "Tree not found during training by DL8.5 - "
                                                    "Check fitting message for more info.")


        # Input validation
        X = check_array(X)

        pred = []

        for i in range(X.shape[0]):
            pred_per_quantile = []
            for t in range(len(self.quantiles)):
                pred_per_quantile.append(self.pred_value_on_dict(X[i, :], tree= self.trees_[t]['tree']))
            pred.append(pred_per_quantile)

        return pred

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

        for i in range(len(self.quantiles)):
            if self.trees_[i]['tree'] is None:
                raise TreeNotFoundError("predict(): ", "Tree not found during training by DL8.5 - "
                                                    "Check fitting message for more info.")


        # Input validation
        X = check_array(X)

        pred = []

        for i in range(X.shape[0]):
            pred_per_quantile = []
            for t in range(len(self.quantiles)):
                pred_per_quantile.append(self.pred_proba_on_dict(X[i, :], tree=self.trees_[t]['tree']))
            pred.append(pred_per_quantile)

        return np.array(pred)


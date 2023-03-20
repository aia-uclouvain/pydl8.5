.. title:: User guide : contents

.. _user_guide:

=============================
User guide: Using PyDL8.5
=============================

Optimal Decision Trees
----------------------

This library implements the DL8.5 algorithm for learning optimal binary decision trees
and provides a Python interface for this algorithm. 

An example of decision trees are classification trees. 
Classification trees are predictors in which the predictions correspond to class labels.
A tree is considered optimal on training data if no tree can be found that scores better on the given training data. 

An explicit aim of this library is to make it easy to specify and solve many different types of 
decision tree learning problems, including, but not limited to, classification trees.

Decision trees are traditionally learned using heuristic algorithms, such as CART and C4.5.
However, due to the heuristic nature of these algorithms, the trees learned using them can be larger than 
necessary; this may make the resulting trees less interpretable. Trees found by DL8.5 are optimal under constraints  that aim to make the resulting trees both interpretable and accurate.

Moreover, given that in DL8.5 it is not necessary to specify a heuristic, solving other learning problems 
than classification problems potentially becomes simpler.

Please note that trees that that are accurate on training data, may not always perform good on test data. To avoid 
problems with overfitting, it is recommended to run DL8.5 using carefully chosen constraints, as specified below. Moreover,
finding optimal decision tree is a hard search problem, and will require more computational resources. 
However, a recent study has shown that DL8.5's performance on this search problem is much better than that of
competing methods.

Classifiers
~~~~~~~~~~~

Decision tree classifiers are learned using the class ``DL85Classifier``. 
``DL85Classifier`` is a scikit-learn compatible classifier and can be used as a scikit-learn
classifier. It inherits from :class:`sklearn.base.BaseEstimator` and reimplements the methods ``fit`` and ``predict``.

The following code illustrates how to use DL8.5 in its most basic setting::

    import numpy as np
    from sklearn.model_selection import train_test_split
    from pydl85 import DL85Classifier

    dataset = np.genfromtext("anneal.txt", delimiter=" ")
    X = dataset[:, 1:]
    y = dataset[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

    clf = DL85Classifier(max_depth=3, min_sup=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

In this example, we use numpy to read a dataset and scikit-learn to split this dataset in training and test data.
Subsequently, the DL85Classifier is initialized, where the recommended parameters are specified; a decision tree is learned 
from the training data, which is applied on the test data.


* when the ``fit(X,y)`` method is executed, an optimal decision tree classifier is learned from ``X`` and ``y``, where ``X`` is a set of Boolean training samples and ``y`` is the  vector of target values; the resulting tree is stored in the ``DL85Classifier`` object. For more information on how the results of the learning algorithm are stored, please check the  `API documentation <api.html>`_.
* when the ``predict(X)`` method is executed, predictions will be computed for the Boolean test samples ``X`` using the tree
  learned during the execution of ``fit``. The output corresponds to a list of predicted classes for all the
  samples.

Parameters of the learning process need to be specified during the construction of the ``DL85Classifier`` object. 
The complete list of parameters can be found in the `API documentation <api.html>`_. We highly recommend to
specify the parameters of the following constraints, as their default values are not useful in many cases:

* ``max_depth``, which specifies the maximum depth of the tree to be learned; low values will ensure that more interpretable trees are found; high values may lead to better training set accuracy, but can also lead to overfitting;
* ``min_sup``, which specifies the minimum number of examples in the training data that every leaf should cover; low values may lead to predictions that are based on too small amounts of data.

Other parameters that may be useful to tune are:

* ``time_limit``, which indicates the maximum amount of time the algorithm is allowed to run; the algorithm will be interrupted when the runtime is exceeded, and the best tree found within the allocated time will be returned. The default value is ``0``, in which case no limit on runtime is imposed.
* ``max_error``, which will direct the search algorithm to only find trees with an error lower than ``max_error``. For instance, if a decision tree has already been found using another algorithm (such as a heuristic algorithm), specifying this parameter could direct DL8.5 to only find trees that are better than the tree found using this other algorithm.



.. In addition, scikit-learn provides a mixin, i.e. :class:`sklearn.base.ClassifierMixin`, which implements the ``score`` method which computes the accuracy score of the predictions.

.. One can import this mixin as::

..    >>> from sklearn.base import ClassifierMixin
.. The method ``fit`` gets ``X`` and ``y``
.. as input and should return ``self``. It should implement the ``predict``
.. function which should output the class inferred by the classifier.


Note that DL8.5 currently only works on boolean data; if the input data is not boolean, the data would have to made boolean first. 

The following example illustrates how to use our classifier within a scikit-learn pipeline to learn an optimal decision tree classifier::

    >>> dataset = np.genfromtxt("binary_dataset.txt", delimiter=' ')
    >>> X = dataset[:, 1:]
    >>> y = dataset[:, 0]
    >>> pipe = make_pipeline(DL85Classifier())
    >>> pipe.fit(X, y)  # doctest: +ELLIPSIS
    Pipeline(...)


Then, you can call ``predict`` to classify these examples::

    >>> pipe.predict(X)  # doctest: +ELLIPSIS
    array([...])



:class:`DL85Classifier` also inherits from 
:class:`sklearn.base.ClassifierMixin`. This allows the use of the ``score`` method to calculate 
the accuracy of the classifier::

    >>> pipe.score(X, y)  # doctest: +ELLIPSIS
    0...

Other predictors
~~~~~~~~~~~~~~~~

Classification trees are one example of decision trees. In their more general form, decision trees
may also predict other structures in their leafs. To support such other learning tasks, the ``DL85Predictor`` class
is provided. In contrast to the ``DL85Classifier`` class, the ``DL85Predictor`` class does not require the specification 
of a vector ``y`` consisting of class labels in the ``fit`` function, and allows for the specification of 
other optimisation criteria than error.

An example of another type of decision tree is the Predictive Clustering tree. In a Predictive Clustering tree
the leafs of the tree correspond to clusters in the unlabeled training data. The quality of the tree 
is determined by the quality of the clusters in the leafs of the tree. Standard measures can be used to
evaluate the quality of the clusters, such as `within-cluster sum of squares  <https://en.wikipedia.org/wiki/K-means_clustering>`_. The predictions in the leafs of the tree then correspond to the centroids of the clusters.

Using DL8.5's ``DL85Predictor`` class, this clustering task can be solved by specifying an error function 
that evaluates the quality of clusters in the leafs. The full code is given below::

    import numpy as np
    from sklearn.neighbors import DistanceMetric
    from pydl85 import DL85Predictor

    dataset = np.genfromtxt("../datasets/anneal.txt", delimiter=' ')
    X = dataset[:, 1:]
    X = X.astype('int32')

    eucl_dist = DistanceMetric.get_metric('euclidean')

    def error(tids):
        X_subset = X[list(tids),:]
        centroid = np.mean(X_subset, axis=0)
        distances = eucl_dist.pairwise(X_subset, [centroid])
        return float(sum(distances))

    def leaf_value(tids):
        return np.mean(X.take(list(tids)))

    clf = DL85Predictor(max_depth=3, min_sup=5, error_function=error, leaf_value_function=leaf_value, time_limit=600)

    clf.fit(X)
    predicted = clf.predict(X)

The ``error`` function in this example has one argument ``tids``. The ``DL85Predictor`` class will call 
this function for each candidate leaf, where ``tids`` lists the identifiers of the training examples that would be part of that leaf. The ``error`` function in this example calculates the mean of the training examples in this list,
and then calculates the euclidian distance of each example in the list towards the mean. The sum of these 
distance is returned as the score for the candidate leaf.

The ``DL85Predictor`` class is initialized with the function that needs to be called to evaluate the quality of the 
leafs. 

Other tree learning tasks can be specified by providing an alternative implementation of the ``error`` function. 
Note that in this example, the ``fit`` function is called on the matrix ``X``, and the error function also operates
on the matrix ``X``. This is not necessary; the only required to the error function is that for a given list 
of row identifiers (coming from the matrix ``X``) it can return a quality score. 

In this example, we call the ``predict`` function. For each example given in the parameter of the ``predict`` function,
``DL85Predictor`` will traverse the tree to determine the prediction specified in the corresponding leaf of the tree. 
This prediction is provided by the ``leaf_value`` function. The ``leaf_value`` function will be called at the 
end of the training process to fill in the predictions in the leafs. Also this function will receive a list of 
identifiers in the training data ``X`` in order to calculate the prediction. In this example, the prediction 
corresponds to the mean.

In principle, classification trees can also be learned using the ``DL85Predictor`` class. The following
error function can be used::

    def error(tids):
        classes, supports = np.unique(y.take(list(tids)), return_counts=True)
        maxindex = np.argmax(supports)
        return sum(supports) - supports[maxindex]

Here ``y`` consists of the labels of the examples in ``X``. We use standard NumPy functions to count the 
number of examples in each class, determine the majority class and finally calculate the error based on this.

However, learning classification trees in this manner is in practice slower than by using the ``DL85Classifier`` class.
The ``DL85Classifier`` class calculates error using optimized code written in C++, instead of using Python.

For supervised data with class labels, a supplementary interface is provided for writing error functions, illustrated
in this example::

    def error(sup_iter):
        supports = list(sup_iter)
        maxindex = np.argmax(supports)
        return sum(supports) - supports[maxindex], maxindex


    clf = DL85Classifier(max_depth=2, fast_error_function=error, time_limit=600)

In this example, a ``fast_error_function`` is specified. If this function is specified, ``DL85Classifier`` 
will call the user-specified function with as argument an iterator over  the 
numbers of examples in each class.

The advantage of this variation is that the calculation of the class distribution is done using optimized C++ code;
the Python code does not have to traverse the data. Only the final calculation of the score is done in Python.
This functionality is useful for instance if a different weight should be given to each class.

Finally, we provide a built-in implementation of predictive clustering in the ``DL85Cluster`` class. 
Using this class, the user does not have to write the example code written above.




